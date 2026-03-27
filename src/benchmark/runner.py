"""
BenchmarkRunner: evaluates multiple models on the same document(s).

Usage:
    runner = BenchmarkRunner(
        models=[ollama_llm, openai_llm],
        extractor_llm=ollama_llm,
        verifier=verifier,
        output_dir="outputs/benchmark_run_001",
    )
    report = runner.run(source_text="...")
    runner.save_report(report)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

from src.llms.base import BaseLLM
from src.modules.nli_verifier import LLMNLIVerifier, TransformersNLIVerifier
from src.pipeline.orchestrator import EvaluationPipeline, EvaluationResult
from src.metrics.metrics import compute_stability

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkReport:
    run_id: str
    timestamp: str
    results: list[dict]
    comparison_table: list[dict]
    stability_results: dict  # model_name → StabilityMetrics

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")
        logger.info("Benchmark report saved to %s", path)


class BenchmarkRunner:
    """
    Runs the evaluation pipeline over a list of models and aggregates results.

    Args:
        models:         List of LLM adapters to benchmark.
        extractor_llm:  LLM used for extraction (shared across models to
                        keep the extraction process constant).
        verifier:       NLI backend (shared for consistency).
        output_dir:     Directory where per-model JSON results are written.
        n_runs:         Number of times to run each model (for stability).
        prompt_paths:   Optional dict of custom prompt file paths.
    """

    _METRIC_KEYS = [
        "fact_score",
        "hallucination_rate",
        "recall",
        "omission_rate",
        "latency_seconds",
        "total_tokens",
        "n_claims",
    ]

    def __init__(
        self,
        models: list[BaseLLM],
        extractor_llm: BaseLLM,
        verifier: Union[LLMNLIVerifier, TransformersNLIVerifier],
        output_dir: str = "outputs",
        n_runs: int = 1,
        prompt_paths: Optional[dict] = None,
    ):
        self.models = models
        self.extractor_llm = extractor_llm
        self.verifier = verifier
        self.output_dir = Path(output_dir)
        self.n_runs = n_runs
        self.prompt_paths = prompt_paths or {}

    def run(
        self,
        source_text: str,
        document_id: str = "doc_001",
        metadata: Optional[dict] = None,
    ) -> BenchmarkReport:
        run_id = f"benchmark_{document_id}_{int(time.time())}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        all_results: list[EvaluationResult] = []
        summaries_by_model: dict[str, list[str]] = {}

        for model in self.models:
            logger.info("=== Benchmarking model: %s ===", model.model_name)
            model_summaries: list[str] = []

            for run_idx in range(self.n_runs):
                logger.info("  Run %d/%d", run_idx + 1, self.n_runs)
                pipeline = EvaluationPipeline(
                    summarizer_llm=model,
                    extractor_llm=self.extractor_llm,
                    verifier=self.verifier,
                    prompt_paths=self.prompt_paths,
                )
                result = pipeline.run(
                    source_text,
                    metadata={**(metadata or {}), "run": run_idx, "document_id": document_id},
                )
                all_results.append(result)
                model_summaries.append(result.summary)

                # Save individual result
                out_path = (
                    self.output_dir / run_id / f"{model.model_name}_run{run_idx}.json"
                )
                result.save(out_path)

            summaries_by_model[model.model_name] = model_summaries

        # Stability across runs (only meaningful when n_runs > 1)
        stability_results = {}
        for model_name, summaries in summaries_by_model.items():
            if len(summaries) >= 2:
                stab = compute_stability(summaries)
                stability_results[model_name] = stab.to_dict()

        comparison_table = self._build_comparison_table(all_results)

        report = BenchmarkReport(
            run_id=run_id,
            timestamp=timestamp,
            results=[r.to_dict() for r in all_results],
            comparison_table=comparison_table,
            stability_results=stability_results,
        )

        report.save(self.output_dir / run_id / "report.json")
        self._print_comparison(comparison_table)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_comparison_table(self, results: list[EvaluationResult]) -> list[dict]:
        """Aggregate per-model average metrics across runs."""
        from collections import defaultdict

        buckets: dict[str, list[EvaluationResult]] = defaultdict(list)
        for r in results:
            buckets[r.model_name].append(r)

        table = []
        for model_name, runs in buckets.items():
            row: dict = {"model": model_name, "n_runs": len(runs)}
            for key in self._METRIC_KEYS:
                values = [r.metrics.get(key, 0) for r in runs]
                row[key] = round(sum(values) / len(values), 4)
            table.append(row)

        # Sort by fact_score descending
        table.sort(key=lambda x: x.get("fact_score", 0), reverse=True)
        return table

    def _print_comparison(self, table: list[dict]) -> None:
        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON")
        print("=" * 80)
        header = f"{'Model':<35} {'FactScore':>9} {'Halluc%':>8} {'Recall':>8} {'Latency':>9} {'Tokens':>8}"
        print(header)
        print("-" * 80)
        for row in table:
            print(
                f"{row['model']:<35} "
                f"{row.get('fact_score', 0):>9.3f} "
                f"{row.get('hallucination_rate', 0):>8.3f} "
                f"{row.get('recall', 0):>8.3f} "
                f"{row.get('latency_seconds', 0):>9.2f}s "
                f"{row.get('total_tokens', 0):>8}"
            )
        print("=" * 80 + "\n")
