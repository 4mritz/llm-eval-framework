#!/usr/bin/env python3
"""
experiments/prompt_ablation.py

Evaluates one model with multiple summarization prompt variants to measure
how prompt wording affects factual accuracy and recall.

Each prompt variant is defined in experiments/prompt_variants/ as .txt files.
Results are saved per-variant so they can be compared.

Usage:
    python experiments/prompt_ablation.py \\
        --input data/samples/jwst_article.txt \\
        --provider ollama \\
        --model llama3.1:8b-instruct-q4_K_M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llms import get_llm
from src.modules.nli_verifier import TransformersNLIVerifier
from src.pipeline.orchestrator import EvaluationPipeline
from src.utils.logging_setup import setup_logging


PROMPT_VARIANTS_DIR = Path(__file__).parent / "prompt_variants"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="llama3.1:8b-instruct-q4_K_M")
    parser.add_argument("--output-dir", default="outputs/experiments/prompt_ablation")
    args = parser.parse_args()

    setup_logging("INFO")

    source_text = Path(args.input).read_text(encoding="utf-8").strip()
    llm = get_llm(args.provider, model_name=args.model)
    verifier = TransformersNLIVerifier()

    variants = sorted(PROMPT_VARIANTS_DIR.glob("*.txt"))
    if not variants:
        # Fall back to the canonical prompt
        variants = [Path("prompts/summarize.txt")]

    ablation_results = []

    for variant_path in variants:
        print(f"\nRunning prompt variant: {variant_path.name}")
        pipeline = EvaluationPipeline(
            summarizer_llm=llm,
            extractor_llm=llm,
            verifier=verifier,
            prompt_paths={
                "summarize": str(variant_path),
                "extract_claims": "prompts/extract_claims.txt",
                "extract_facts": "prompts/extract_facts.txt",
            },
        )
        result = pipeline.run(source_text, metadata={"prompt_variant": variant_path.name})
        ablation_results.append({
            "prompt_variant": variant_path.name,
            "metrics": result.metrics,
            "summary": result.summary,
        })

        out_path = Path(args.output_dir) / f"{variant_path.stem}_result.json"
        result.save(out_path)

    # Print comparison table
    print("\n" + "=" * 70)
    print("PROMPT ABLATION RESULTS")
    print("=" * 70)
    header = f"{'Variant':<30} {'FactScore':>10} {'Recall':>8} {'Halluc%':>8}"
    print(header)
    print("-" * 70)
    for r in ablation_results:
        m = r["metrics"]
        print(
            f"{r['prompt_variant']:<30} "
            f"{m.get('fact_score', 0):>10.3f} "
            f"{m.get('recall', 0):>8.3f} "
            f"{m.get('hallucination_rate', 0):>8.3f}"
        )
    print("=" * 70)

    summary_path = Path(args.output_dir) / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(ablation_results, indent=2), encoding="utf-8")
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
