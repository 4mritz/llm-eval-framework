#!/usr/bin/env python3
"""
LLM Summarization Evaluation Framework — CLI

Commands
--------
  run          Evaluate a single model on a text file
  benchmark    Evaluate multiple models and compare
  report       Pretty-print an existing result JSON

Examples
--------
  python run_pipeline.py run \\
      --provider ollama \\
      --model llama3.1:8b-instruct-q4_K_M \\
      --input data/samples/cnn_001.txt \\
      --config configs/default.yaml \\
      --output outputs/

  python run_pipeline.py benchmark \\
      --config configs/benchmark.yaml \\
      --input data/samples/cnn_001.txt \\
      --output outputs/

  python run_pipeline.py report --result outputs/my_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config
from src.utils.logging_setup import setup_logging
from src.llms import get_llm
from src.modules.nli_verifier import LLMNLIVerifier, TransformersNLIVerifier
from src.pipeline.orchestrator import EvaluationPipeline
from src.benchmark.runner import BenchmarkRunner


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_verifier(cfg: dict):
    vcfg = cfg["verifier"]
    if vcfg["backend"] == "transformers":
        return TransformersNLIVerifier(
            model_name=vcfg["model_name"],
            device=vcfg.get("device", "cpu"),
        )
    elif vcfg["backend"] == "llm":
        llm = get_llm(
            vcfg["provider"],
            model_name=vcfg["model_name"],
        )
        return LLMNLIVerifier(llm)
    else:
        raise ValueError(f"Unknown verifier backend: {vcfg['backend']!r}")


def _build_llm(cfg_section: dict):
    return get_llm(
        cfg_section["provider"],
        model_name=cfg_section["model_name"],
    )


def _read_input(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Input file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return p.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    overrides: dict = {}
    if args.provider:
        overrides.setdefault("llm", {})["provider"] = args.provider
    if args.model:
        overrides.setdefault("llm", {})["model_name"] = args.model

    cfg = load_config(args.config, overrides)
    setup_logging(args.log_level)

    source_text = _read_input(args.input)
    verifier = _build_verifier(cfg)
    summarizer_llm = _build_llm(cfg["llm"])
    extractor_llm = _build_llm(cfg["extractor"])

    pipeline = EvaluationPipeline(
        summarizer_llm=summarizer_llm,
        extractor_llm=extractor_llm,
        verifier=verifier,
        prompt_paths=cfg.get("prompts"),
        summarize_kwargs={
            "temperature": cfg["llm"].get("temperature", 0.3),
            "max_tokens": cfg["llm"].get("max_tokens", 512),
        },
    )

    result = pipeline.run(source_text)

    output_dir = Path(args.output or cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = summarizer_llm.model_name.replace("/", "_").replace(":", "-")
    out_path = output_dir / f"{safe_name}_result.json"
    result.save(out_path)

    print(f"\n✓ Result saved to {out_path}")
    _print_result_summary(result.to_dict())


# ---------------------------------------------------------------------------
# Subcommand: benchmark
# ---------------------------------------------------------------------------

def cmd_benchmark(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    setup_logging(args.log_level)

    source_text = _read_input(args.input)

    models_cfg: list[dict] = cfg.get("models", [cfg["llm"]])
    models = [get_llm(m["provider"], model_name=m["model_name"]) for m in models_cfg]

    extractor_llm = _build_llm(cfg["extractor"])
    verifier = _build_verifier(cfg)

    output_dir = args.output or cfg["output"]["dir"]
    runner = BenchmarkRunner(
        models=models,
        extractor_llm=extractor_llm,
        verifier=verifier,
        output_dir=output_dir,
        n_runs=cfg["evaluation"].get("n_runs", 1),
        prompt_paths=cfg.get("prompts"),
    )

    runner.run(source_text, document_id=Path(args.input).stem)


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    _print_result_summary(data)


def _print_result_summary(data: dict) -> None:
    m = data.get("metrics", {})
    print("\n" + "=" * 60)
    print(f"  MODEL : {data.get('model_name', '?')}")
    print("=" * 60)
    print(f"  FactScore       : {m.get('fact_score', 0):.3f}")
    print(f"  Halluc. rate    : {m.get('hallucination_rate', 0):.3f}")
    print(f"  Recall          : {m.get('recall', 0):.3f}")
    print(f"  Omission rate   : {m.get('omission_rate', 0):.3f}")
    print(f"  Latency         : {m.get('latency_seconds', 0):.2f}s")
    print(f"  Total tokens    : {m.get('total_tokens', 0)}")
    print(f"  Claims          : {m.get('n_claims', 0)} "
          f"({m.get('n_supported', 0)} supported, "
          f"{m.get('n_refuted', 0)} refuted, "
          f"{m.get('n_unverifiable', 0)} unverifiable)")
    print("=" * 60)
    print(f"\nSUMMARY:\n{data.get('summary', '')}\n")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline",
        description="LLM Summarization Evaluation Framework",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Evaluate a single model")
    p_run.add_argument("--provider", help="LLM provider (ollama|openai|anthropic|hf)")
    p_run.add_argument("--model", help="Model name/tag")
    p_run.add_argument("--input", required=True, help="Path to source text file")
    p_run.add_argument("--config", default="configs/default.yaml")
    p_run.add_argument("--output", help="Output directory (overrides config)")
    p_run.set_defaults(func=cmd_run)

    # --- benchmark ---
    p_bench = sub.add_parser("benchmark", help="Compare multiple models")
    p_bench.add_argument("--input", required=True, help="Path to source text file")
    p_bench.add_argument("--config", default="configs/benchmark.yaml")
    p_bench.add_argument("--output", help="Output directory")
    p_bench.set_defaults(func=cmd_benchmark)

    # --- report ---
    p_rep = sub.add_parser("report", help="Display an existing result JSON")
    p_rep.add_argument("--result", required=True, help="Path to result JSON file")
    p_rep.set_defaults(func=cmd_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
