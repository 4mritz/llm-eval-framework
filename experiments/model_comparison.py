#!/usr/bin/env python3
"""
experiments/model_comparison.py

Runs a controlled multi-model benchmark across all documents in data/samples/.
Results are saved to outputs/experiments/model_comparison/.

Usage:
    python experiments/model_comparison.py
    python experiments/model_comparison.py --config configs/benchmark.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llms import get_llm
from src.modules.nli_verifier import TransformersNLIVerifier
from src.benchmark.runner import BenchmarkRunner
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--data-dir", default="data/samples")
    parser.add_argument("--output-dir", default="outputs/experiments/model_comparison")
    args = parser.parse_args()

    setup_logging("INFO")
    cfg = load_config(args.config)

    models = [
        get_llm(m["provider"], model_name=m["model_name"])
        for m in cfg["models"]
    ]
    extractor_llm = get_llm(
        cfg["extractor"]["provider"],
        model_name=cfg["extractor"]["model_name"],
    )
    verifier = TransformersNLIVerifier(
        model_name=cfg["verifier"]["model_name"],
        device=cfg["verifier"].get("device", "cpu"),
    )

    runner = BenchmarkRunner(
        models=models,
        extractor_llm=extractor_llm,
        verifier=verifier,
        output_dir=args.output_dir,
        n_runs=cfg["evaluation"].get("n_runs", 1),
        prompt_paths=cfg.get("prompts"),
    )

    data_dir = Path(args.data_dir)
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    all_reports = []
    for txt_path in txt_files:
        print(f"\n{'='*60}\nDocument: {txt_path.name}\n{'='*60}")
        source_text = txt_path.read_text(encoding="utf-8").strip()
        report = runner.run(source_text, document_id=txt_path.stem)
        all_reports.append(report.to_dict())

    # Save aggregate summary
    summary_path = Path(args.output_dir) / "aggregate_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print(f"\nAggregate summary saved to {summary_path}")


if __name__ == "__main__":
    main()
