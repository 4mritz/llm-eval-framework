#!/usr/bin/env python3
"""
experiments/failure_analysis.py

Loads existing benchmark output JSON files and performs a structured
failure analysis:

  1. Lists all hallucinated (refuted) claims across models
  2. Lists all omitted reference facts
  3. Computes per-model failure breakdown
  4. Exports a Markdown failure report

Usage:
    python experiments/failure_analysis.py \\
        --results-dir outputs/experiments/model_comparison/benchmark_doc_001_<timestamp>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_results(results_dir: Path) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(results_dir.glob("*.json"))
        if p.name != "report.json"
    ]


def analyse(results: list[dict]) -> dict:
    analysis: dict = defaultdict(lambda: {
        "hallucinated_claims": [],
        "unverifiable_claims": [],
        "omitted_facts": [],
        "fact_score": 0.0,
        "recall": 0.0,
    })

    for r in results:
        model = r["model_name"]
        analysis[model]["fact_score"] = r["metrics"].get("fact_score", 0)
        analysis[model]["recall"] = r["metrics"].get("recall", 0)
        analysis[model]["hallucinated_claims"].extend(r.get("refuted_claims", []))
        analysis[model]["unverifiable_claims"].extend(r.get("unverifiable_claims", []))

        # Omitted facts = reference facts not in supported claims
        supported = set(r.get("supported_claims", []))
        for fact in r.get("reference_facts", []):
            # A crude check — in a real system you'd re-run NLI
            if not any(fact.lower()[:30] in c.lower() for c in supported):
                analysis[model]["omitted_facts"].append(fact)

    return dict(analysis)


def render_markdown(analysis: dict) -> str:
    lines = ["# Failure Analysis Report\n"]
    for model, data in sorted(analysis.items()):
        lines.append(f"## {model}\n")
        lines.append(f"- **FactScore**: {data['fact_score']:.3f}")
        lines.append(f"- **Recall**: {data['recall']:.3f}\n")

        lines.append("### Hallucinated Claims (REFUTED)\n")
        if data["hallucinated_claims"]:
            for c in data["hallucinated_claims"]:
                lines.append(f"- {c}")
        else:
            lines.append("_None detected._")

        lines.append("\n### Unverifiable Claims\n")
        if data["unverifiable_claims"]:
            for c in data["unverifiable_claims"]:
                lines.append(f"- {c}")
        else:
            lines.append("_None detected._")

        lines.append("\n### Omitted Reference Facts\n")
        if data["omitted_facts"]:
            for f in data["omitted_facts"]:
                lines.append(f"- {f}")
        else:
            lines.append("_All reference facts covered._")

        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", default=None, help="Output .md path (optional)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)

    if not results:
        print(f"No result JSON files found in {results_dir}")
        return

    analysis = analyse(results)
    report_md = render_markdown(analysis)

    out_path = Path(args.output) if args.output else results_dir / "failure_analysis.md"
    out_path.write_text(report_md, encoding="utf-8")
    print(report_md)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
