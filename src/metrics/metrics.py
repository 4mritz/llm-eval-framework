"""
Evaluation metrics for summarization quality.

All metric functions are pure (no side effects) and accept structured inputs
so they can be tested, compared, and reused independently of the pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.modules.nli_verifier import ClaimVerification, Verdict


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SummarizationMetrics:
    """Single-run metric snapshot for one model on one document."""

    # Factuality
    fact_score: float = 0.0           # supported / total claims
    hallucination_rate: float = 0.0   # (refuted + unverifiable) / total claims
    refuted_rate: float = 0.0         # refuted / total claims
    unverifiable_rate: float = 0.0    # unverifiable / total claims

    # Coverage
    recall: float = 0.0               # reference facts captured
    omission_rate: float = 0.0        # reference facts missed

    # Efficiency
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Optional[float] = None

    # Counts (useful for debugging)
    n_claims: int = 0
    n_supported: int = 0
    n_refuted: int = 0
    n_unverifiable: int = 0
    n_reference_facts: int = 0
    n_recalled_facts: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StabilityMetrics:
    """Aggregate stability across multiple runs on the same document."""
    mean_jaccard: float = 0.0
    min_jaccard: float = 0.0
    max_jaccard: float = 0.0
    std_jaccard: float = 0.0
    n_runs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core metric computations
# ---------------------------------------------------------------------------

def compute_factuality_metrics(
    verifications: list[ClaimVerification],
) -> dict[str, float | int]:
    """
    Compute FactScore, hallucination rate, and per-verdict rates.
    """
    n = len(verifications)
    if n == 0:
        return {
            "fact_score": 0.0,
            "hallucination_rate": 0.0,
            "refuted_rate": 0.0,
            "unverifiable_rate": 0.0,
            "n_claims": 0,
            "n_supported": 0,
            "n_refuted": 0,
            "n_unverifiable": 0,
        }

    n_supported    = sum(1 for v in verifications if v.verdict == Verdict.SUPPORTED)
    n_refuted      = sum(1 for v in verifications if v.verdict == Verdict.REFUTED)
    n_unverifiable = sum(1 for v in verifications if v.verdict == Verdict.UNVERIFIABLE)

    return {
        "fact_score":         n_supported / n,
        "hallucination_rate": (n_refuted + n_unverifiable) / n,
        "refuted_rate":       n_refuted / n,
        "unverifiable_rate":  n_unverifiable / n,
        "n_claims":           n,
        "n_supported":        n_supported,
        "n_refuted":          n_refuted,
        "n_unverifiable":     n_unverifiable,
    }


def compute_recall_metrics(
    summary_claims: list[str],
    reference_facts: list[str],
    verifier,
    source_text: str,
) -> dict[str, float | int]:
    """
    Recall: fraction of reference facts entailed by the summary.

    We verify each reference fact against the summary (reversed NLI direction).
    """
    if not reference_facts:
        return {"recall": 0.0, "omission_rate": 0.0, "n_reference_facts": 0, "n_recalled_facts": 0}

    summary_text = " ".join(summary_claims) if summary_claims else source_text
    verifications = verifier.verify(reference_facts, summary_text)
    n_recalled = sum(1 for v in verifications if v.verdict == Verdict.SUPPORTED)
    n = len(reference_facts)

    return {
        "recall":           n_recalled / n,
        "omission_rate":    1 - n_recalled / n,
        "n_reference_facts": n,
        "n_recalled_facts": n_recalled,
    }


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Token-level Jaccard similarity between two sets of sentences."""
    tokens_a = _tokenize_set(set_a)
    tokens_b = _tokenize_set(set_b)
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def _tokenize_set(sentences: set[str]) -> set[str]:
    import re
    tokens: set[str] = set()
    for s in sentences:
        tokens.update(re.findall(r"\w+", s.lower()))
    return tokens


def compute_stability(summaries: list[str]) -> StabilityMetrics:
    """
    Pairwise Jaccard similarity across multiple runs on the same input.

    A high mean_jaccard (> 0.7) indicates the model is consistent;
    a low value signals sensitivity to random sampling.
    """
    import itertools
    import statistics

    if len(summaries) < 2:
        return StabilityMetrics(n_runs=len(summaries))

    sets = [set(s.split()) for s in summaries]
    scores = [
        jaccard_similarity(a, b)
        for a, b in itertools.combinations(sets, 2)
    ]

    return StabilityMetrics(
        mean_jaccard=statistics.mean(scores),
        min_jaccard=min(scores),
        max_jaccard=max(scores),
        std_jaccard=statistics.stdev(scores) if len(scores) > 1 else 0.0,
        n_runs=len(summaries),
    )


# ---------------------------------------------------------------------------
# Latency context manager
# ---------------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer for measuring generation latency."""

    def __init__(self):
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
