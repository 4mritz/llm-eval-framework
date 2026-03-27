"""Unit tests for the metrics module."""
import pytest
from src.metrics.metrics import (
    compute_factuality_metrics,
    compute_stability,
    jaccard_similarity,
    SummarizationMetrics,
)
from src.modules.nli_verifier import ClaimVerification, Verdict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_verifications(verdicts: list[str]) -> list[ClaimVerification]:
    return [
        ClaimVerification(claim=f"Claim {i}", verdict=Verdict(v))
        for i, v in enumerate(verdicts)
    ]


# ---------------------------------------------------------------------------
# factuality metrics
# ---------------------------------------------------------------------------

class TestFactualityMetrics:
    def test_all_supported(self):
        verifs = _make_verifications(["SUPPORTED"] * 5)
        m = compute_factuality_metrics(verifs)
        assert m["fact_score"] == pytest.approx(1.0)
        assert m["hallucination_rate"] == pytest.approx(0.0)
        assert m["n_supported"] == 5

    def test_all_refuted(self):
        verifs = _make_verifications(["REFUTED"] * 4)
        m = compute_factuality_metrics(verifs)
        assert m["fact_score"] == pytest.approx(0.0)
        assert m["hallucination_rate"] == pytest.approx(1.0)
        assert m["refuted_rate"] == pytest.approx(1.0)

    def test_mixed(self):
        verifs = _make_verifications(
            ["SUPPORTED", "SUPPORTED", "REFUTED", "UNVERIFIABLE"]
        )
        m = compute_factuality_metrics(verifs)
        assert m["fact_score"] == pytest.approx(0.5)
        assert m["hallucination_rate"] == pytest.approx(0.5)
        assert m["n_claims"] == 4

    def test_empty(self):
        m = compute_factuality_metrics([])
        assert m["fact_score"] == 0.0
        assert m["n_claims"] == 0


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

class TestJaccardSimilarity:
    def test_identical_sets(self):
        s = {"hello world", "foo bar"}
        assert jaccard_similarity(s, s) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        a = {"alpha beta"}
        b = {"gamma delta"}
        assert jaccard_similarity(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = {"the quick brown fox"}
        b = {"the quick lazy dog"}
        score = jaccard_similarity(a, b)
        assert 0.0 < score < 1.0

    def test_empty_sets(self):
        assert jaccard_similarity(set(), set()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

class TestStability:
    def test_identical_summaries(self):
        summaries = ["The cat sat on the mat."] * 3
        stab = compute_stability(summaries)
        assert stab.mean_jaccard == pytest.approx(1.0)
        assert stab.n_runs == 3

    def test_single_summary(self):
        stab = compute_stability(["Only one run."])
        assert stab.n_runs == 1
        assert stab.mean_jaccard == pytest.approx(0.0)

    def test_divergent_summaries(self):
        summaries = [
            "Alpha beta gamma delta epsilon",
            "Zeta eta theta iota kappa lambda",
            "Mu nu xi omicron pi rho",
        ]
        stab = compute_stability(summaries)
        assert stab.mean_jaccard < 0.1
        assert stab.std_jaccard >= 0.0
