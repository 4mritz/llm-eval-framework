"""
Integration test: runs the full pipeline end-to-end with a mock LLM
and mock verifier, so no external services are needed.
"""
import pytest
from unittest.mock import MagicMock

from src.llms.base import BaseLLM, LLMResponse
from src.modules.nli_verifier import ClaimVerification, Verdict
from src.pipeline.orchestrator import EvaluationPipeline


class MockLLM(BaseLLM):
    """Returns deterministic responses for testing."""

    SUMMARY = (
        "JWST launched on December 25, 2021. "
        "It cost approximately $10 billion. "
        "The telescope orbits at L2, 1.5 million km from Earth."
    )
    CLAIMS = '["JWST launched on December 25, 2021.", "$10 billion cost.", "Orbits at L2."]'
    FACTS  = '["JWST launched December 25, 2021.", "Cost $10 billion.", "L2 is 1.5 million km away.", "18 mirror segments."]'

    def __init__(self, response_text: str = SUMMARY):
        super().__init__(model_name="mock-llm")
        self._response_text = response_text
        self._call_count = 0

    def generate(self, prompt, system_prompt=None, temperature=0.3, max_tokens=1024, **kwargs):
        self._call_count += 1
        # Return different content based on prompt context
        if "Extract every distinct" in prompt or "Extract all important" in prompt:
            text = self.CLAIMS if "summary" in prompt.lower() or "Extract every" in prompt else self.FACTS
        else:
            text = self._response_text
        return LLMResponse(
            text=text,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model_name=self.model_name,
        )


class MockVerifier:
    def verify(self, claims, source_text):
        return [
            ClaimVerification(claim=c, verdict=Verdict.SUPPORTED, confidence=0.95)
            for c in claims
        ]


SOURCE = (
    "The James Webb Space Telescope launched December 25 2021 at a cost of $10 billion. "
    "It orbits at the second Lagrange point 1.5 million km from Earth. "
    "It has 18 gold-coated mirror segments."
)


class TestFullPipeline:
    def setup_method(self):
        self.llm = MockLLM()
        self.verifier = MockVerifier()
        self.pipeline = EvaluationPipeline(
            summarizer_llm=self.llm,
            extractor_llm=self.llm,
            verifier=self.verifier,
        )

    def test_pipeline_returns_evaluation_result(self):
        result = self.pipeline.run(SOURCE)
        assert result.model_name == "mock-llm"
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_metrics_are_populated(self):
        result = self.pipeline.run(SOURCE)
        m = result.metrics
        assert "fact_score" in m
        assert "hallucination_rate" in m
        assert "recall" in m
        assert "latency_seconds" in m
        assert m["latency_seconds"] >= 0.0

    def test_fact_score_is_1_when_all_supported(self):
        result = self.pipeline.run(SOURCE)
        # All claims are SUPPORTED by our mock verifier
        assert result.metrics["fact_score"] == pytest.approx(1.0)
        assert result.metrics["hallucination_rate"] == pytest.approx(0.0)

    def test_to_dict_is_json_serialisable(self):
        import json
        result = self.pipeline.run(SOURCE)
        serialised = json.dumps(result.to_dict())
        assert len(serialised) > 0

    def test_claims_and_verifications_match(self):
        result = self.pipeline.run(SOURCE)
        assert len(result.verifications) == len(result.claims)
