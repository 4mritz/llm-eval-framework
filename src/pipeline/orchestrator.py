"""
EvaluationPipeline: the central orchestrator.

Coordinates the full evaluation flow:
  source text
    ↓
  Summarizer          → summary text + token usage
    ↓
  ClaimExtractor      → list[str] (atomic claims from summary)
  ReferenceExtractor  → list[str] (ground-truth facts from source)
    ↓
  NLIVerifier         → list[ClaimVerification]
    ↓
  Metrics             → SummarizationMetrics
    ↓
  EvaluationResult    (structured JSON-serialisable output)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Union

from src.llms.base import BaseLLM
from src.modules.summarizer import Summarizer
from src.modules.claim_extractor import ClaimExtractor
from src.modules.fact_extractor import ReferenceFactExtractor
from src.modules.nli_verifier import (
    ClaimVerification,
    LLMNLIVerifier,
    TransformersNLIVerifier,
    Verdict,
)
from src.metrics.metrics import (
    SummarizationMetrics,
    Timer,
    compute_factuality_metrics,
    compute_recall_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    model_name: str
    source_text: str
    summary: str
    claims: list[str]
    reference_facts: list[str]
    verifications: list[dict]          # ClaimVerification serialised
    supported_claims: list[str]
    refuted_claims: list[str]
    unverifiable_claims: list[str]
    metrics: dict
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")
        logger.info("Result saved to %s", path)


class EvaluationPipeline:
    """
    Pluggable evaluation pipeline.

    Args:
        summarizer_llm:     LLM used to produce the summary.
        extractor_llm:      LLM used for claim and fact extraction
                            (can be the same as summarizer_llm).
        verifier:           NLI backend — LLMNLIVerifier or TransformersNLIVerifier.
        summarize_kwargs:   Forwarded to Summarizer.summarize().
        extract_kwargs:     Forwarded to ClaimExtractor.extract().
    """

    def __init__(
        self,
        summarizer_llm: BaseLLM,
        extractor_llm: BaseLLM,
        verifier: Union[LLMNLIVerifier, TransformersNLIVerifier],
        prompt_paths: Optional[dict[str, str]] = None,
        summarize_kwargs: Optional[dict] = None,
        extract_kwargs: Optional[dict] = None,
    ):
        paths = prompt_paths or {}

        self.summarizer = Summarizer(
            llm=summarizer_llm,
            prompt_path=paths.get("summarize"),
        )
        self.claim_extractor = ClaimExtractor(
            llm=extractor_llm,
            prompt_path=paths.get("extract_claims"),
        )
        self.reference_extractor = ReferenceFactExtractor(
            llm=extractor_llm,
            prompt_path=paths.get("extract_facts"),
        )
        self.verifier = verifier

        self._summarize_kwargs = summarize_kwargs or {}
        self._extract_kwargs = extract_kwargs or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, source_text: str, metadata: Optional[dict] = None) -> EvaluationResult:
        """Run the full pipeline on a source document."""

        logger.info("Starting pipeline for model: %s", self.summarizer.llm.model_name)

        # 1. Summarize
        with Timer() as t:
            summary_response = self.summarizer.summarize(source_text, **self._summarize_kwargs)
        summary = summary_response.text
        latency = t.elapsed
        logger.info("Summary generated in %.2fs (%d tokens)", latency, summary_response.total_tokens)

        # 2. Extract claims from the summary
        logger.info("Extracting claims from summary…")
        claims = self.claim_extractor.extract(summary, **self._extract_kwargs)
        logger.info("  → %d claims extracted", len(claims))

        # 3. Extract reference facts from the source
        logger.info("Extracting reference facts from source…")
        reference_facts = self.reference_extractor.extract(source_text)
        logger.info("  → %d reference facts extracted", len(reference_facts))

        # 4. NLI verification: each summary claim vs. source
        logger.info("Verifying %d claims via NLI…", len(claims))
        verifications: list[ClaimVerification] = self.verifier.verify(claims, source_text)

        # 5. Compute factuality metrics
        factuality = compute_factuality_metrics(verifications)

        # 6. Compute recall (reference facts covered by summary claims)
        recall_metrics = compute_recall_metrics(
            summary_claims=claims,
            reference_facts=reference_facts,
            verifier=self.verifier,
            source_text=source_text,
        )

        # 7. Merge metrics
        metrics = SummarizationMetrics(
            fact_score=factuality["fact_score"],
            hallucination_rate=factuality["hallucination_rate"],
            refuted_rate=factuality["refuted_rate"],
            unverifiable_rate=factuality["unverifiable_rate"],
            recall=recall_metrics["recall"],
            omission_rate=recall_metrics["omission_rate"],
            latency_seconds=latency,
            prompt_tokens=summary_response.prompt_tokens,
            completion_tokens=summary_response.completion_tokens,
            total_tokens=summary_response.total_tokens,
            n_claims=factuality["n_claims"],
            n_supported=factuality["n_supported"],
            n_refuted=factuality["n_refuted"],
            n_unverifiable=factuality["n_unverifiable"],
            n_reference_facts=recall_metrics["n_reference_facts"],
            n_recalled_facts=recall_metrics["n_recalled_facts"],
        )

        # Partition claims by verdict for easy inspection
        def _filter(verdict: Verdict) -> list[str]:
            return [v.claim for v in verifications if v.verdict == verdict]

        return EvaluationResult(
            model_name=self.summarizer.llm.model_name,
            source_text=source_text,
            summary=summary,
            claims=claims,
            reference_facts=reference_facts,
            verifications=[
                {"claim": v.claim, "verdict": v.verdict.value, "explanation": v.explanation}
                for v in verifications
            ],
            supported_claims=_filter(Verdict.SUPPORTED),
            refuted_claims=_filter(Verdict.REFUTED),
            unverifiable_claims=_filter(Verdict.UNVERIFIABLE),
            metrics=metrics.to_dict(),
            latency_seconds=latency,
            prompt_tokens=summary_response.prompt_tokens,
            completion_tokens=summary_response.completion_tokens,
            total_tokens=summary_response.total_tokens,
            metadata=metadata or {},
        )
