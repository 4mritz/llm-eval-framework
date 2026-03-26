"""
NLI-based claim verification.

Each claim is classified as:
  - SUPPORTED   : the source text entails the claim
  - REFUTED     : the source text contradicts the claim
  - UNVERIFIABLE: the source text neither confirms nor denies the claim

Two backends are provided:
  1. TransformersNLIVerifier  – fast, local, uses a cross-encoder NLI model
  2. LLMNLIVerifier           – uses a prompt on any BaseLLM (slower, more flexible)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.llms.base import BaseLLM


class Verdict(str, Enum):
    SUPPORTED    = "SUPPORTED"
    REFUTED      = "REFUTED"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass
class ClaimVerification:
    claim: str
    verdict: Verdict
    confidence: float = 1.0  # meaningful for TransformersNLIVerifier
    explanation: str = ""    # meaningful for LLMNLIVerifier


# ---------------------------------------------------------------------------
# Backend 1: transformer cross-encoder (fast, local)
# ---------------------------------------------------------------------------

class TransformersNLIVerifier:
    """
    Uses a pretrained NLI cross-encoder to verify each claim against the
    full source text (treated as the premise).

    Default model: facebook/bart-large-mnli (good balance of speed/accuracy)
    Alternatives:  cross-encoder/nli-deberta-v3-large (more accurate, slower)
    """

    LABEL_MAP = {
        "entailment":    Verdict.SUPPORTED,
        "contradiction": Verdict.REFUTED,
        "neutral":       Verdict.UNVERIFIABLE,
    }

    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: str = "cpu"):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("pip install transformers torch") from exc

        self._pipe = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )

    def verify(self, claims: list[str], source_text: str) -> list[ClaimVerification]:
        results = []
        for claim in claims:
            out = self._pipe(
                claim,
                candidate_labels=["entailment", "contradiction", "neutral"],
                hypothesis_template="{}",
                multi_label=False,
            )
            top_label = out["labels"][0]
            top_score = out["scores"][0]
            verdict = self.LABEL_MAP.get(top_label.lower(), Verdict.UNVERIFIABLE)
            results.append(
                ClaimVerification(
                    claim=claim,
                    verdict=verdict,
                    confidence=top_score,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Backend 2: LLM-based (flexible, works with any adapter)
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You are a rigorous fact-checking assistant. "
    "Return only one of: SUPPORTED, REFUTED, or UNVERIFIABLE."
)

_LLM_PROMPT = """\
SOURCE TEXT:
{source}

CLAIM:
{claim}

Does the source text support, refute, or neither confirm nor deny the claim?

Answer with exactly one word on the first line:
  SUPPORTED    – the source explicitly supports the claim
  REFUTED      – the source contradicts the claim
  UNVERIFIABLE – the source says nothing about the claim

Then, on the next line, give a one-sentence reason.

VERDICT:"""


def _parse_llm_verdict(raw: str) -> tuple[Verdict, str]:
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    verdict_str = lines[0].upper() if lines else ""
    explanation = lines[1] if len(lines) > 1 else ""

    for v in Verdict:
        if v.value in verdict_str:
            return v, explanation
    return Verdict.UNVERIFIABLE, explanation


class LLMNLIVerifier:
    """
    Uses a language model to classify each claim against the source text.

    Slower than a cross-encoder but works without GPU and can leverage
    any of the registered LLM adapters (useful for API-based models).
    """

    def __init__(self, llm: BaseLLM, source_truncate_chars: int = 4000):
        self.llm = llm
        self.source_truncate_chars = source_truncate_chars

    def verify(self, claims: list[str], source_text: str) -> list[ClaimVerification]:
        truncated_source = source_text[: self.source_truncate_chars]
        results = []
        for claim in claims:
            prompt = _LLM_PROMPT.format(source=truncated_source, claim=claim)
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=_LLM_SYSTEM,
                temperature=0.0,
                max_tokens=128,
            )
            verdict, explanation = _parse_llm_verdict(response.text)
            results.append(
                ClaimVerification(
                    claim=claim,
                    verdict=verdict,
                    explanation=explanation,
                )
            )
        return results
