import json
import re
from pathlib import Path
from typing import Optional

from src.llms.base import BaseLLM


_DEFAULT_SYSTEM = (
    "You are a precise information-extraction assistant. "
    "Return only valid JSON arrays — no commentary, no markdown fences."
)

_DEFAULT_PROMPT = """\
Extract every distinct, atomic, verifiable factual claim from the text below.

Rules:
- Each claim must be a single, self-contained statement.
- Do NOT merge multiple facts into one claim.
- Do NOT invent or infer facts not explicitly stated.
- Output ONLY a JSON array of strings.

TEXT:
{text}

OUTPUT (JSON array only):"""


def _parse_claim_list(raw: str) -> list[str]:
    """Robustly extract a JSON array from the model output."""
    # strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(c).strip() for c in data if str(c).strip()]
    except json.JSONDecodeError:
        pass

    # fallback: extract lines that look like list items
    lines = [
        re.sub(r'^[\-\*\d\.\s"\']+', "", line).strip().rstrip('",')
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith("[") and not line.strip().startswith("]")
    ]
    return [ln for ln in lines if len(ln) > 10]


class ClaimExtractor:
    """
    Decomposes a summary (or any text) into atomic, verifiable claims.

    Using a separate LLM call for extraction decouples the summarizer
    from the evaluator and reduces self-consistency bias.
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompt_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM
        if prompt_path and Path(prompt_path).exists():
            self._template = Path(prompt_path).read_text(encoding="utf-8")
        else:
            self._template = _DEFAULT_PROMPT

    def extract(self, text: str, temperature: float = 0.1, max_tokens: int = 1024) -> list[str]:
        prompt = self._template.format(text=text.strip())
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _parse_claim_list(response.text)
