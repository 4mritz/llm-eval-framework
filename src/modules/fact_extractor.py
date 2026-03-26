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
Extract every important, verifiable fact from the source text below.
These facts will serve as the ground-truth reference set for evaluating summaries.

Rules:
- One atomic fact per entry.
- Include named entities, quantities, dates, relationships, and key claims.
- Do NOT include opinions or unverifiable assertions.
- Output ONLY a JSON array of strings.

SOURCE TEXT:
{text}

OUTPUT (JSON array only):"""


def _parse_fact_list(raw: str) -> list[str]:
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(f).strip() for f in data if str(f).strip()]
    except json.JSONDecodeError:
        pass

    lines = [
        re.sub(r'^[\-\*\d\.\s"\']+', "", line).strip().rstrip('",')
        for line in raw.splitlines()
        if line.strip() and not line.strip().startswith(("[", "]"))
    ]
    return [ln for ln in lines if len(ln) > 10]


class ReferenceFactExtractor:
    """
    Extracts the complete set of facts from the source document.

    This reference set is used to compute recall and omission metrics —
    i.e. which important facts the summary failed to capture.
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

    def extract(self, source_text: str, temperature: float = 0.1, max_tokens: int = 2048) -> list[str]:
        prompt = self._template.format(text=source_text.strip())
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _parse_fact_list(response.text)
