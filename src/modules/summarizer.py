from pathlib import Path
from typing import Optional

from src.llms.base import BaseLLM, LLMResponse


def _load_prompt(path: Optional[str], fallback: str) -> str:
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    return fallback


_DEFAULT_SYSTEM = (
    "You are an expert summarization assistant. "
    "Produce concise, accurate, and faithful summaries."
)

_DEFAULT_PROMPT = """\
Summarize the following text. Be factually faithful — do not add information \
not present in the source. Aim for {target_words} words.

TEXT:
{text}

SUMMARY:"""


class Summarizer:
    """
    Wraps an LLM to produce a summary of arbitrary input text.

    Args:
        llm:            Any BaseLLM adapter.
        prompt_path:    Optional path to a custom prompt template file.
        system_prompt:  System instruction override.
        target_words:   Soft word-count target communicated to the model.
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompt_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        target_words: int = 150,
    ):
        self.llm = llm
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM
        self.target_words = target_words
        self._template = _load_prompt(prompt_path, _DEFAULT_PROMPT)

    def summarize(self, text: str, temperature: float = 0.3, max_tokens: int = 512) -> LLMResponse:
        prompt = self._template.format(
            text=text.strip(),
            target_words=self.target_words,
        )
        return self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
