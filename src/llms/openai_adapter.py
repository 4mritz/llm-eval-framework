from typing import Optional

from .base import BaseLLM, LLMResponse

# Pricing per 1M tokens as of mid-2024; update as needed.
_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":            {"input": 5.0,   "output": 15.0},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":       {"input": 10.0,  "output": 30.0},
    "gpt-3.5-turbo":     {"input": 0.50,  "output": 1.50},
}


class OpenAILLM(BaseLLM):
    """
    Adapter for OpenAI's Chat Completions API.

    Requires the `openai` package and the OPENAI_API_KEY environment variable
    (or an explicit api_key argument).
    """

    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        self._client = OpenAI(api_key=api_key)  # falls back to OPENAI_API_KEY env var
        self._pricing = _PRICING.get(model_name)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        usage = completion.usage
        text = completion.choices[0].message.content.strip()

        return LLMResponse(
            text=text,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model_name=self.model_name,
            raw=completion.model_dump(),
        )

    @property
    def cost_estimate_usd(self) -> Optional[float]:
        return None  # computed per-response; see _compute_cost

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        if not self._pricing:
            return None
        return (
            prompt_tokens * self._pricing["input"] / 1_000_000
            + completion_tokens * self._pricing["output"] / 1_000_000
        )
