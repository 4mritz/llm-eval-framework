from typing import Optional

from .base import BaseLLM, LLMResponse

_PRICING: dict[str, dict[str, float]] = {
    "claude-3-5-sonnet-20241022": {"input": 3.0,  "output": 15.0},
    "claude-3-5-haiku-20241022":  {"input": 0.80, "output": 4.0},
    "claude-3-opus-20240229":     {"input": 15.0, "output": 75.0},
    "claude-3-haiku-20240307":    {"input": 0.25, "output": 1.25},
}


class AnthropicLLM(BaseLLM):
    """
    Adapter for Anthropic's Messages API.

    Requires the `anthropic` package and the ANTHROPIC_API_KEY environment
    variable (or an explicit api_key argument).
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-haiku-20241022",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc

        self._client = anthropic.Anthropic(api_key=api_key)
        self._pricing = _PRICING.get(model_name)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        create_kwargs: dict = dict(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        if system_prompt:
            create_kwargs["system"] = system_prompt

        message = self._client.messages.create(**create_kwargs)

        text = "".join(
            block.text for block in message.content if hasattr(block, "text")
        ).strip()

        usage = message.usage
        return LLMResponse(
            text=text,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            model_name=self.model_name,
            raw=message.model_dump(),
        )

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
        if not self._pricing:
            return None
        return (
            prompt_tokens * self._pricing["input"] / 1_000_000
            + completion_tokens * self._pricing["output"] / 1_000_000
        )
