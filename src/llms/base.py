from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_name: str = ""
    raw: Optional[dict] = field(default=None, repr=False)

    @property
    def cost_estimate_usd(self) -> Optional[float]:
        """Override in subclasses that have pricing info."""
        return None


class BaseLLM(ABC):
    """
    Abstract interface for all LLM adapters.

    Every adapter must implement `generate`, accepting a prompt string and
    returning a normalized LLMResponse.  Optional kwargs (temperature, max_tokens,
    system_prompt) should be forwarded to the underlying API where supported.
    """

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.default_params = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"
