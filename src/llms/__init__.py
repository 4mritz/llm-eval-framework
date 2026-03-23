"""
LLM adapter registry.

Usage:
    from src.llms import get_llm

    llm = get_llm("ollama", model_name="llama3.1:8b-instruct-q4_K_M")
    llm = get_llm("openai", model_name="gpt-4o-mini")
    llm = get_llm("anthropic", model_name="claude-3-5-haiku-20241022")
"""

from .base import BaseLLM, LLMResponse
from .ollama_adapter import OllamaLLM
from .openai_adapter import OpenAILLM
from .anthropic_adapter import AnthropicLLM
from .hf_adapter import HuggingFaceLLM

_REGISTRY: dict[str, type[BaseLLM]] = {
    "ollama":    OllamaLLM,
    "openai":    OpenAILLM,
    "anthropic": AnthropicLLM,
    "huggingface": HuggingFaceLLM,
    "hf":        HuggingFaceLLM,
}


def get_llm(provider: str, **kwargs) -> BaseLLM:
    provider = provider.lower()
    if provider not in _REGISTRY:
        raise ValueError(
            f"Unknown provider {provider!r}. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[provider](**kwargs)


__all__ = [
    "BaseLLM",
    "LLMResponse",
    "OllamaLLM",
    "OpenAILLM",
    "AnthropicLLM",
    "HuggingFaceLLM",
    "get_llm",
]
