from typing import Optional

import requests

from .base import BaseLLM, LLMResponse


class OllamaLLM(BaseLLM):
    """
    Adapter for locally-hosted models served via Ollama's REST API.

    Args:
        model_name: Ollama model tag, e.g. "llama3.1:8b-instruct-q4_K_M"
        base_url:   Ollama server root, default http://localhost:11434
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        self._chat_endpoint = f"{self.base_url}/api/chat"

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

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **{k: v for k, v in kwargs.items()},
            },
        }

        response = requests.post(self._chat_endpoint, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        text = data["message"]["content"].strip()
        prompt_eval_count = data.get("prompt_eval_count", 0)
        eval_count = data.get("eval_count", 0)

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            total_tokens=prompt_eval_count + eval_count,
            model_name=self.model_name,
            raw=data,
        )
