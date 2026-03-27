from typing import Optional

from .base import BaseLLM, LLMResponse


class HuggingFaceLLM(BaseLLM):
    """
    Adapter for local HuggingFace models using the `transformers` pipeline.

    Args:
        model_name: HF model hub id, e.g. "mistralai/Mistral-7B-Instruct-v0.2"
        device:     "cpu", "cuda", or "mps"
        load_in_8bit / load_in_4bit: quantization via bitsandbytes
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
        except ImportError as exc:
            raise ImportError("Install transformers and torch: pip install transformers torch") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs: dict = {}
        if load_in_8bit or load_in_4bit:
            model_kwargs["quantization_config"] = self._build_quant_config(
                load_in_8bit, load_in_4bit
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device if device != "cpu" else None,
            **model_kwargs,
        )

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self._tokenizer,
            device=None if device != "cpu" else -1,
        )

    @staticmethod
    def _build_quant_config(load_in_8bit: bool, load_in_4bit: bool):
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        outputs = self._pipeline(
            full_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            return_full_text=False,
            **kwargs,
        )

        text = outputs[0]["generated_text"].strip()
        prompt_tokens = len(self._tokenizer.encode(full_prompt))
        completion_tokens = len(self._tokenizer.encode(text))

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model_name=self.model_name,
        )
