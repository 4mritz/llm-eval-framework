"""Unit tests for LLM adapters using mocks — no live API calls."""
from unittest.mock import MagicMock, patch
import pytest

from src.llms.base import LLMResponse
from src.llms import get_llm


class TestGetLlm:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm("nonexistent_provider", model_name="x")


class TestOllamaAdapter:
    @patch("src.llms.ollama_adapter.requests.post")
    def test_generate_returns_response(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "message": {"content": "Test summary output."},
                "prompt_eval_count": 50,
                "eval_count": 20,
            },
        )
        llm = get_llm("ollama", model_name="llama3.1:8b")
        response = llm.generate("Summarize this.")

        assert isinstance(response, LLMResponse)
        assert response.text == "Test summary output."
        assert response.prompt_tokens == 50
        assert response.completion_tokens == 20
        assert response.total_tokens == 70
        assert response.model_name == "llama3.1:8b"

    @patch("src.llms.ollama_adapter.requests.post")
    def test_system_prompt_included_in_messages(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "message": {"content": "ok"},
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )
        llm = get_llm("ollama", model_name="llama3.1:8b")
        llm.generate("Hello", system_prompt="You are helpful.")

        call_payload = mock_post.call_args.kwargs["json"]
        messages = call_payload["messages"]
        roles = [m["role"] for m in messages]
        assert "system" in roles
        assert "user" in roles


class TestLLMResponse:
    def test_total_tokens_matches_sum(self):
        r = LLMResponse(text="hi", prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert r.total_tokens == 15

    def test_default_cost_is_none(self):
        r = LLMResponse(text="hi")
        assert r.cost_estimate_usd is None
