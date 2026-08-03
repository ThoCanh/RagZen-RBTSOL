"""Unit tests for FallbackLLMProvider and OpenAICompatibleLLM."""

from __future__ import annotations

import pytest

from ragzen.exceptions import AllProvidersFailedError
from ragzen.llms.fallback import FallbackLLMProvider
from ragzen.llms.mock import MockLLMProvider
from ragzen.llms.openai_compatible import OpenAICompatibleLLM


class FailingLLMProvider:
    @property
    def model_name(self) -> str:
        return "failing-model"

    def generate(self, prompt: str, **kwargs: object) -> str:
        msg = "Service unavailable"
        raise RuntimeError(msg)

    def health_check(self) -> bool:
        return False


class TestLLMProviders:
    def test_fallback_chain_success(self) -> None:
        p1 = FailingLLMProvider()
        p2 = MockLLMProvider("mock-fallback")

        chain = FallbackLLMProvider([p1, p2])
        assert chain.model_name == "failing-model"
        assert chain.health_check() is True

        res = chain.generate("Quy trình xử lý sản phẩm lỗi là gì?")
        assert "Quy trình" in res

    def test_fallback_chain_all_failed(self) -> None:
        p1 = FailingLLMProvider()
        chain = FallbackLLMProvider([p1])
        assert chain.health_check() is False

        with pytest.raises(AllProvidersFailedError):
            chain.generate("Hello")

    def test_fallback_empty_init_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one provider"):
            FallbackLLMProvider([])

    def test_openai_compatible_init_and_close(self) -> None:
        llm = OpenAICompatibleLLM(
            base_url="http://localhost:11434/v1",
            api_key="test-key",
            model="qwen2.5",
        )
        assert llm.model_name == "qwen2.5"
        assert llm.health_check() is False
        llm.close()
