"""LLM provider base protocol."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt with context.
            system_prompt: System instructions.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        ...

    def health_check(self) -> bool: ...

    def astream_generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]: ...
