"""Dependency-free extractive generation for zero-config local mode."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator


class ExtractiveLLM:
    """Returns concise source excerpts without calling an external model."""

    @property
    def model_name(self) -> str:
        return "ragzen-extractive"

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        del system_prompt, temperature
        context = prompt.split("\n\nQuestion:", maxsplit=1)[0]
        entries = re.findall(
            r"(\[Source\s+\d+\][^\n]*\n.*?)(?=\n\[Source\s+\d+\]|\Z)",
            context,
            flags=re.DOTALL,
        )
        if not entries:
            return "No relevant source context was available."
        character_budget = max_tokens * 4
        return (
            "Extracted answer from the most relevant sources:\n\n"
            + "\n\n".join(entry.strip() for entry in entries)[:character_budget]
        )

    def stream_generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> list[str]:
        return [
            self.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        ]

    async def astream_generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        yield self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def health_check(self) -> bool:
        return True

    def close(self) -> None:
        """No resources to release."""
