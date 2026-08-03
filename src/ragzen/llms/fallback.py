"""Fallback LLM Provider chain.

Tries primary LLM provider first. On failure/unavailability, falls back
to secondary providers with warning log.
"""

from __future__ import annotations

import logging

from ragzen.exceptions import AllProvidersFailedError
from ragzen.llms.base import LLMProvider

logger = logging.getLogger("ragzen.llms.fallback")


class FallbackLLMProvider:
    """Chain of LLM providers with automatic fallback."""

    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            msg = "FallbackLLMProvider requires at least one provider"
            raise ValueError(msg)
        self._providers = providers

    @property
    def model_name(self) -> str:
        return self._providers[0].model_name

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        errors: list[Exception] = []

        for provider in self._providers:
            try:
                if hasattr(provider, "health_check") and not provider.health_check():
                    name = getattr(provider, "model_name", "unknown")
                    logger.debug("Provider %s unhealthy, skipping", name)
                    continue
                return provider.generate(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                logger.warning(
                    "LLM Provider %s failed: %s. Trying fallback...",
                    getattr(provider, "model_name", "unknown"),
                    e,
                )
                errors.append(e)

        # If all fail or were unhealthy, try forcing primary as last resort
        try:
            return self._providers[0].generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            errors.append(e)

        raise AllProvidersFailedError(
            "All LLM providers in fallback chain failed",
            errors=errors,
        )

    def health_check(self) -> bool:
        return any(getattr(p, "health_check", lambda: True)() for p in self._providers)

    def close(self) -> None:
        for provider in self._providers:
            close = getattr(provider, "close", None)
            if close:
                close()
