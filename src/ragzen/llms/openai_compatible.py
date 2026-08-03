"""OpenAI-compatible LLM provider.

Works with OpenAI, vLLM, llama.cpp server, Ollama, and any
OpenAI-compatible API endpoint.
"""

from __future__ import annotations

import logging

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ragzen.exceptions import ProviderError, ProviderTimeoutError

logger = logging.getLogger("ragzen.llms.openai_compatible")


class OpenAICompatibleLLM:
    """LLM provider for OpenAI-compatible APIs.

    Supports vLLM, llama.cpp, Ollama (/v1), and OpenAI.
    Includes retry with exponential backoff and timeout.

    Never logs API keys or Authorization headers.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "",
        model: str = "qwen2.5",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._max_retries = max_retries

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
        )

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt with context.
            system_prompt: System instructions.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            Generated text.

        Raises:
            ProviderError: On API errors.
            ProviderTimeoutError: On timeout.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self._call_api(
            messages,
            temperature=temperature or self._temperature,
            max_tokens=max_tokens or self._max_tokens,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(ProviderError),
        reraise=True,
    )
    def _call_api(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the chat completions API with retry."""
        try:
            response = self._client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(
                provider="openai_compatible",
                details={"model": self._model},
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"LLM API error: {e.response.status_code}",
                provider="openai_compatible",
                retriable=e.response.status_code >= 500,
            ) from e
        except Exception as e:
            raise ProviderError(
                f"LLM generation failed: {e}",
                provider="openai_compatible",
            ) from e

    def health_check(self) -> bool:
        """Check if the LLM API is reachable."""
        try:
            response = self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
