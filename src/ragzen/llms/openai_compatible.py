"""OpenAI-compatible LLM provider.

Works with OpenAI, vLLM, llama.cpp server, Ollama, and any
OpenAI-compatible API endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections.abc import AsyncGenerator

import httpx
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ragzen.exceptions import ProviderError, ProviderTimeoutError

logger = logging.getLogger("ragzen.llms.openai_compatible")


def _is_retriable_error(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and getattr(exc, "retriable", False)


class OpenAICompatibleLLM:
    """LLM provider for OpenAI-compatible endpoints (Ollama, vLLM, OpenAI)."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "",
        model: str = "llama3:latest",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        timeout_seconds: float = 30.0,
        max_retries: int = 1,
        concurrency_limit: int = 10,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._sync_semaphore = threading.BoundedSemaphore(concurrency_limit)
        self._async_semaphore = asyncio.Semaphore(concurrency_limit)

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
        )
        self._headers = headers

    @property
    def model_name(self) -> str:
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
            temperature=self._temperature if temperature is None else temperature,
            max_tokens=self._max_tokens if max_tokens is None else max_tokens,
        )

    def _call_api(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the chat completions API with configured retries."""
        retrying = Retrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
            retry=retry_if_exception(_is_retriable_error),
            reraise=True,
        )
        for attempt in retrying:
            with attempt, self._sync_semaphore:
                return self._request(messages, temperature=temperature, max_tokens=max_tokens)
        raise ProviderError("LLM retry loop ended unexpectedly", provider="openai_compatible")

    def _request(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        try:
            response = self._client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": self._top_p,
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
        except httpx.TransportError as e:
            raise ProviderError(
                f"LLM transport error: {e}",
                provider="openai_compatible",
                retriable=True,
            ) from e
        except Exception as e:
            raise ProviderError(
                f"LLM generation failed: {e}",
                provider="openai_compatible",
                retriable=False,
            ) from e

    def health_check(self) -> bool:
        """Check if the LLM API is reachable."""
        try:
            response = self._client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    async def astream_generate(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield tokens from an OpenAI-compatible streaming response."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        async with (
            self._async_semaphore,
            httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._headers,
                timeout=httpx.Timeout(self._timeout_seconds),
            ) as client,
            client.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": (self._temperature if temperature is None else temperature),
                    "top_p": self._top_p,
                    "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
                    "stream": True,
                },
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                token = data.get("choices", [{}])[0].get("delta", {}).get("content")
                if token:
                    yield str(token)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
