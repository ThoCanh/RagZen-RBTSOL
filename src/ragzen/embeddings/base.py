"""Embedding provider protocol and utilities."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    All implementations must support:
    - Batch embedding with configurable batch size
    - Dimension validation
    - Model fingerprint for version tracking
    """

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions."""
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: Query text.

        Returns:
            Embedding vector.
        """
        ...

    def health_check(self) -> bool:
        """Check if the embedding provider is healthy."""
        ...
