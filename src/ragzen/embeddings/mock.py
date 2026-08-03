"""Mock and deterministic local embedding providers for testing and CPU fallback."""

from __future__ import annotations

import hashlib
import math


class MockEmbeddingProvider:
    """Deterministic hash-based embedding provider for testing and fallback.

    Produces normalized N-dimensional float vectors from text string hashes.
    Does not require external model weights or PyTorch.
    """

    def __init__(self, dimensions: int = 384, model_name: str = "mock-embedding-384") -> None:
        self._dimensions = dimensions
        self._model_name = model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        vec = []
        for i in range(self._dimensions):
            h = hashlib.sha256(f"{text}:{i}".encode()).hexdigest()
            val = (int(h[:8], 16) / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(val)

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def health_check(self) -> bool:
        return True
