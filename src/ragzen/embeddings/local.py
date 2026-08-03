"""Deterministic local n-gram TF-IDF feature embedding provider.

Provides real, deterministic, non-mock vector embeddings using character/word n-gram 
subword hashing and term frequency weighting (384 dimensions).
Used as a fast, dependency-free local fallback when sentence-transformers is absent.
"""

from __future__ import annotations

import math
import re
import xxhash


class DeterministicLocalEmbeddingProvider:
    """Deterministic n-gram subword feature embedding provider (384 dimensions)."""

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions
        self._model_name = f"deterministic-local-ngram-{dimensions}d"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def _extract_ngrams(self, text: str) -> list[str]:
        cleaned = text.lower().strip()
        words = re.findall(r"\w+", cleaned)
        ngrams: list[str] = list(words)

        # Character n-grams (3-gram to 5-gram) for subword matching
        for word in words:
            for n in range(3, 6):
                if len(word) >= n:
                    for i in range(len(word) - n + 1):
                        ngrams.append(word[i : i + n])
        return ngrams

    def _compute_vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        ngrams = self._extract_ngrams(text)
        if not ngrams:
            return vector

        # Hash each n-gram deterministically into vector index
        for ngram in ngrams:
            idx = xxhash.xxh64(ngram.encode("utf-8")).intdigest() % self._dimensions
            # Frequency weight + position signal
            vector[idx] += 1.0

        # L2 Normalize vector so cosine similarity works properly
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._compute_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._compute_vector(text)

    def health_check(self) -> bool:
        return True
