"""SentenceTransformer local embedding provider for RagZen.

Uses HuggingFace sentence-transformers for local, dense semantic embeddings.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("ragzen.embeddings.sentence_transformer")


class SentenceTransformerEmbeddingProvider:
    """Real dense vector embedding provider backed by sentence-transformers."""

    @classmethod
    def is_available(cls) -> bool:
        """Check if sentence-transformers package is installed."""
        try:
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import (
                    SentenceTransformer,
                )

                logger.info("Loading SentenceTransformer model: %s", self._model_name)
                self._model = SentenceTransformer(self._model_name)
            except ImportError as err:
                msg = (
                    "package 'sentence-transformers' is not installed. "
                    "Install it via: pip install 'ragzen[local]' "
                    "or pip install sentence-transformers"
                )
                raise ImportError(msg) from err
        return self._model

    @property
    def dimensions(self) -> int:
        model = self._load_model()
        return int(model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        values = embeddings.tolist()
        return [[float(value) for value in row] for row in values]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            model = self._load_model()
            dim = int(model.get_sentence_embedding_dimension())
            return [0.0] * dim
        embeddings = self.embed([text])
        return embeddings[0]

    def health_check(self) -> bool:
        try:
            self.embed_query("healthcheck")
            return True
        except Exception as e:
            logger.warning("SentenceTransformer healthcheck failed: %s", e)
            return False
