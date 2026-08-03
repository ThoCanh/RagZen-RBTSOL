"""Optional cross-encoder reranking."""

from __future__ import annotations

from typing import Any

from ragzen.exceptions import MissingOptionalDependencyError
from ragzen.models import SearchResult


class CrossEncoderReranker:
    """Lazy sentence-transformers cross-encoder reranker."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise MissingOptionalDependencyError(
                    "sentence-transformers", "local", "cross-encoder reranking"
                ) from exc
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, results: list[SearchResult], *, top_k: int) -> list[SearchResult]:
        if not results:
            return []
        scores = self._load().predict([(query, result.content) for result in results])
        reranked = [
            result.model_copy(update={"score": float(score), "retrieval_method": "reranked"})
            for result, score in zip(results, scores, strict=True)
        ]
        reranked.sort(key=lambda result: result.score, reverse=True)
        return reranked[:top_k]
