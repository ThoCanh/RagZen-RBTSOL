"""Hybrid retrieval with fusion strategies.

Combines dense and sparse retrieval with configurable fusion.
Permission filters are applied at the storage layer.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from ragzen.models import SearchResult

logger = logging.getLogger("ragzen.retrieval")


@runtime_checkable
class Retriever(Protocol):
    """Protocol for retrieval implementations."""

    def retrieve(
        self,
        query: str,
        *,
        query_vector: list[float] | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...


@runtime_checkable
class FusionStrategy(Protocol):
    """Protocol for result fusion strategies."""

    def fuse(
        self,
        result_sets: list[list[SearchResult]],
    ) -> list[SearchResult]: ...


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (RRF) strategy.

    Combines multiple ranked lists using:
    score = sum(1 / (k + rank)) for each list.
    """

    def __init__(self, *, k: int = 60) -> None:
        self._k = k

    def fuse(
        self, result_sets: list[list[SearchResult]]
    ) -> list[SearchResult]:
        """Fuse multiple result sets using RRF."""
        scores: dict[str, float] = {}
        results_map: dict[str, SearchResult] = {}

        for result_set in result_sets:
            for rank, result in enumerate(result_set):
                cid = result.chunk_id
                rrf_score = 1.0 / (self._k + rank + 1)
                scores[cid] = scores.get(cid, 0.0) + rrf_score
                # Keep the result with the highest individual score
                if cid not in results_map or result.score > results_map[cid].score:
                    results_map[cid] = result

        # Sort by RRF score
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

        return [
            results_map[cid].model_copy(update={"score": scores[cid]})
            for cid in sorted_ids
            if cid in results_map
        ]


class HybridRetriever:
    """Combines dense and sparse retrieval with fusion.

    Permission filters are passed through to the underlying stores
    and applied AT THE STORAGE LAYER — never post-filtered.
    """

    def __init__(
        self,
        *,
        vector_store: Any,
        sparse_index: Any,
        embedding_provider: Any,
        collection: str = "documents",
        fusion: FusionStrategy | None = None,
        top_k_dense: int = 30,
        top_k_sparse: int = 30,
    ) -> None:
        self._vector_store = vector_store
        self._sparse_index = sparse_index
        self._embedding = embedding_provider
        self._collection = collection
        self._fusion = fusion or ReciprocalRankFusion()
        self._top_k_dense = top_k_dense
        self._top_k_sparse = top_k_sparse

    def retrieve(
        self,
        query: str,
        *,
        query_vector: list[float] | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Retrieve using hybrid search.

        Filters are applied at storage layer for both dense and sparse.
        """
        dense_results = self._dense_search(
            query, query_vector=query_vector, filters=filters
        )
        sparse_results = self._sparse_search(query, filters=filters)

        # Fuse results
        fused = self._fusion.fuse([dense_results, sparse_results])

        # Deduplicate
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        for result in fused:
            if result.chunk_id not in seen:
                seen.add(result.chunk_id)
                deduped.append(result)

        return deduped[:top_k]

    def _dense_search(
        self,
        query: str,
        *,
        query_vector: list[float] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Dense retrieval using vector similarity."""
        if query_vector is None:
            query_vector = self._embedding.embed_query(query)

        raw = self._vector_store.search(
            self._collection,
            query_vector,
            top_k=self._top_k_dense,
            filters=filters,
        )

        return [
            SearchResult(
                chunk_id=cid,
                document_id=meta.get("document_id", ""),
                content=meta.get("content", ""),
                score=score,
                page=meta.get("page"),
                file_name=meta.get("file_name", ""),
                metadata=meta,
                retrieval_method="dense",
            )
            for cid, score, meta in raw
        ]

    def _sparse_search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Sparse retrieval using BM25."""
        raw = self._sparse_index.search(
            query,
            top_k=self._top_k_sparse,
            filters=filters,
        )

        return [
            SearchResult(
                chunk_id=cid,
                document_id=meta.get("document_id", ""),
                content=meta.get("content", ""),
                score=score,
                page=meta.get("page"),
                file_name=meta.get("file_name", ""),
                metadata=meta,
                retrieval_method="sparse",
            )
            for cid, score, meta in raw
        ]
