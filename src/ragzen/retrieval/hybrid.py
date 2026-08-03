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

    def fuse(self, result_sets: list[list[SearchResult]]) -> list[SearchResult]:
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


class WeightedScoreFusion:
    """Fuse result sets after min-max score normalization."""

    def __init__(self, weights: list[float] | None = None) -> None:
        self._weights = weights or []

    def fuse(self, result_sets: list[list[SearchResult]]) -> list[SearchResult]:
        scores: dict[str, float] = {}
        results: dict[str, SearchResult] = {}
        for index, result_set in enumerate(result_sets):
            if not result_set:
                continue
            weight = self._weights[index] if index < len(self._weights) else 1.0
            values = [item.score for item in result_set]
            low, high = min(values), max(values)
            for item in result_set:
                normalized = (item.score - low) / (high - low) if high > low else 1.0
                scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + weight * normalized
                if item.chunk_id not in results or item.score > results[item.chunk_id].score:
                    results[item.chunk_id] = item
        return [
            results[chunk_id].model_copy(update={"score": scores[chunk_id]})
            for chunk_id in sorted(scores, key=lambda current: scores[current], reverse=True)
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
        graph_index: Any = None,
        collection: str = "documents",
        fusion: FusionStrategy | None = None,
        mode: str = "hybrid",
        top_k_dense: int = 30,
        top_k_sparse: int = 30,
    ) -> None:
        self._vector_store = vector_store
        self._sparse_index = sparse_index
        self._embedding = embedding_provider
        self._graph_index = graph_index
        self._collection = collection
        self._fusion = fusion or ReciprocalRankFusion()
        self._mode = mode
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
        result_sets: list[list[SearchResult]] = []
        if self._mode in {"dense", "hybrid", "hybrid_graph"}:
            result_sets.append(
                self._dense_search(query, query_vector=query_vector, filters=filters)
            )
        if self._mode in {"sparse", "hybrid", "hybrid_graph"}:
            result_sets.append(self._sparse_search(query, filters=filters))
        if self._mode in {"graph", "hybrid_graph"}:
            result_sets.append(self._graph_search(query, filters=filters))

        fused = self._fusion.fuse(result_sets)

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
                source_uri=meta.get("source_uri", ""),
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
                source_uri=meta.get("source_uri", ""),
                metadata=meta,
                retrieval_method="sparse",
            )
            for cid, score, meta in raw
        ]

    def _graph_search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._graph_index is None:
            return []
        raw = self._graph_index.search(
            query,
            top_k=max(self._top_k_dense, self._top_k_sparse),
            filters=filters,
        )
        return [
            SearchResult(
                chunk_id=chunk_id,
                document_id=metadata.get("document_id", ""),
                content=metadata.get("content", ""),
                score=score,
                page=metadata.get("page"),
                file_name=metadata.get("file_name", ""),
                source_uri=metadata.get("source_uri", ""),
                metadata=metadata,
                retrieval_method="graph",
            )
            for chunk_id, score, metadata in raw
        ]
