"""In-memory vector store with numpy cosine similarity.

Suitable for development, testing, and small datasets.
Data is lost on process restart.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

logger = logging.getLogger("ragzen.vectorstores.memory")


class InMemoryVectorStore:
    """In-memory vector store using brute-force cosine similarity.

    Stores vectors, chunks, and metadata in Python dicts.
    Supports tenant-scoped operations and metadata filtering.
    """

    def __init__(self) -> None:
        # collection_name -> {chunk_id -> (vector, chunk, metadata)}
        self._collections: dict[str, dict[str, tuple[list[float], dict[str, Any]]]] = {}
        self._lock = threading.RLock()

    def create_collection(
        self,
        name: str,
        *,
        dimensions: int = 0,
    ) -> None:
        """Create a new collection."""
        with self._lock:
            if name not in self._collections:
                self._collections[name] = {}
                logger.debug("Created collection: %s", name)

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        with self._lock:
            return name in self._collections

    def upsert(
        self,
        collection: str,
        chunk_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Insert or update a vector with metadata."""
        with self._lock:
            if collection not in self._collections:
                self.create_collection(collection)
            self._collections[collection][chunk_id] = (vector, metadata)

    def batch_upsert(
        self,
        collection: str,
        items: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        """Batch insert/update vectors."""
        for chunk_id, vector, metadata in items:
            self.upsert(collection, chunk_id, vector, metadata)

    def delete(self, collection: str, chunk_id: str) -> bool:
        """Delete a vector by chunk_id."""
        with self._lock:
            if collection in self._collections:
                return self._collections[collection].pop(chunk_id, None) is not None
            return False

    def delete_by_filter(self, collection: str, filters: dict[str, Any]) -> int:
        """Delete vectors matching filter criteria."""
        with self._lock:
            if collection not in self._collections:
                return 0
            to_delete = []
            for cid, (_, meta) in self._collections[collection].items():
                if self._matches_filters(meta, filters):
                    to_delete.append(cid)
            for cid in to_delete:
                del self._collections[collection][cid]
            return len(to_delete)

    def search(
        self,
        collection: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar vectors.

        Args:
            collection: Collection name.
            query_vector: Query embedding.
            top_k: Number of results.
            filters: Metadata filters (applied BEFORE scoring).

        Returns:
            List of (chunk_id, score, metadata) sorted by score desc.
        """
        with self._lock:
            items = list(self._collections.get(collection, {}).items())
        results: list[tuple[str, float, dict[str, Any]]] = []
        for chunk_id, (vector, metadata) in items:
            # Apply mandatory filters FIRST
            if filters and not self._matches_filters(metadata, filters):
                continue

            score = self._cosine_similarity(query_vector, vector)
            results.append((chunk_id, score, metadata))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def count(self, collection: str) -> int:
        """Count vectors in a collection."""
        with self._lock:
            return len(self._collections.get(collection, {}))

    def health(self) -> bool:
        """Always healthy for in-memory store."""
        return True

    def clear(
        self,
        collection: str | None = None,
        *,
        filters: dict[str, Any] | None = None,
    ) -> None:
        """Clear a collection or all collections."""
        if collection and filters:
            self.delete_by_filter(collection, filters)
            return
        with self._lock:
            if collection:
                self._collections.pop(collection, None)
            else:
                self._collections.clear()

    def close(self) -> None:
        """Release resources (no-op for the in-memory backend)."""

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches all filter criteria.

        Supports:
        - Tenant isolation
        - Department matching
        - Role matching (if document specifies roles)
        - Scalar/list filters
        """
        from ragzen.security.filters import metadata_matches_filters

        return metadata_matches_filters(metadata, filters)
