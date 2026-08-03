"""Vector store base protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector store implementations."""

    def create_collection(
        self, name: str, *, dimensions: int = 0
    ) -> None: ...

    def collection_exists(self, name: str) -> bool: ...

    def upsert(
        self,
        collection: str,
        chunk_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None: ...

    def delete(self, collection: str, chunk_id: str) -> bool: ...

    def search(
        self,
        collection: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]: ...

    def count(self, collection: str) -> int: ...

    def health(self) -> bool: ...
