"""Sparse index base protocol."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SparseIndex(Protocol):
    """Protocol for sparse index implementations."""

    def add(self, chunk_id: str, content: str, metadata: dict[str, Any] | None = None) -> None: ...

    def remove(self, chunk_id: str) -> bool: ...

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]: ...

    def count(self) -> int: ...

    def clear(self, *, filters: dict[str, Any] | None = None) -> None: ...

    def close(self) -> None: ...
