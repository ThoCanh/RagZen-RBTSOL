"""Chunker base protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragzen.models import Chunk, Document


@runtime_checkable
class Chunker(Protocol):
    """Protocol for document chunkers.

    All chunkers must:
    - Preserve metadata from the source document
    - Track page numbers and source offsets
    - Handle Unicode correctly (including Vietnamese)
    - Produce deterministic output
    """

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            document: The document to chunk.

        Returns:
            List of Chunk objects with metadata preserved.
        """
        ...
