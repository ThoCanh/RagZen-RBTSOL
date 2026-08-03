"""Fixed-size text chunker.

Splits text into fixed-size chunks with configurable overlap.
Supports both character-based and token-based modes.
"""

from __future__ import annotations

import hashlib
import logging

from ragzen.models import Chunk, Document

logger = logging.getLogger("ragzen.chunkers.fixed")


class FixedSizeChunker:
    """Splits documents into fixed-size chunks.

    Supports character-based chunking with Unicode-safe boundaries.
    Does not split in the middle of a Unicode character.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        if chunk_overlap >= chunk_size:
            msg = f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            raise ValueError(msg)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into fixed-size chunks.

        Args:
            document: The document to chunk.

        Returns:
            List of Chunk objects.
        """
        text = document.content
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        sequence = 0

        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk_text = text[start:end]

            if chunk_text.strip():
                content_hash = hashlib.sha256(
                    chunk_text.encode("utf-8")
                ).hexdigest()

                chunks.append(
                    Chunk(
                        document_id=document.document_id,
                        document_version=document.version,
                        content=chunk_text,
                        content_hash=content_hash,
                        start_offset=start,
                        end_offset=end,
                        sequence=sequence,
                        metadata=document.metadata.copy(),
                        access_control=document.access_control,
                    )
                )
                sequence += 1

            step = self._chunk_size - self._chunk_overlap
            start += max(step, 1)

        return chunks
