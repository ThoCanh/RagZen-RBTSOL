"""Recursive text chunker.

Splits text using a hierarchy of separators, trying larger splits first
and recursing into smaller ones as needed. Preserves metadata and
handles Unicode text including Vietnamese.
"""

from __future__ import annotations

import hashlib
import logging

from ragzen.models import Chunk, Document

logger = logging.getLogger("ragzen.chunkers.recursive")

# Default separators ordered from most to least aggressive
DEFAULT_SEPARATORS = [
    "\n\n",  # Paragraph
    "\n",  # Line break
    ". ",  # Sentence (with space)
    "! ",  # Exclamation
    "? ",  # Question
    "; ",  # Semicolon
    ", ",  # Comma
    " ",  # Space (word boundary)
]

# Vietnamese-friendly separators
VIETNAMESE_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
]


class RecursiveChunker:
    """Splits text recursively using separator hierarchy.

    Tries to split at natural boundaries (paragraphs, sentences)
    before falling back to character-level splits.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            msg = f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
            raise ValueError(msg)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or list(DEFAULT_SEPARATORS)

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into chunks using recursive splitting.

        Args:
            document: The document to chunk.

        Returns:
            List of Chunk objects.
        """
        text = document.content
        if not text.strip():
            return []

        raw_chunks = self._split_text(text, self._separators)

        chunks: list[Chunk] = []
        current_offset = 0

        for seq, chunk_text in enumerate(raw_chunks):
            # Find actual offset in original text
            idx = text.find(chunk_text, current_offset)
            start_offset = idx if idx >= 0 else current_offset
            end_offset = start_offset + len(chunk_text)

            content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    document_version=document.version,
                    content=chunk_text,
                    content_hash=content_hash,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    sequence=seq,
                    metadata=document.metadata.copy(),
                    access_control=document.access_control,
                )
            )
            current_offset = start_offset + 1

        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separator hierarchy."""
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []

        # Try each separator
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                return self._merge_splits(parts, sep, separators)

        # No separator works — force split at chunk_size
        return self._force_split(text)

    def _merge_splits(
        self,
        parts: list[str],
        separator: str,
        separators: list[str],
    ) -> list[str]:
        """Merge split parts into chunks that respect size limits."""
        result: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part) + len(separator)

            if current_len + part_len > self._chunk_size and current:
                merged = separator.join(current)
                if merged.strip():
                    result.append(merged)

                # Keep overlap
                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) > self._chunk_overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + len(separator)
                current = overlap_parts
                current_len = overlap_len

            if len(part) > self._chunk_size:
                # Recursively split with remaining separators
                remaining_seps = separators[separators.index(separator) + 1 :]
                sub_chunks = self._split_text(part, remaining_seps)
                for sub in sub_chunks:
                    result.append(sub)
                current = []
                current_len = 0
            else:
                current.append(part)
                current_len += part_len

        if current:
            merged = separator.join(current)
            if merged.strip():
                result.append(merged)

        return result

    def _force_split(self, text: str) -> list[str]:
        """Force split text at chunk_size boundaries."""
        result: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self._chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                result.append(chunk)
            start += self._chunk_size - self._chunk_overlap
        return result
