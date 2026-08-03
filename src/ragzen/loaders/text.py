"""Plain text document loader."""

from __future__ import annotations

import logging
from pathlib import Path

from ragzen.loaders.base import compute_file_hash, validate_file
from ragzen.models import Document

logger = logging.getLogger("ragzen.loaders.text")

_TEXT_MIME_TYPES = frozenset({
    "text/plain",
    "text/markdown",
    "text/csv",
})


class TextLoader:
    """Loads plain text files (.txt, .md, .csv).

    Supports encoding detection for non-UTF-8 files.
    """

    def __init__(
        self,
        *,
        max_size_mb: float = 100.0,
        encoding: str = "utf-8",
        fallback_encoding: str = "latin-1",
    ) -> None:
        self._max_size_mb = max_size_mb
        self._encoding = encoding
        self._fallback_encoding = fallback_encoding

    def supported_mime_types(self) -> frozenset[str]:
        """Return supported MIME types."""
        return _TEXT_MIME_TYPES

    def load(self, source: str | Path) -> list[Document]:
        """Load a text file.

        Args:
            source: Path to the text file.

        Returns:
            List containing one Document.
        """
        path = Path(source)
        mime_type = validate_file(
            path,
            max_size_mb=self._max_size_mb,
            allowed_mime_types=_TEXT_MIME_TYPES,
        )

        content = self._read_file(path)
        content_hash = compute_file_hash(path)

        return [
            Document(
                tenant_id="__unassigned__",
                content=content,
                content_hash=content_hash,
                source="file",
                source_uri=str(path.resolve()),
                file_name=path.name,
                mime_type=mime_type,
            )
        ]

    def _read_file(self, path: Path) -> str:
        """Read file with encoding fallback."""
        try:
            return path.read_text(encoding=self._encoding)
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 decode failed for %s, trying %s",
                path,
                self._fallback_encoding,
            )
            return path.read_text(encoding=self._fallback_encoding)
