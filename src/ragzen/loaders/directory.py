"""Directory loader with glob support and path traversal prevention."""

from __future__ import annotations

import logging
from pathlib import Path

from ragzen.loaders.base import safe_resolve_path, validate_file
from ragzen.loaders.text import TextLoader
from ragzen.models import Document

logger = logging.getLogger("ragzen.loaders.directory")


class DirectoryLoader:
    """Loads all supported files from a directory.

    Traverses directories with configurable glob patterns and
    prevents path traversal attacks via symlinks.
    """

    def __init__(
        self,
        *,
        glob_pattern: str = "**/*",
        max_size_mb: float = 100.0,
        recursive: bool = True,
    ) -> None:
        self._glob_pattern = glob_pattern
        self._max_size_mb = max_size_mb
        self._recursive = recursive
        self._text_loader = TextLoader(max_size_mb=max_size_mb)

    def supported_mime_types(self) -> frozenset[str]:
        """Return all supported MIME types."""
        return self._text_loader.supported_mime_types()

    def load(self, source: str | Path) -> list[Document]:
        """Load all files from a directory.

        Args:
            source: Path to the directory.

        Returns:
            List of loaded Document objects.
        """
        base_dir = Path(source).resolve()
        if not base_dir.is_dir():
            msg = f"Not a directory: {source}"
            raise NotADirectoryError(msg)

        documents: list[Document] = []
        pattern = self._glob_pattern if self._recursive else "*"

        for file_path in sorted(base_dir.glob(pattern)):
            if not file_path.is_file():
                continue

            # Prevent path traversal via symlinks
            try:
                safe_resolve_path(file_path, base_dir)
            except ValueError:
                logger.warning(
                    "Skipping file outside base directory: %s", file_path
                )
                continue

            # Try to load with text loader
            try:
                validate_file(
                    file_path,
                    max_size_mb=self._max_size_mb,
                )
                docs = self._text_loader.load(file_path)
                documents.extend(docs)
            except Exception:
                logger.debug("Skipping unsupported file: %s", file_path)
                continue

        logger.info("Loaded %d documents from %s", len(documents), base_dir)
        return documents
