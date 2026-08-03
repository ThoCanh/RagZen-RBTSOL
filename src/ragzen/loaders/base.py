"""Document loader base protocol and utilities.

All loaders implement the DocumentLoader protocol to provide a consistent
interface for loading documents from various sources.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import Protocol, runtime_checkable

from ragzen.exceptions import (
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from ragzen.models import Document

logger = logging.getLogger("ragzen.loaders")

# Default limits
DEFAULT_MAX_FILE_SIZE_MB = 100.0
DEFAULT_ALLOWED_MIME_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/html",
        "text/csv",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/json",
        "message/rfc822",
    }
)


@runtime_checkable
class DocumentLoader(Protocol):
    """Protocol for document loaders.

    Implementations must handle their own file format parsing and
    return Document objects with extracted text content.
    """

    def supported_mime_types(self) -> frozenset[str]:
        """Return the MIME types this loader supports."""
        ...

    def load(self, source: str | Path) -> list[Document]:
        """Load documents from a source.

        Args:
            source: Path to file or directory.

        Returns:
            List of loaded Document objects.
        """
        ...


def validate_file(
    path: Path,
    *,
    max_size_mb: float = DEFAULT_MAX_FILE_SIZE_MB,
    allowed_mime_types: frozenset[str] | None = None,
) -> str:
    """Validate a file before loading.

    Args:
        path: Path to the file.
        max_size_mb: Maximum allowed file size in MB.
        allowed_mime_types: Set of allowed MIME types.

    Returns:
        The detected MIME type.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        FileTooLargeError: If the file exceeds size limit.
        UnsupportedFileTypeError: If the MIME type is not allowed.
    """
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        msg = f"File too large: {path} ({size_mb:.1f} MB > {max_size_mb} MB limit)"
        raise FileTooLargeError(msg)

    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "application/octet-stream"

    # Check allowlist
    if allowed_mime_types and mime_type not in allowed_mime_types:
        msg = (
            f"Unsupported file type: {mime_type} for {path}. Allowed: {sorted(allowed_mime_types)}"
        )
        raise UnsupportedFileTypeError(msg)

    return mime_type


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's content."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def safe_resolve_path(path: Path, base_dir: Path | None = None) -> Path:
    """Safely resolve a path, preventing path traversal.

    Args:
        path: The path to resolve.
        base_dir: If provided, ensure resolved path is under this directory.

    Returns:
        Resolved absolute path.

    Raises:
        ValueError: If path traversal is detected.
    """
    resolved = path.resolve()

    if base_dir is not None:
        base_resolved = base_dir.resolve()
        # Check that the resolved path is within the base directory
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            msg = (
                f"Path traversal detected: {path} resolves to {resolved} "
                f"which is outside {base_resolved}"
            )
            raise ValueError(msg) from None

    return resolved
