"""Unit tests for TextLoader, DirectoryLoader, and loader base utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragzen.exceptions import FileTooLargeError, UnsupportedFileTypeError
from ragzen.loaders.base import compute_file_hash, safe_resolve_path, validate_file
from ragzen.loaders.directory import DirectoryLoader
from ragzen.loaders.text import TextLoader


class TestLoaders:
    def test_text_loader_utf8(self, tmp_path: Path) -> None:
        p = tmp_path / "sample.txt"
        p.write_text("Nội dung tiếng Việt thử nghiệm.", encoding="utf-8")

        loader = TextLoader()
        docs = loader.load(p)
        assert len(docs) == 1
        assert docs[0].content == "Nội dung tiếng Việt thử nghiệm."
        assert docs[0].file_name == "sample.txt"

    def test_text_loader_fallback_encoding(self, tmp_path: Path) -> None:
        p = tmp_path / "latin.txt"
        p.write_bytes("Hello world".encode("latin-1"))

        loader = TextLoader(encoding="utf-8", fallback_encoding="latin-1")
        docs = loader.load(p)
        assert len(docs) == 1

    def test_validate_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            validate_file(Path("/nonexistent/file.txt"))

    def test_validate_file_too_large(self, tmp_path: Path) -> None:
        p = tmp_path / "large.txt"
        p.write_bytes(b"x" * 2000)
        with pytest.raises(FileTooLargeError):
            validate_file(p, max_size_mb=0.001)

    def test_validate_file_unsupported_mime(self, tmp_path: Path) -> None:
        p = tmp_path / "app.exe"
        p.write_bytes(b"exe")
        with pytest.raises(UnsupportedFileTypeError):
            validate_file(p, allowed_mime_types=frozenset({"text/plain"}))

    def test_compute_file_hash(self, tmp_path: Path) -> None:
        p = tmp_path / "data.txt"
        p.write_text("test data", encoding="utf-8")
        h1 = compute_file_hash(p)
        h2 = compute_file_hash(p)
        assert len(h1) == 64
        assert h1 == h2

    def test_safe_resolve_path_traversal(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("outside")

        with pytest.raises(ValueError, match="Path traversal"):
            safe_resolve_path(outside, base)

    def test_directory_loader(self, tmp_path: Path) -> None:
        d = tmp_path / "docs"
        d.mkdir()
        (d / "f1.txt").write_text("File 1 content", encoding="utf-8")
        (d / "f2.md").write_text("# File 2 Markdown", encoding="utf-8")

        sub = d / "sub"
        sub.mkdir()
        (sub / "f3.txt").write_text("File 3 sub", encoding="utf-8")

        loader = DirectoryLoader(recursive=True)
        docs = loader.load(d)
        assert len(docs) == 3

    def test_directory_loader_not_a_directory(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("content")
        loader = DirectoryLoader()
        with pytest.raises(NotADirectoryError):
            loader.load(f)
