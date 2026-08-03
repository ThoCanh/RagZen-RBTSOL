"""Unit tests for FixedSizeChunker and RecursiveChunker."""

from __future__ import annotations

import pytest

from ragzen.chunkers.fixed import FixedSizeChunker
from ragzen.chunkers.recursive import RecursiveChunker
from ragzen.models import AccessControl, Document


class TestFixedSizeChunker:
    def test_chunking_fixed(self) -> None:
        doc = Document(tenant_id="t1", content="1234567890" * 10)
        chunker = FixedSizeChunker(chunk_size=30, chunk_overlap=10)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        assert chunks[0].content == ("1234567890" * 10)[:30]

    def test_chunking_empty_doc(self) -> None:
        doc = Document(tenant_id="t1", content="   ")
        chunker = FixedSizeChunker()
        assert chunker.chunk(doc) == []

    def test_invalid_overlap(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            FixedSizeChunker(chunk_size=50, chunk_overlap=50)


class TestRecursiveChunker:
    def test_recursive_chunking_vietnamese(self) -> None:
        content = (
            "Quy trình xử lý sản phẩm lỗi bao gồm 3 bước chính.\n\n"
            "Bước 1: Ghi nhận biên bản sự cố và phân loại mức độ.\n\n"
            "Bước 2: Chuyển sang bộ phận kỹ thuật để thẩm định.\n\n"
            "Bước 3: Thực hiện tái chế hoặc tiêu hủy theo quy định."
        )
        ac = AccessControl(tenant_id="t1", departments=["production"])
        doc = Document(tenant_id="t1", content=content, access_control=ac)

        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 3
        assert all(c.access_control == ac for c in chunks)
        assert all(c.document_id == doc.document_id for c in chunks)

    def test_recursive_chunking_empty(self) -> None:
        doc = Document(tenant_id="t1", content="")
        chunker = RecursiveChunker()
        assert chunker.chunk(doc) == []

    def test_force_split(self) -> None:
        # Text with no separators that exceeds chunk_size
        doc = Document(tenant_id="t1", content="A" * 200)
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
