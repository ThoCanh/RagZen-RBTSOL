"""Integration tests for RagZen engine quickstart & end-to-end flows."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragzen import RagZen, SecurityContext


@pytest.fixture
def temp_rag_dir(tmp_path: Path) -> Path:
    return tmp_path / ".ragzen"


class TestRagZenEngine:
    """Integration test suite for RagZen engine."""

    def test_quickstart_flow(self, temp_rag_dir: Path) -> None:
        # Create test document
        doc_dir = temp_rag_dir / "documents"
        doc_dir.mkdir(parents=True, exist_ok=True)
        file_path = doc_dir / "quy_trinh.txt"
        file_path.write_text(
            "Quy trình xử lý sản phẩm lỗi bao gồm phân loại, "
            "ghi nhận biên bản và chuyển bộ phận tái chế.",
            encoding="utf-8",
        )

        rag = RagZen.local(storage_path=str(temp_rag_dir))

        # Ingest
        job = rag.add(
            file_path,
            metadata={
                "tenant_id": "company-a",
                "department": "production",
                "access_level": "internal",
            },
        )
        assert job.processed_documents == 1

        # Search
        sec_context = SecurityContext(
            tenant_id="company-a",
            user_id="user-123",
            roles=["production_manager"],
            departments=["production"],
        )

        results = rag.search("Quy trình xử lý sản phẩm lỗi", security_context=sec_context)
        assert len(results) > 0
        assert results[0].metadata["tenant_id"] == "company-a"

        # Ask
        response = rag.ask(
            "Quy trình xử lý sản phẩm lỗi là gì?",
            security_context=sec_context,
        )

        assert response.answer
        assert len(response.sources) > 0
        assert len(response.citations) > 0

    def test_tenant_isolation_in_engine(self, temp_rag_dir: Path) -> None:
        rag = RagZen.local(storage_path=str(temp_rag_dir))

        # Add doc for tenant A
        rag.add_text(
            "Báo cáo tài chính công ty A: Lợi nhuận 10 tỷ.",
            metadata={"tenant_id": "company-a", "department": "finance"},
        )

        # Add doc for tenant B
        rag.add_text(
            "Báo cáo tài chính công ty B: Lợi nhuận 50 tỷ.",
            metadata={"tenant_id": "company-b", "department": "finance"},
        )

        ctx_a = SecurityContext(tenant_id="company-a", user_id="user-a", departments=["finance"])
        ctx_b = SecurityContext(tenant_id="company-b", user_id="user-b", departments=["finance"])

        # Search with Tenant A context
        results_a = rag.search("báo cáo tài chính", security_context=ctx_a)
        assert all(r.metadata["tenant_id"] == "company-a" for r in results_a)
        assert not any(r.metadata["tenant_id"] == "company-b" for r in results_a)

        # Search with Tenant B context
        results_b = rag.search("báo cáo tài chính", security_context=ctx_b)
        assert all(r.metadata["tenant_id"] == "company-b" for r in results_b)
        assert not any(r.metadata["tenant_id"] == "company-a" for r in results_b)

    @pytest.mark.asyncio
    async def test_async_api(self, temp_rag_dir: Path) -> None:
        rag = RagZen.local(storage_path=str(temp_rag_dir))

        await rag.aadd(
            tmp_path_file(temp_rag_dir, "test.txt", "Quy trình vận hành máy nén khí."),
            metadata={"tenant_id": "company-a"},
        )

        ctx = SecurityContext(tenant_id="company-a")
        results = await rag.asearch("máy nén khí", security_context=ctx)
        assert len(results) > 0

        resp = await rag.aask("máy nén khí", security_context=ctx)
        assert resp.answer

        tokens = []
        async for chunk in rag.stream("máy nén khí", security_context=ctx):
            tokens.append(chunk)
        assert len(tokens) > 0

    def test_health_and_stats(self, temp_rag_dir: Path) -> None:
        rag = RagZen.local(storage_path=str(temp_rag_dir))
        rag.add_text("Dữ liệu mẫu.", metadata={"tenant_id": "company-a"})

        health = rag.health()
        assert health.healthy is True
        assert health.document_count == 1

        stats = rag.stats()
        assert stats["document_count"] == 1
        assert stats["environment"] == "development"

    def test_delete_and_clear(self, temp_rag_dir: Path) -> None:
        rag = RagZen.local(storage_path=str(temp_rag_dir))
        doc = rag.add_text("Nội dung cần xóa.", metadata={"tenant_id": "company-a"})

        ctx = SecurityContext(tenant_id="company-a")
        assert len(rag.search("xóa", security_context=ctx)) == 1

        # Delete
        deleted = rag.delete(doc.document_id, tenant_id="company-a")
        assert deleted is True
        assert len(rag.search("xóa", security_context=ctx)) == 0

        # Clear
        rag.add_text("Nội dung 1", metadata={"tenant_id": "company-a"})
        rag.clear()
        assert rag.stats()["document_count"] == 0


def tmp_path_file(base: Path, name: str, content: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    p = base / name
    p.write_text(content, encoding="utf-8")
    return p
