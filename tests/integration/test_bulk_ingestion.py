"""Integration test for bulk document ingestion."""

from __future__ import annotations

from pathlib import Path

from ragzen import RagZen, SecurityContext


def test_bulk_ingestion(tmp_path: Path) -> None:
    rag = RagZen.local(storage_path=str(tmp_path))

    # Bulk ingest 50 documents
    docs_dir = tmp_path / "bulk_docs"
    docs_dir.mkdir()

    for i in range(50):
        f = docs_dir / f"doc_{i:03d}.txt"
        f.write_text(f"Nội dung văn bản số {i} phục vụ kiểm thử số lượng lớn.", encoding="utf-8")

    job = rag.add(docs_dir, metadata={"tenant_id": "bulk-tenant"})
    assert job.processed_documents == 50
    assert job.failed_documents == 0

    ctx = SecurityContext(tenant_id="bulk-tenant")
    results = rag.search("số lượng lớn", top_k=10, security_context=ctx)
    assert len(results) == 10
    assert rag.stats()["document_count"] == 50

    rag.close()
