"""Unit tests for DocumentRegistry storage operations."""

from __future__ import annotations

from pathlib import Path

from ragzen.models import AccessControl, Document, DocumentStatus
from ragzen.storage.documents import DocumentRegistry


class TestDocumentRegistry:
    def test_save_get_count(self, tmp_path: Path) -> None:
        db_path = tmp_path / "docs.db"
        reg = DocumentRegistry(db_path)

        doc = Document(
            tenant_id="t1",
            content="Nội dung kiểm thử kho lưu trữ.",
            metadata={"category": "test"},
            access_control=AccessControl(tenant_id="t1", departments=["engineering"]),
        )

        reg.save(doc, idempotency_key="key-1")
        assert reg.count() == 1
        assert reg.count(tenant_id="t1") == 1
        assert reg.count(tenant_id="t2") == 0

        # Get doc
        retrieved = reg.get(doc.document_id, tenant_id="t1")
        assert retrieved is not None
        assert retrieved.content_hash == doc.compute_content_hash()
        assert retrieved.access_control.departments == ["engineering"]

        # Find by idempotency key
        found = reg.find_by_idempotency_key("key-1", tenant_id="t1")
        assert found is not None
        assert found.document_id == doc.document_id

        # Find by content hash
        hash_val = doc.compute_content_hash()
        found_hash = reg.find_by_content_hash(hash_val, "t1")
        assert found_hash is not None
        assert found_hash.document_id == doc.document_id

        # Update status
        updated = reg.update_status(doc.document_id, DocumentStatus.ARCHIVED, tenant_id="t1")
        assert updated is True
        assert reg.get(doc.document_id).status == DocumentStatus.ARCHIVED

        # List by tenant
        docs = reg.list_by_tenant("t1")
        assert len(docs) == 1

        # Delete
        deleted = reg.delete(doc.document_id, tenant_id="t1")
        assert deleted is True
        assert reg.count() == 0

        reg.close()

    def test_clear_tenant(self, tmp_path: Path) -> None:
        db_path = tmp_path / "docs.db"
        reg = DocumentRegistry(db_path)

        reg.save(Document(tenant_id="t1", content="d1"))
        reg.save(Document(tenant_id="t2", content="d2"))

        assert reg.count() == 2
        reg.clear(tenant_id="t1")
        assert reg.count(tenant_id="t1") == 0
        assert reg.count(tenant_id="t2") == 1

        reg.close()
