"""Tests for RagZen core domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragzen.models import (
    AccessControl,
    AccessLevel,
    AuditEvent,
    Chunk,
    Citation,
    Document,
    DocumentStatus,
    EmbeddingRecord,
    HealthStatus,
    IngestionJob,
    JobStatus,
    ProviderStatus,
    QueryMetrics,
    QueryRequest,
    RagResponse,
    SearchResult,
    SecurityContext,
)


class TestDocument:
    """Tests for Document model."""

    def test_create_document(self) -> None:
        doc = Document(tenant_id="tenant-1", content="Hello world")
        assert doc.tenant_id == "tenant-1"
        assert doc.content == "Hello world"
        assert doc.status == DocumentStatus.PENDING
        assert doc.document_id  # auto-generated
        assert doc.version == 1

    def test_document_requires_tenant_id(self) -> None:
        with pytest.raises(ValidationError, match="tenant_id"):
            Document(tenant_id="", content="test")

    def test_document_tenant_id_stripped(self) -> None:
        doc = Document(tenant_id="  tenant-1  ", content="test")
        assert doc.tenant_id == "tenant-1"

    def test_document_is_frozen(self) -> None:
        doc = Document(tenant_id="tenant-1", content="test")
        with pytest.raises(ValidationError):
            doc.content = "modified"  # type: ignore[misc]

    def test_document_content_hash(self) -> None:
        doc = Document(tenant_id="t", content="Hello")
        h = doc.compute_content_hash()
        assert len(h) == 64  # SHA-256 hex
        # Deterministic
        assert doc.compute_content_hash() == h

    def test_document_serialization(self) -> None:
        doc = Document(tenant_id="t1", content="data", file_name="test.txt")
        data = doc.model_dump()
        assert data["tenant_id"] == "t1"
        assert data["file_name"] == "test.txt"
        # Round-trip
        doc2 = Document.model_validate(data)
        assert doc2.document_id == doc.document_id

    def test_document_all_fields(self) -> None:
        ac = AccessControl(tenant_id="t1", access_level=AccessLevel.CONFIDENTIAL)
        doc = Document(
            tenant_id="t1",
            content="confidential data",
            source="upload",
            source_uri="file:///test.pdf",
            file_name="test.pdf",
            mime_type="application/pdf",
            page_count=5,
            document_type="report",
            access_control=ac,
            status=DocumentStatus.INDEXED,
        )
        assert doc.access_control is not None
        assert doc.access_control.access_level == AccessLevel.CONFIDENTIAL
        assert doc.page_count == 5


class TestChunk:
    """Tests for Chunk model."""

    def test_create_chunk(self) -> None:
        chunk = Chunk(document_id="doc-1", content="chunk text", sequence=0)
        assert chunk.document_id == "doc-1"
        assert chunk.content == "chunk text"
        assert chunk.chunk_id  # auto-generated

    def test_chunk_requires_content(self) -> None:
        with pytest.raises(ValidationError, match="content"):
            Chunk(document_id="doc-1", content="")

    def test_chunk_whitespace_only_content(self) -> None:
        with pytest.raises(ValidationError, match="content"):
            Chunk(document_id="doc-1", content="   ")

    def test_chunk_is_frozen(self) -> None:
        chunk = Chunk(document_id="d", content="text")
        with pytest.raises(ValidationError):
            chunk.content = "new"  # type: ignore[misc]

    def test_chunk_content_hash(self) -> None:
        chunk = Chunk(document_id="d", content="test")
        h = chunk.compute_content_hash()
        assert len(h) == 64

    def test_chunk_with_access_control(self) -> None:
        ac = AccessControl(tenant_id="t1", departments=["eng"])
        chunk = Chunk(
            document_id="d",
            content="text",
            access_control=ac,
            page=3,
            start_offset=100,
            end_offset=200,
        )
        assert chunk.access_control is not None
        assert chunk.access_control.departments == ["eng"]
        assert chunk.page == 3


class TestSecurityContext:
    """Tests for SecurityContext model."""

    def test_create_security_context(self) -> None:
        ctx = SecurityContext(
            tenant_id="company-a",
            user_id="user-1",
            roles=["admin"],
            departments=["it"],
        )
        assert ctx.tenant_id == "company-a"
        assert ctx.user_id == "user-1"
        assert "admin" in ctx.roles

    def test_security_context_requires_tenant(self) -> None:
        with pytest.raises(ValidationError, match="tenant_id"):
            SecurityContext(tenant_id="", user_id="u1")

    def test_security_context_is_frozen(self) -> None:
        ctx = SecurityContext(tenant_id="t1")
        with pytest.raises(ValidationError):
            ctx.tenant_id = "t2"  # type: ignore[misc]

    def test_fingerprint_deterministic(self) -> None:
        ctx = SecurityContext(
            tenant_id="t1",
            user_id="u1",
            roles=["admin", "viewer"],
            departments=["hr"],
        )
        fp1 = ctx.fingerprint()
        fp2 = ctx.fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_fingerprint_role_order_independent(self) -> None:
        ctx1 = SecurityContext(tenant_id="t1", roles=["admin", "viewer"])
        ctx2 = SecurityContext(tenant_id="t1", roles=["viewer", "admin"])
        assert ctx1.fingerprint() == ctx2.fingerprint()

    def test_fingerprint_different_tenants(self) -> None:
        ctx1 = SecurityContext(tenant_id="t1", user_id="u1")
        ctx2 = SecurityContext(tenant_id="t2", user_id="u1")
        assert ctx1.fingerprint() != ctx2.fingerprint()


class TestAccessControl:
    """Tests for AccessControl model."""

    def test_create_access_control(self) -> None:
        ac = AccessControl(
            tenant_id="t1",
            departments=["finance"],
            access_level=AccessLevel.RESTRICTED,
        )
        assert ac.tenant_id == "t1"
        assert ac.access_level == AccessLevel.RESTRICTED

    def test_access_control_requires_tenant(self) -> None:
        with pytest.raises(ValidationError, match="tenant_id"):
            AccessControl(tenant_id="")


class TestRagResponse:
    """Tests for RagResponse model."""

    def test_create_response(self) -> None:
        resp = RagResponse(
            answer="Test answer",
            model="test-model",
            retrieval_strategy="hybrid",
        )
        assert resp.answer == "Test answer"
        assert resp.request_id  # auto-generated
        assert resp.created_at is not None

    def test_response_with_sources(self) -> None:
        source = SearchResult(
            chunk_id="c1",
            document_id="d1",
            content="source text",
            score=0.95,
            page=1,
        )
        resp = RagResponse(
            answer="Answer with source",
            sources=[source],
        )
        assert len(resp.sources) == 1
        assert resp.sources[0].score == 0.95

    def test_response_with_citations(self) -> None:
        citation = Citation(
            document_id="d1",
            chunk_id="c1",
            page=1,
            score=0.9,
        )
        resp = RagResponse(
            answer="Answer",
            citations=[citation],
        )
        assert len(resp.citations) == 1
        assert resp.citations[0].valid is True


class TestQueryRequest:
    """Tests for QueryRequest model."""

    def test_create_query(self) -> None:
        q = QueryRequest(query="test query", tenant_id="t1")
        assert q.query == "test query"
        assert q.top_k == 10  # default

    def test_query_requires_text(self) -> None:
        with pytest.raises(ValidationError, match="query"):
            QueryRequest(query="")

    def test_query_top_k_bounds(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=1001)


class TestEmbeddingRecord:
    """Tests for EmbeddingRecord model."""

    def test_create_embedding_record(self) -> None:
        rec = EmbeddingRecord(
            chunk_id="c1",
            document_id="d1",
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model",
            dimensions=3,
        )
        assert len(rec.embedding) == 3
        assert rec.dimensions == 3

    def test_embedding_requires_positive_dimensions(self) -> None:
        with pytest.raises(ValidationError, match="dimensions"):
            EmbeddingRecord(
                chunk_id="c1",
                document_id="d1",
                embedding=[],
                model_name="m",
                dimensions=0,
            )


class TestAuditEvent:
    """Tests for AuditEvent model."""

    def test_create_audit_event(self) -> None:
        event = AuditEvent(
            event_type="document_ingested",
            tenant_id="t1",
            resource_type="document",
            resource_id="d1",
            action="create",
        )
        assert event.event_type == "document_ingested"
        assert event.status == "success"
        assert event.event_id  # auto-generated


class TestIngestionJob:
    """Tests for IngestionJob model."""

    def test_create_job(self) -> None:
        job = IngestionJob(
            tenant_id="t1",
            source="./docs",
            total_documents=10,
        )
        assert job.status == JobStatus.PENDING
        assert job.total_documents == 10


class TestHealthStatus:
    """Tests for HealthStatus model."""

    def test_healthy_status(self) -> None:
        provider = ProviderStatus(name="sqlite", healthy=True, latency_ms=1.5)
        health = HealthStatus(
            healthy=True,
            version="0.1.0",
            providers=[provider],
        )
        assert health.healthy is True
        assert len(health.providers) == 1


class TestQueryMetrics:
    """Tests for QueryMetrics model."""

    def test_default_metrics(self) -> None:
        m = QueryMetrics()
        assert m.total_ms == 0.0
        assert m.cache_hit is False


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_result(self) -> None:
        r = SearchResult(
            chunk_id="c1",
            document_id="d1",
            content="found text",
            score=0.87,
            page=2,
            file_name="report.pdf",
        )
        assert r.score == 0.87
        assert r.page == 2
