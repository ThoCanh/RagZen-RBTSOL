"""RagZen core domain models.

All models use Pydantic v2 with strict validation. Models that represent
stored data are frozen (immutable) to prevent accidental mutation.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- Enums ---


class DocumentStatus(StrEnum):
    """Lifecycle status of a document."""

    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"
    ARCHIVED = "archived"


class JobStatus(StrEnum):
    """Status of an ingestion or background job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class RetentionAction(StrEnum):
    """Action to take when retention period expires."""

    DELETE = "delete"
    ARCHIVE = "archive"
    NOTIFY = "notify"


class AccessLevel(StrEnum):
    """Document access level classification."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


# --- Helper functions ---


def _utc_now() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(UTC)


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# --- Access Control ---


class AccessControl(BaseModel):
    """Access control metadata attached to documents and chunks."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    owner_id: str = ""
    departments: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    access_level: AccessLevel = AccessLevel.INTERNAL

    @field_validator("tenant_id")
    @classmethod
    def tenant_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            msg = "tenant_id must not be empty"
            raise ValueError(msg)
        return v.strip()


class RetentionPolicy(BaseModel):
    """Data retention policy for a document."""

    model_config = ConfigDict(frozen=True)

    retention_days: int | None = None
    action: RetentionAction = RetentionAction.DELETE
    expires_at: datetime | None = None


# --- Core Domain ---


class Document(BaseModel):
    """Represents an ingested document in the registry."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(default_factory=_new_id)
    version: int = 1
    tenant_id: str
    content: str = ""
    content_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = ""
    source_uri: str = ""
    file_name: str = ""
    mime_type: str = "text/plain"
    page_count: int = 0
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    indexed_at: datetime | None = None
    document_type: str = ""
    access_control: AccessControl | None = None
    retention_policy: RetentionPolicy | None = None
    status: DocumentStatus = DocumentStatus.PENDING

    @field_validator("tenant_id")
    @classmethod
    def tenant_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            msg = "tenant_id must not be empty"
            raise ValueError(msg)
        return v.strip()

    def compute_content_hash(self) -> str:
        """Compute SHA-256 hash of the document content."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class DocumentVersion(BaseModel):
    """Version record for document change tracking."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    version: int
    content_hash: str
    tenant_id: str = ""
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    created_by: str = ""
    change_reason: str = ""


class Chunk(BaseModel):
    """A chunk of a document with position and access control metadata."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(default_factory=_new_id)
    document_id: str
    document_version: int = 1
    content: str
    content_hash: str = ""
    start_offset: int = 0
    end_offset: int = 0
    page: int | None = None
    sequence: int = 0
    parent_chunk_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    access_control: AccessControl | None = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            msg = "Chunk content must not be empty"
            raise ValueError(msg)
        return v

    def compute_content_hash(self) -> str:
        """Compute SHA-256 hash of the chunk content."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class EmbeddingRecord(BaseModel):
    """Associates a chunk with its embedding vector."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    embedding: list[float]
    model_name: str
    model_version: str = ""
    dimensions: int
    created_at: datetime = Field(default_factory=_utc_now)

    @field_validator("dimensions")
    @classmethod
    def dimensions_positive(cls, v: int) -> int:
        if v <= 0:
            msg = "dimensions must be positive"
            raise ValueError(msg)
        return v


# --- Search & Response ---


class Citation(BaseModel):
    """A verified citation linking a claim to a source chunk."""

    model_config = ConfigDict(frozen=True)

    citation_id: str = Field(default_factory=_new_id)
    document_id: str
    chunk_id: str
    page: int | None = None
    score: float = 0.0
    file_name: str = ""
    content_snippet: str = ""
    valid: bool = True
    warning: str = ""


class SearchResult(BaseModel):
    """A single search result with score and source information."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    content: str
    score: float
    page: int | None = None
    file_name: str = ""
    source_uri: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    citation_id: str = ""
    retrieval_method: str = ""


class QueryMetrics(BaseModel):
    """Performance metrics for a single query."""

    model_config = ConfigDict(frozen=True)

    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    retrieved_chunks: int = 0
    reranked_chunks: int = 0
    final_chunks: int = 0
    tokens_used: int = 0
    cache_hit: bool = False


class RagResponse(BaseModel):
    """Complete response from a RAG query."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(default_factory=_new_id)
    answer: str
    sources: list[SearchResult] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    metrics: QueryMetrics = Field(default_factory=QueryMetrics)
    model: str = ""
    retrieval_strategy: str = ""
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)


class QueryRequest(BaseModel):
    """Incoming query request."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(default_factory=_new_id)
    query: str
    tenant_id: str = ""
    user_id: str = ""
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=10, ge=1, le=1000)
    rerank: bool = True
    stream: bool = False

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            msg = "query must not be empty"
            raise ValueError(msg)
        return v


# --- Security ---


class SecurityContext(BaseModel):
    """Security context for permission-aware operations.

    Immutable after creation. Used to generate mandatory filters
    and cache keys.
    """

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    user_id: str = ""
    roles: list[str] = Field(default_factory=list)
    departments: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tenant_id")
    @classmethod
    def tenant_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            msg = "tenant_id must not be empty"
            raise ValueError(msg)
        return v.strip()

    def fingerprint(self) -> str:
        """Generate a deterministic fingerprint for cache keying.

        Includes tenant, user, roles, departments, groups, permissions.
        """
        parts = [
            f"t:{self.tenant_id}",
            f"u:{self.user_id}",
            f"r:{','.join(sorted(self.roles))}",
            f"d:{','.join(sorted(self.departments))}",
            f"g:{','.join(sorted(self.groups))}",
            f"p:{','.join(sorted(self.permissions))}",
            f"a:{json.dumps(self.attributes, sort_keys=True, default=str)}",
        ]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# --- Audit ---


class AuditEvent(BaseModel):
    """Structured audit event for compliance and debugging."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=_new_id)
    event_type: str
    timestamp: datetime = Field(default_factory=_utc_now)
    tenant_id: str = ""
    user_id: str = ""
    resource_type: str = ""
    resource_id: str = ""
    action: str = ""
    status: str = "success"
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str = ""
    trace_id: str = ""


# --- Jobs ---


class IngestionJob(BaseModel):
    """Tracks the status of an ingestion job."""

    model_config = ConfigDict(frozen=True)

    job_id: str = Field(default_factory=_new_id)
    tenant_id: str = ""
    status: JobStatus = JobStatus.PENDING
    source: str = ""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    idempotency_key: str = ""
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None
    error_message: str = ""
    errors: list[dict[str, Any]] = Field(default_factory=list)


# --- Health ---


class ProviderStatus(BaseModel):
    """Health status of a single provider."""

    model_config = ConfigDict(frozen=True)

    name: str
    healthy: bool
    latency_ms: float = 0.0
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    """Aggregated health status of all RagZen components."""

    model_config = ConfigDict(frozen=True)

    healthy: bool
    version: str = ""
    uptime_seconds: float = 0.0
    providers: list[ProviderStatus] = Field(default_factory=list)
    document_count: int = 0
    chunk_count: int = 0
    checked_at: datetime = Field(default_factory=_utc_now)
