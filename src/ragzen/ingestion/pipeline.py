"""Ingestion pipeline orchestration.

Handles document loading, validation, chunking, embedding, vector storage,
sparse indexing, registry persistence, deduplication, and idempotency.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, cast

from ragzen.chunkers.recursive import RecursiveChunker
from ragzen.loaders.directory import DirectoryLoader
from ragzen.loaders.documents import UniversalDocumentLoader
from ragzen.models import (
    AccessControl,
    Document,
    DocumentStatus,
    IngestionJob,
    JobStatus,
    SecurityContext,
)
from ragzen.security.context import validate_tenant_access

logger = logging.getLogger("ragzen.ingestion.pipeline")


class IngestionPipeline:
    """Ingestion pipeline for processing documents."""

    def __init__(
        self,
        *,
        document_registry: Any,
        vector_store: Any,
        sparse_index: Any,
        embedding_provider: Any,
        chunker: Any = None,
        graph_index: Any = None,
        collection: str = "documents",
        max_file_size_mb: float = 100.0,
        allowed_abac_keys: list[str] | None = None,
    ) -> None:
        self._registry = document_registry
        self._vector_store = vector_store
        self._sparse_index = sparse_index
        self._embedding = embedding_provider
        self._chunker = chunker or RecursiveChunker()
        self._graph_index = graph_index
        self._collection = collection
        self._lock = threading.RLock()
        self._allowed_abac_keys = set(allowed_abac_keys or [])

        self._document_loader = UniversalDocumentLoader(max_size_mb=max_file_size_mb)
        self._dir_loader = DirectoryLoader(max_size_mb=max_file_size_mb)

    def ingest_path(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        security_context: SecurityContext | None = None,
        idempotency_key: str = "",
    ) -> IngestionJob:
        """Ingest a file or directory path into RagZen.

        Supports idempotency: if idempotency_key matches an existing document/job,
        duplicate processing is avoided.
        """
        source_path = Path(path)
        meta = metadata or {}
        tenant_id = meta.get("tenant_id", "default")

        if security_context:
            validate_tenant_access(security_context, tenant_id)

        job_id = f"job-{idempotency_key}" if idempotency_key else f"job-{source_path.name}"

        if idempotency_key:
            existing = self._registry.find_by_idempotency_key(idempotency_key, tenant_id=tenant_id)
            if existing:
                logger.info(
                    "Idempotency key '%s' found existing document. Skipping ingestion.",
                    idempotency_key,
                )
                return IngestionJob(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    status=JobStatus.COMPLETED,
                    source=str(source_path),
                    total_documents=1,
                    processed_documents=1,
                    idempotency_key=idempotency_key,
                )

        # Load docs
        if source_path.is_dir():
            raw_docs = self._dir_loader.load(source_path)
        else:
            raw_docs = self._document_loader.load(source_path)

        ac = self._build_access_control(meta, tenant_id)

        processed_count = 0
        failed_count = 0

        for raw_doc in raw_docs:
            try:
                document_key = (
                    f"{idempotency_key}:{raw_doc.source_uri}"
                    if idempotency_key and len(raw_docs) > 1
                    else idempotency_key
                )
                if document_key:
                    existing = self._registry.find_by_idempotency_key(
                        document_key, tenant_id=tenant_id
                    )
                    if existing:
                        processed_count += 1
                        continue
                existing_hash = self._registry.find_by_content_hash(
                    raw_doc.content_hash or raw_doc.compute_content_hash(), tenant_id
                )
                if existing_hash:
                    processed_count += 1
                    continue

                doc = Document(
                    tenant_id=tenant_id,
                    content=raw_doc.content,
                    source=raw_doc.source,
                    source_uri=raw_doc.source_uri,
                    file_name=raw_doc.file_name,
                    mime_type=raw_doc.mime_type,
                    metadata=meta,
                    access_control=ac,
                    status=DocumentStatus.PROCESSING,
                )

                self.ingest_document(doc, idempotency_key=document_key)
                processed_count += 1
            except Exception as e:
                logger.exception("Failed to ingest document %s: %s", raw_doc.file_name, e)
                failed_count += 1

        return IngestionJob(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatus.COMPLETED if failed_count == 0 else JobStatus.FAILED,
            source=str(source_path),
            total_documents=len(raw_docs),
            processed_documents=processed_count,
            failed_documents=failed_count,
            idempotency_key=idempotency_key,
        )

    def ingest_text(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        security_context: SecurityContext | None = None,
        idempotency_key: str = "",
    ) -> Document:
        """Ingest raw text directly into RagZen."""
        meta = metadata or {}
        tenant_id = meta.get("tenant_id", "default")

        if security_context:
            validate_tenant_access(security_context, tenant_id)

        with self._lock:
            if idempotency_key:
                existing = self._registry.find_by_idempotency_key(
                    idempotency_key, tenant_id=tenant_id
                )
                if existing:
                    return cast("Document", existing)
            content_hash = Document(tenant_id=tenant_id, content=text).compute_content_hash()
            existing = self._registry.find_by_content_hash(content_hash, tenant_id)
            if existing:
                return cast("Document", existing)

        ac = self._build_access_control(meta, tenant_id)

        doc = Document(
            tenant_id=tenant_id,
            content=text,
            source="text",
            metadata=meta,
            access_control=ac,
            status=DocumentStatus.PROCESSING,
        )

        return self.ingest_document(doc, idempotency_key=idempotency_key)

    def ingest_document(
        self,
        document: Document,
        *,
        idempotency_key: str = "",
    ) -> Document:
        """Persist and index a pre-constructed document with compensating rollback."""
        with self._lock:
            self._registry.save(document, idempotency_key=idempotency_key)
            try:
                chunks = self._chunker.chunk(document)
                embeddings = self._embedding.embed([chunk.content for chunk in chunks])
                if len(embeddings) != len(chunks):
                    msg = (
                        f"Embedding provider returned {len(embeddings)} vectors "
                        f"for {len(chunks)} chunks"
                    )
                    raise ValueError(msg)
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    metadata = self._chunk_metadata(document, chunk)
                    self._vector_store.upsert(self._collection, chunk.chunk_id, embedding, metadata)
                    self._sparse_index.add(chunk.chunk_id, chunk.content, metadata)
                    if self._graph_index is not None:
                        self._graph_index.add(chunk.chunk_id, chunk.content, metadata)
                self._registry.update_status(
                    document.document_id,
                    DocumentStatus.INDEXED,
                    tenant_id=document.tenant_id,
                )
                return document.model_copy(update={"status": DocumentStatus.INDEXED})
            except Exception:
                self._vector_store.delete_by_filter(
                    self._collection, {"document_id": document.document_id}
                )
                if hasattr(self._sparse_index, "remove_by_document_id"):
                    self._sparse_index.remove_by_document_id(document.document_id)
                if self._graph_index is not None:
                    self._graph_index.remove_by_document_id(document.document_id)
                self._registry.update_status(
                    document.document_id,
                    DocumentStatus.FAILED,
                    tenant_id=document.tenant_id,
                )
                raise

    def _build_access_control(self, meta: dict[str, Any], tenant_id: str) -> AccessControl:
        departments = meta.get(
            "departments", [meta["department"]] if meta.get("department") else []
        )
        roles = meta.get("roles", [meta["role"]] if meta.get("role") else [])
        attributes = meta.get("attributes", {})
        unknown_attributes = set(attributes) - self._allowed_abac_keys
        if unknown_attributes:
            msg = (
                "Document ABAC attributes are not declared in security.abac_keys: "
                f"{sorted(unknown_attributes)}"
            )
            raise ValueError(msg)
        return AccessControl(
            tenant_id=tenant_id,
            owner_id=meta.get("owner_id", ""),
            departments=departments,
            roles=roles,
            groups=meta.get("groups", []),
            permissions=meta.get("permissions", []),
            attributes=attributes,
            access_level=meta.get("access_level", "internal"),
        )

    @staticmethod
    def _chunk_metadata(document: Document, chunk: Any) -> dict[str, Any]:
        metadata = {
            **document.metadata,
            "document_id": document.document_id,
            "document_version": document.version,
            "tenant_id": document.tenant_id,
            "content": chunk.content,
            "file_name": document.file_name or "raw_text",
            "source_uri": document.source_uri,
            "page": chunk.page,
        }
        access = document.access_control
        if access:
            metadata.update(
                {
                    "owner_id": access.owner_id,
                    "departments": access.departments,
                    "roles": access.roles,
                    "groups": access.groups,
                    "permissions": access.permissions,
                    "attributes": access.attributes,
                    "access_level": access.access_level.value,
                }
            )
        return metadata
