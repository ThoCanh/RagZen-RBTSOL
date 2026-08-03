"""Ingestion pipeline orchestration.

Handles document loading, validation, chunking, embedding, vector storage,
sparse indexing, registry persistence, deduplication, and idempotency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ragzen.chunkers.recursive import RecursiveChunker
from ragzen.loaders.directory import DirectoryLoader
from ragzen.loaders.text import TextLoader
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
        collection: str = "documents",
    ) -> None:
        self._registry = document_registry
        self._vector_store = vector_store
        self._sparse_index = sparse_index
        self._embedding = embedding_provider
        self._chunker = chunker or RecursiveChunker()
        self._collection = collection

        self._text_loader = TextLoader()
        self._dir_loader = DirectoryLoader()

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
            raw_docs = self._text_loader.load(source_path)

        depts = meta.get("departments", [meta.get("department")] if meta.get("department") else [])
        roles = meta.get("roles", [meta.get("role")] if meta.get("role") else [])
        ac = AccessControl(
            tenant_id=tenant_id,
            departments=depts,
            roles=roles,
            access_level=meta.get("access_level", "internal"),
        )

        processed_count = 0
        failed_count = 0

        for raw_doc in raw_docs:
            try:
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

                # Save doc to registry
                self._registry.save(doc, idempotency_key=idempotency_key)

                # Chunk doc
                chunks = self._chunker.chunk(doc)

                if chunks:
                    chunk_texts = [c.content for c in chunks]
                    embeddings = self._embedding.embed(chunk_texts)

                    # Vector store & Sparse index upsert
                    for chunk, emb in zip(chunks, embeddings, strict=False):
                        chunk_meta = {
                            "document_id": doc.document_id,
                            "tenant_id": tenant_id,
                            "content": chunk.content,
                            "file_name": doc.file_name,
                            "page": chunk.page,
                            **doc.metadata,
                        }
                        if ac:
                            chunk_meta["departments"] = ac.departments
                            chunk_meta["roles"] = ac.roles

                        self._vector_store.upsert(self._collection, chunk.chunk_id, emb, chunk_meta)
                        self._sparse_index.add(chunk.chunk_id, chunk.content, chunk_meta)

                self._registry.update_status(
                    doc.document_id, DocumentStatus.INDEXED, tenant_id=tenant_id
                )
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

        depts = meta.get("departments", [meta.get("department")] if meta.get("department") else [])
        roles = meta.get("roles", [meta.get("role")] if meta.get("role") else [])
        ac = AccessControl(
            tenant_id=tenant_id,
            departments=depts,
            roles=roles,
            access_level=meta.get("access_level", "internal"),
        )

        doc = Document(
            tenant_id=tenant_id,
            content=text,
            source="text",
            metadata=meta,
            access_control=ac,
            status=DocumentStatus.PROCESSING,
        )

        self._registry.save(doc, idempotency_key=idempotency_key)
        chunks = self._chunker.chunk(doc)

        if chunks:
            chunk_texts = [c.content for c in chunks]
            embeddings = self._embedding.embed(chunk_texts)

            for chunk, emb in zip(chunks, embeddings, strict=False):
                chunk_meta = {
                    "document_id": doc.document_id,
                    "tenant_id": tenant_id,
                    "content": chunk.content,
                    "file_name": "raw_text",
                    **doc.metadata,
                }
                if ac:
                    chunk_meta["departments"] = ac.departments
                    chunk_meta["roles"] = ac.roles

                self._vector_store.upsert(self._collection, chunk.chunk_id, emb, chunk_meta)
                self._sparse_index.add(chunk.chunk_id, chunk.content, chunk_meta)

        self._registry.update_status(doc.document_id, DocumentStatus.INDEXED, tenant_id=tenant_id)
        return doc
