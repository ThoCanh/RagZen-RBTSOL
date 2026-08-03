"""RagZen main engine.

Provides the primary user-facing API:
- RagZen.local()
- RagZen.from_config()
- RagZen.from_components()

Supports both synchronous and asynchronous operations without creating nested event loops.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from ragzen.config import RagZenConfig
from ragzen.generation.generator import RAGGenerator
from ragzen.ingestion.pipeline import IngestionPipeline
from ragzen.llms.openai_compatible import OpenAICompatibleLLM
from ragzen.models import (
    Document,
    HealthStatus,
    IngestionJob,
    ProviderStatus,
    RagResponse,
    SearchResult,
    SecurityContext,
)
from ragzen.observability.metrics import global_metrics
from ragzen.retrieval.hybrid import HybridRetriever
from ragzen.security.audit import LogAuditSink
from ragzen.security.context import require_security_context
from ragzen.security.filters import build_mandatory_filters
from ragzen.security.prompt_injection import PromptInjectionDetector
from ragzen.sparse.bm25 import BM25Index
from ragzen.storage.backup import BackupManager
from ragzen.storage.documents import DocumentRegistry
from ragzen.storage.migrations import MigrationEngine
from ragzen.vectorstores.memory import InMemoryVectorStore

logger = logging.getLogger("ragzen.engine")


class RagZen:
    """Enterprise-grade, local-first RAG engine."""

    def __init__(
        self,
        *,
        config: RagZenConfig | None = None,
        document_registry: Any = None,
        embedding: Any = None,
        vector_store: Any = None,
        sparse_index: Any = None,
        retriever: Any = None,
        llm: Any = None,
        audit_sink: Any = None,
    ) -> None:
        self.config = config or RagZenConfig.local_default()

        # Initialize storage & core components
        storage_path = Path(self.config.storage.path)
        self.registry = document_registry or DocumentRegistry(storage_path)

        # Initialize embedding provider (Default: SentenceTransformer if installed -> DeterministicLocal)
        if embedding:
            self.embedding = embedding
        else:
            from ragzen.embeddings.sentence_transformer import (
                SentenceTransformerEmbeddingProvider,
            )

            if SentenceTransformerEmbeddingProvider.is_available():
                self.embedding = SentenceTransformerEmbeddingProvider()
            else:
                from ragzen.embeddings.local import DeterministicLocalEmbeddingProvider

                self.embedding = DeterministicLocalEmbeddingProvider()

        # Initialize vector store & sparse index
        self.vector_store = vector_store or InMemoryVectorStore()
        self.sparse_index = sparse_index or BM25Index()

        # Initialize retriever
        self.retriever = retriever or HybridRetriever(
            vector_store=self.vector_store,
            sparse_index=self.sparse_index,
            embedding_provider=self.embedding,
            top_k_dense=self.config.retrieval.top_k_dense,
            top_k_sparse=self.config.retrieval.top_k_sparse,
        )

        # Initialize LLM provider
        if llm:
            self.llm = llm
        else:
            import os

            api_key = (
                self.config.llm.api_key.get_secret_value()
                if self.config.llm.api_key
                else os.getenv("OPENAI_API_KEY", "")
            )
            self.llm = OpenAICompatibleLLM(
                base_url=self.config.llm.base_url,
                api_key=api_key,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                timeout_seconds=5.0,
            )

        # Initialize ingestion pipeline & generator
        self.ingestion = IngestionPipeline(
            document_registry=self.registry,
            vector_store=self.vector_store,
            sparse_index=self.sparse_index,
            embedding_provider=self.embedding,
        )

        self.generator = RAGGenerator(
            llm=self.llm,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )

        self.audit_sink = audit_sink or LogAuditSink()
        self.prompt_detector = PromptInjectionDetector()

    # --- Factory Methods ---

    @classmethod
    def local(cls, storage_path: str = ".ragzen") -> RagZen:
        """Quick start factory for local mode.

        Args:
            storage_path: Path to directory where SQLite and indexes are stored.

        Returns:
            Initialized RagZen engine.
        """
        cfg = RagZenConfig.local_default(storage_path)
        return cls(config=cfg)

    @classmethod
    def from_config(cls, config_path: str | Path) -> RagZen:
        """Initialize RagZen from a YAML configuration file.

        Args:
            config_path: Path to ragzen.yaml

        Returns:
            Initialized RagZen engine.
        """
        cfg = RagZenConfig.from_yaml(config_path)
        return cls(config=cfg)

    @classmethod
    def from_components(
        cls,
        *,
        document_registry: Any = None,
        embedding: Any = None,
        vector_store: Any = None,
        sparse_index: Any = None,
        retriever: Any = None,
        llm: Any = None,
        audit_sink: Any = None,
    ) -> RagZen:
        """Dependency injection factory for custom component composition.

        Returns:
            Initialized RagZen engine.
        """
        return cls(
            document_registry=document_registry,
            embedding=embedding,
            vector_store=vector_store,
            sparse_index=sparse_index,
            retriever=retriever,
            llm=llm,
            audit_sink=audit_sink,
        )

    # --- Ingestion API ---

    def add(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        security_context: SecurityContext | dict[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> IngestionJob:
        """Ingest a file or directory into RagZen.

        Args:
            path: Path to document or directory.
            metadata: Metadata dict (e.g. tenant_id, department, access_level).
            security_context: Security context object or dict.
            idempotency_key: Unique key for idempotent ingestion.

        Returns:
            IngestionJob tracking job status.
        """
        ctx = self._normalize_security_context(security_context)
        return self.ingestion.ingest_path(
            path,
            metadata=metadata,
            security_context=ctx,
            idempotency_key=idempotency_key,
        )

    async def aadd(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        security_context: SecurityContext | dict[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> IngestionJob:
        """Async version of add()."""
        return await asyncio.to_thread(
            self.add,
            path,
            metadata=metadata,
            security_context=security_context,
            idempotency_key=idempotency_key,
        )

    def add_text(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        security_context: SecurityContext | dict[str, Any] | None = None,
        idempotency_key: str = "",
    ) -> Document:
        """Ingest raw text directly into RagZen."""
        ctx = self._normalize_security_context(security_context)
        return self.ingestion.ingest_text(
            text,
            metadata=metadata,
            security_context=ctx,
            idempotency_key=idempotency_key,
        )

    def add_documents(
        self,
        documents: list[Document],
        *,
        security_context: SecurityContext | dict[str, Any] | None = None,
    ) -> list[Document]:
        """Ingest pre-constructed Document objects."""
        ctx = self._normalize_security_context(security_context)
        res = []
        for doc in documents:
            res.append(
                self.add_text(
                    doc.content,
                    metadata={"tenant_id": doc.tenant_id, **doc.metadata},
                    security_context=ctx,
                )
            )
        return res

    def update(
        self, document_id: str, content: str, *, metadata: dict[str, Any] | None = None
    ) -> bool:
        """Update an existing document."""
        doc = self.registry.get(document_id)
        if not doc:
            return False
        meta = metadata or doc.metadata
        self.delete(document_id, tenant_id=doc.tenant_id)
        self.add_text(content, metadata={"tenant_id": doc.tenant_id, **meta})
        return True

    def delete(self, document_id: str, *, tenant_id: str = "") -> bool:
        """Delete a document and propagate deletion across all indices and storage."""
        doc = self.registry.get(document_id, tenant_id=tenant_id)
        if not doc:
            return False

        # Remove from vector store
        self.vector_store.delete_by_filter("documents", {"document_id": document_id})
        # Remove from sparse index
        if hasattr(self.sparse_index, "remove_by_document_id"):
            self.sparse_index.remove_by_document_id(document_id)
        else:
            self.sparse_index.remove(document_id)
        # Remove from registry
        return self.registry.delete(document_id, tenant_id=tenant_id)

    # --- Query API ---

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        security_context: SecurityContext | dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for relevant documents/chunks.

        Permission filters are built from security_context and applied AT THE STORAGE LAYER.
        """
        ctx = self._normalize_security_context(security_context)

        if self.config.security.require_security_context:
            require_security_context(ctx, require=True)

        mandatory_filters = build_mandatory_filters(
            ctx,
            additional_filters=filters,
            require_context=self.config.security.require_security_context,
        )

        return self.retriever.retrieve(
            query,
            top_k=top_k,
            filters=mandatory_filters,
        )

    async def asearch(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        security_context: SecurityContext | dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Async version of search()."""
        return await asyncio.to_thread(
            self.search,
            query,
            top_k=top_k,
            filters=filters,
            security_context=security_context,
        )

    def ask(
        self,
        question: str,
        *,
        filters: dict[str, Any] | None = None,
        security_context: SecurityContext | dict[str, Any] | None = None,
    ) -> RagResponse:
        """Perform a permission-aware RAG query and generate an answer with citations.

        Args:
            question: Query text.
            filters: Additional metadata filters.
            security_context: Security context object or dict.

        Returns:
            RagResponse object.
        """
        ctx = self._normalize_security_context(security_context)

        # Check prompt injection
        if self.config.security.prompt_injection_screening:
            inj_check = self.prompt_detector.check(question)
            if inj_check.is_suspicious and inj_check.confidence >= 0.85:
                logger.warning(
                    "Query rejected due to high-confidence prompt injection pattern: %s",
                    question,
                )
                return RagResponse(
                    answer="Yêu cầu bị từ chối do vi phạm chính sách an toàn thông tin.",
                    warnings=["Query flagged by prompt injection screening"],
                )

        results = self.search(
            question,
            top_k=self.config.retrieval.final_top_k,
            filters=filters,
            security_context=ctx,
        )

        return self.generator.generate(question, results)

    async def aask(
        self,
        question: str,
        *,
        filters: dict[str, Any] | None = None,
        security_context: SecurityContext | dict[str, Any] | None = None,
    ) -> RagResponse:
        """Async version of ask()."""
        return await asyncio.to_thread(
            self.ask,
            question,
            filters=filters,
            security_context=security_context,
        )

    async def stream(
        self,
        question: str,
        *,
        filters: dict[str, Any] | None = None,
        security_context: SecurityContext | dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens (yields chunks of text)."""
        resp = await self.aask(question, filters=filters, security_context=security_context)
        # Yield answer in chunks to simulate streaming
        words = resp.answer.split(" ")
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)

    # --- System & Diagnostics ---

    def health(self) -> HealthStatus:
        """Check overall health status of components."""
        db_healthy = True
        try:
            self.registry.count()
        except Exception:
            db_healthy = False

        vs_healthy = self.vector_store.health()
        emb_healthy = getattr(self.embedding, "health_check", lambda: True)()
        llm_healthy = getattr(self.llm, "health_check", lambda: True)()

        providers = [
            ProviderStatus(name="sqlite_registry", healthy=db_healthy),
            ProviderStatus(name="vector_store", healthy=vs_healthy),
            ProviderStatus(name="embedding_provider", healthy=emb_healthy),
            ProviderStatus(name="llm_provider", healthy=llm_healthy),
        ]

        # Core local RAG engine components (registry, vector store, embedding) determine core engine health
        core_healthy = db_healthy and vs_healthy and emb_healthy
        return HealthStatus(
            healthy=core_healthy,
            version="0.1.0",
            providers=providers,
            document_count=self.registry.count(),
            chunk_count=self.sparse_index.count(),
        )

    def backup(self, dest_path: str | Path, *, compress: bool = True) -> Path:
        """Backup SQLite document registry database."""
        mgr = BackupManager(self.config.storage.path)
        return mgr.backup(dest_path, compress=compress)

    def restore(self, backup_path: str | Path) -> bool:
        """Restore SQLite database from backup file."""
        mgr = BackupManager(self.config.storage.path)
        # Close active connection before restoring
        self.registry.close()
        res = mgr.restore(backup_path)
        # Re-open connection
        self.registry = DocumentRegistry(self.config.storage.path)
        return res

    def migrate(self, action: str = "apply") -> dict[str, Any]:
        """Run database migrations (plan, apply, status)."""
        engine = MigrationEngine(self.config.storage.path)
        if action == "plan":
            return {"pending_migrations": [m.name for m in engine.plan()]}
        if action == "status":
            return engine.status()
        applied = engine.apply()
        return {"applied_count": applied, "current_version": engine.current_version()}

    def stats(self) -> dict[str, Any]:
        """Return engine statistics and telemetry metrics."""
        return {
            "document_count": self.registry.count(),
            "indexed_chunk_count": self.sparse_index.count(),
            "vector_store_count": self.vector_store.count("documents"),
            "environment": self.config.environment,
            "metrics": global_metrics.get_stats(),
        }

    def clear(self, *, tenant_id: str = "") -> None:
        """Clear documents and indices."""
        self.registry.clear(tenant_id=tenant_id)
        self.vector_store.clear("documents")
        self.sparse_index.clear()

    def close(self) -> None:
        """Close storage connections and release resources."""
        if hasattr(self.registry, "close"):
            self.registry.close()

    # --- Helpers ---

    def _normalize_security_context(
        self,
        security_context: SecurityContext | dict[str, Any] | None,
    ) -> SecurityContext | None:
        """Convert dict or SecurityContext object into SecurityContext instance."""
        if security_context is None:
            return None
        if isinstance(security_context, SecurityContext):
            return security_context
        if isinstance(security_context, dict):
            return SecurityContext.model_validate(security_context)
        return None
