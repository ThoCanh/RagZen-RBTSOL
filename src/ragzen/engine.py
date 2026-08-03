"""RagZen main engine.

Provides the primary user-facing API:
- RagZen.local()
- RagZen.from_config()
- RagZen.from_components()

Supports both synchronous and asynchronous operations without creating nested event loops.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

from ragzen.cache import MemorySearchCache, RedisSearchCache
from ragzen.chunkers.fixed import FixedSizeChunker
from ragzen.chunkers.recursive import RecursiveChunker
from ragzen.config import RagZenConfig
from ragzen.embeddings.local import DeterministicLocalEmbeddingProvider
from ragzen.exceptions import ConfigurationError
from ragzen.generation.generator import RAGGenerator
from ragzen.graph import KnowledgeGraphIndex
from ragzen.ingestion.pipeline import IngestionPipeline
from ragzen.llms.openai_compatible import OpenAICompatibleLLM
from ragzen.models import (
    AuditEvent,
    Document,
    DocumentStatus,
    DocumentVersion,
    HealthStatus,
    IngestionJob,
    ProviderStatus,
    RagResponse,
    SearchResult,
    SecurityContext,
)
from ragzen.observability.metrics import global_metrics
from ragzen.registry import get_registry
from ragzen.retrieval.hybrid import (
    HybridRetriever,
    ReciprocalRankFusion,
    WeightedScoreFusion,
)
from ragzen.retrieval.rerank import CrossEncoderReranker
from ragzen.security.audit import LogAuditSink
from ragzen.security.context import (
    check_access_control,
    require_security_context,
    validate_tenant_access,
)
from ragzen.security.filters import build_mandatory_filters
from ragzen.security.prompt_injection import PromptInjectionDetector
from ragzen.sparse.bm25 import BM25Index
from ragzen.storage.backup import BackupManager, BundleBackupManager
from ragzen.storage.documents import DocumentRegistry
from ragzen.storage.migrations import MigrationEngine
from ragzen.vectorstores.memory import InMemoryVectorStore
from ragzen.vectorstores.sqlite import SQLiteVectorStore

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
        self.plugins = get_registry()
        self.plugins.discover_entry_points()
        self.cache = self._create_cache()

        # Initialize storage & core components
        storage_path = Path(self.config.storage.path)
        self.registry = document_registry or DocumentRegistry(storage_path)

        self.embedding = embedding or self._create_embedding()
        self.vector_store = vector_store or self._create_vector_store()
        if self.config.sparse_index.provider != "bm25" and sparse_index is None:
            raise ConfigurationError(
                f"Unsupported sparse index provider: {self.config.sparse_index.provider}"
            )
        self.sparse_index = sparse_index or BM25Index(path=self.config.sparse_index.path)
        graph_enabled = self.config.graph.enabled or self.config.retrieval.mode in {
            "graph",
            "hybrid_graph",
        }
        self.graph_index = (
            KnowledgeGraphIndex(
                self.config.graph.path,
                max_hops=self.config.graph.max_hops,
                min_entity_length=self.config.graph.min_entity_length,
            )
            if graph_enabled
            else None
        )

        chunker: Any
        if self.config.chunking.strategy == "fixed":
            chunker = FixedSizeChunker(
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap,
            )
        else:
            chunker = RecursiveChunker(
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap,
            )

        # Initialize retriever
        self.retriever = retriever or HybridRetriever(
            vector_store=self.vector_store,
            sparse_index=self.sparse_index,
            embedding_provider=self.embedding,
            graph_index=self.graph_index,
            collection=self.config.vector_store.collection,
            fusion=(
                WeightedScoreFusion()
                if self.config.retrieval.fusion == "weighted"
                else ReciprocalRankFusion()
            ),
            mode=self.config.retrieval.mode,
            top_k_dense=self.config.retrieval.top_k_dense,
            top_k_sparse=self.config.retrieval.top_k_sparse,
        )
        self.reranker = (
            CrossEncoderReranker(self.config.reranker.model)
            if self.config.reranker.enabled
            else None
        )

        # Initialize LLM provider
        self.llm = llm or self._create_llm()

        # Initialize ingestion pipeline & generator
        self.ingestion = IngestionPipeline(
            document_registry=self.registry,
            vector_store=self.vector_store,
            sparse_index=self.sparse_index,
            embedding_provider=self.embedding,
            graph_index=self.graph_index,
            chunker=chunker,
            collection=self.config.vector_store.collection,
            max_file_size_mb=self.config.security.max_file_size_mb,
            allowed_abac_keys=self.config.security.abac_keys,
        )

        self.generator = RAGGenerator(
            llm=self.llm,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            streaming_enabled=self.config.llm.streaming,
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
        config: RagZenConfig | None = None,
        storage_path: str | None = None,
    ) -> RagZen:
        """Dependency injection factory for custom component composition.

        Returns:
            Initialized RagZen engine.
        """
        resolved_config = config or (
            RagZenConfig.local_default(storage_path) if storage_path else None
        )
        return cls(
            config=resolved_config,
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
        job = self.ingestion.ingest_path(
            path,
            metadata=metadata,
            security_context=ctx,
            idempotency_key=idempotency_key,
        )
        self.cache.clear()
        self._audit(
            "ingestion.completed",
            tenant_id=job.tenant_id,
            resource_id=job.job_id,
            details={"processed": job.processed_documents, "failed": job.failed_documents},
        )
        return job

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
        document = self.ingestion.ingest_text(
            text,
            metadata=metadata,
            security_context=ctx,
            idempotency_key=idempotency_key,
        )
        self.cache.clear()
        self._audit(
            "document.ingested",
            tenant_id=document.tenant_id,
            resource_id=document.document_id,
        )
        return document

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
            if ctx:
                validate_tenant_access(ctx, doc.tenant_id)
            res.append(self.ingestion.ingest_document(doc))
        self.cache.clear()
        return res

    def update(
        self,
        document_id: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
        security_context: SecurityContext | dict[str, Any] | None = None,
    ) -> bool:
        """Update an existing document."""
        ctx = self._normalize_security_context(security_context)
        if self.config.security.require_security_context:
            require_security_context(ctx, require=True)
        doc = self.registry.get(document_id)
        if not doc:
            return False
        if ctx:
            validate_tenant_access(ctx, doc.tenant_id)
        meta = metadata or doc.metadata
        updated = doc.model_copy(
            update={
                "content": content,
                "metadata": {**meta, "tenant_id": doc.tenant_id},
                "version": doc.version + 1,
                "status": DocumentStatus.PROCESSING,
            }
        )
        self.vector_store.delete_by_filter(
            self.config.vector_store.collection, {"document_id": document_id}
        )
        self.sparse_index.remove_by_document_id(document_id)
        if self.graph_index:
            self.graph_index.remove_by_document_id(document_id)
        try:
            self.ingestion.ingest_document(updated)
        except Exception:
            self.ingestion.ingest_document(
                doc.model_copy(update={"status": DocumentStatus.PROCESSING})
            )
            raise
        self.cache.clear()
        return True

    def delete(
        self,
        document_id: str,
        *,
        tenant_id: str = "",
        security_context: SecurityContext | dict[str, Any] | None = None,
    ) -> bool:
        """Delete a document and propagate deletion across all indices and storage."""
        ctx = self._normalize_security_context(security_context)
        if self.config.security.require_security_context:
            require_security_context(ctx, require=True)
        doc = self.registry.get(document_id, tenant_id=tenant_id)
        if not doc:
            return False
        if ctx:
            validate_tenant_access(ctx, doc.tenant_id)

        # Remove from vector store
        self.vector_store.delete_by_filter(
            self.config.vector_store.collection, {"document_id": document_id}
        )
        # Remove from sparse index
        if hasattr(self.sparse_index, "remove_by_document_id"):
            self.sparse_index.remove_by_document_id(document_id)
        else:
            self.sparse_index.remove(document_id)
        if self.graph_index:
            self.graph_index.remove_by_document_id(document_id)
        # Remove from registry
        deleted = self.registry.delete(document_id, tenant_id=tenant_id)
        if deleted:
            self.cache.clear()
            self._audit(
                "document.deleted",
                tenant_id=doc.tenant_id,
                resource_id=document_id,
                user_id=ctx.user_id if ctx else "",
            )
        return deleted

    # --- Query API ---

    def get_document(
        self,
        document_id: str,
        *,
        security_context: SecurityContext | dict[str, Any] | None = None,
    ) -> Document | None:
        ctx = self._normalize_security_context(security_context)
        if self.config.security.require_security_context:
            require_security_context(ctx, require=True)
        document = self.registry.get(document_id, tenant_id=ctx.tenant_id if ctx else "")
        if (
            document
            and ctx
            and document.access_control
            and not check_access_control(
                ctx,
                document.access_control,
                fail_closed=self.config.security.fail_closed,
            )
        ):
            return None
        return document

    def list_documents(
        self,
        *,
        tenant_id: str = "",
        limit: int = 100,
        offset: int = 0,
        security_context: SecurityContext | dict[str, Any] | None = None,
    ) -> list[Document]:
        ctx = self._normalize_security_context(security_context)
        if self.config.security.require_security_context:
            require_security_context(ctx, require=True)
        if ctx:
            if tenant_id:
                validate_tenant_access(ctx, tenant_id)
            tenant_id = ctx.tenant_id
        if not tenant_id:
            raise ConfigurationError("tenant_id is required to list documents")
        documents = self.registry.list_by_tenant(tenant_id, limit=limit, offset=offset)
        if not ctx:
            return documents
        return [
            document
            for document in documents
            if document.access_control is None
            or check_access_control(
                ctx,
                document.access_control,
                fail_closed=self.config.security.fail_closed,
            )
        ]

    def list_versions(
        self,
        document_id: str,
        *,
        security_context: SecurityContext | dict[str, Any] | None = None,
    ) -> list[DocumentVersion]:
        ctx = self._normalize_security_context(security_context)
        if self.config.security.require_security_context:
            require_security_context(ctx, require=True)
        if self.get_document(document_id, security_context=ctx) is None:
            return []
        return self.registry.list_versions(document_id, tenant_id=ctx.tenant_id if ctx else "")

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
            abac_keys=self.config.security.abac_keys,
        )

        cache_key = self._search_cache_key(query, top_k, mandatory_filters, ctx)
        cached = cast("list[SearchResult] | None", self.cache.get(cache_key))
        if cached is not None:
            global_metrics.increment("ragzen_search_cache_hit_total")
            return cached

        started = time.perf_counter()
        retrieval_limit = max(top_k, self.config.retrieval.rerank_top_k) if self.reranker else top_k
        results = self.retriever.retrieve(
            query,
            top_k=retrieval_limit,
            filters=mandatory_filters,
        )
        if self.reranker:
            results = self.reranker.rerank(query, results, top_k=top_k)
        duration = time.perf_counter() - started
        global_metrics.increment("ragzen_search_total")
        global_metrics.observe_latency("ragzen_search", duration)
        self.cache.set(cache_key, results)
        self._audit(
            "retrieval.completed",
            tenant_id=ctx.tenant_id if ctx else "",
            user_id=ctx.user_id if ctx else "",
            details={"result_count": len(results), "top_k": top_k},
        )
        return results

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
                logger.warning("Query rejected due to high-confidence prompt injection pattern")
                return RagResponse(
                    answer="Yêu cầu bị từ chối do vi phạm chính sách an toàn thông tin.",
                    warnings=["Query flagged by prompt injection screening"],
                )

        retrieval_started = time.perf_counter()
        results = self.search(
            question,
            top_k=self.config.retrieval.final_top_k,
            filters=filters,
            security_context=ctx,
        )

        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        response = self.generator.generate(question, results)
        metrics = response.metrics.model_copy(
            update={
                "retrieval_ms": retrieval_ms,
                "total_ms": response.metrics.total_ms + retrieval_ms,
                "retrieved_chunks": len(results),
            }
        )
        global_metrics.increment("ragzen_query_total")
        global_metrics.observe_latency("ragzen_generation", response.metrics.generation_ms / 1000)
        return response.model_copy(
            update={
                "metrics": metrics,
                "retrieval_strategy": self.config.retrieval.mode,
            }
        )

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
        results = await self.asearch(
            question,
            top_k=self.config.retrieval.final_top_k,
            filters=filters,
            security_context=security_context,
        )
        async for token in self.generator.stream(question, results):
            yield token

    # --- System & Diagnostics ---

    def health(self, *, deep: bool = False) -> HealthStatus:
        """Check overall health status of components."""
        db_healthy = True
        try:
            self.registry.count()
        except Exception:
            db_healthy = False

        vs_healthy = self.vector_store.health()
        emb_healthy = getattr(self.embedding, "health_check", lambda: True)() if deep else True
        llm_healthy = getattr(self.llm, "health_check", lambda: True)() if deep else True

        providers = [
            ProviderStatus(name="sqlite_registry", healthy=db_healthy),
            ProviderStatus(name="vector_store", healthy=vs_healthy),
            ProviderStatus(name="embedding_provider", healthy=emb_healthy),
            ProviderStatus(name="llm_provider", healthy=llm_healthy),
        ]

        # Local storage, vector and embedding components determine core health.
        core_healthy = db_healthy and vs_healthy and emb_healthy
        return HealthStatus(
            healthy=core_healthy,
            version=self._version(),
            providers=providers,
            document_count=self.registry.count(),
            chunk_count=self.sparse_index.count(),
        )

    def backup(self, dest_path: str | Path, *, compress: bool = True) -> Path:
        """Backup all local RagZen stores into a consistent bundle."""
        if self.config.vector_store.provider != "sqlite":
            raise ConfigurationError(
                "Bundle backup is available for the local SQLite backend. "
                "Use the external vector provider's snapshot mechanism otherwise."
            )
        manager = self._bundle_backup_manager()
        return manager.backup(dest_path, compress=compress)

    def restore(self, backup_path: str | Path) -> bool:
        """Restore SQLite database from backup file."""
        path = Path(backup_path)
        if path.suffix == ".zip":
            for component in (
                self.graph_index,
                self.sparse_index,
                self.vector_store,
                self.registry,
            ):
                close = getattr(component, "close", None)
                if close:
                    close()
            result = self._bundle_backup_manager().restore(path)
            self._reopen_local_stores()
            return result
        manager = BackupManager(self.config.storage.path)
        self.registry.close()
        result = manager.restore(path)
        self.registry = DocumentRegistry(self.config.storage.path)
        self.ingestion._registry = self.registry
        return result

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
            "vector_store_count": self.vector_store.count(self.config.vector_store.collection),
            "environment": self.config.environment,
            "metrics": global_metrics.get_stats(),
        }

    def clear(
        self,
        *,
        tenant_id: str = "",
        security_context: SecurityContext | dict[str, Any] | None = None,
    ) -> None:
        """Clear documents and indices."""
        ctx = self._normalize_security_context(security_context)
        if self.config.security.require_security_context:
            require_security_context(ctx, require=True)
        if ctx:
            if tenant_id:
                validate_tenant_access(ctx, tenant_id)
            else:
                tenant_id = ctx.tenant_id
        self.registry.clear(tenant_id=tenant_id)
        filters = {"tenant_id": tenant_id} if tenant_id else None
        self.vector_store.clear(
            self.config.vector_store.collection,
            filters=filters,
        )
        self.sparse_index.clear(filters=filters)
        if self.graph_index:
            self.graph_index.clear(filters=filters)
        self.cache.clear()

    def close(self) -> None:
        """Close storage connections and release resources."""
        for component in (
            self.audit_sink,
            self.cache,
            self.graph_index,
            self.sparse_index,
            self.vector_store,
            self.llm,
            self.registry,
        ):
            close = getattr(component, "close", None)
            if close:
                close()

    def __enter__(self) -> RagZen:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _create_embedding(self) -> Any:
        provider = self.config.embedding.provider
        if provider in {"local", "deterministic", "feature_hash"}:
            return DeterministicLocalEmbeddingProvider(
                dimensions=self.config.embedding.dimensions or 384
            )
        if provider == "sentence_transformers":
            from ragzen.embeddings.sentence_transformer import (
                SentenceTransformerEmbeddingProvider,
            )

            if not SentenceTransformerEmbeddingProvider.is_available():
                from ragzen.exceptions import MissingOptionalDependencyError

                raise MissingOptionalDependencyError(
                    "sentence-transformers", "local", "local semantic embeddings"
                )
            return SentenceTransformerEmbeddingProvider(self.config.embedding.model)
        plugin = self._create_plugin("embedding", provider, self.config.embedding)
        if plugin is not None:
            return plugin
        raise ConfigurationError(f"Unsupported embedding provider: {provider}")

    def _create_vector_store(self) -> Any:
        provider = self.config.vector_store.provider
        if provider == "memory":
            return InMemoryVectorStore()
        if provider == "sqlite":
            return SQLiteVectorStore(self.config.vector_store.path)
        if provider == "qdrant":
            from ragzen.vectorstores.qdrant import QdrantVectorStore

            return QdrantVectorStore(
                url=self.config.vector_store.url or "http://localhost:6333",
                api_key=(
                    self.config.vector_store.api_key.get_secret_value()
                    if self.config.vector_store.api_key
                    else ""
                ),
                timeout_seconds=self.config.vector_store.timeout_seconds,
            )
        plugin = self._create_plugin("vector_store", provider, self.config.vector_store)
        if plugin is not None:
            return plugin
        raise ConfigurationError(f"Unsupported vector store provider: {provider}")

    def _create_llm(self) -> Any:
        if self.config.llm.provider in {"extractive", "local"}:
            from ragzen.llms.extractive import ExtractiveLLM

            return ExtractiveLLM()
        if self.config.llm.provider in {"openai_compatible", "openai", "ollama"}:
            import os

            api_key = (
                self.config.llm.api_key.get_secret_value()
                if self.config.llm.api_key
                else os.getenv("OPENAI_API_KEY", "")
            )
            return OpenAICompatibleLLM(
                base_url=self.config.llm.base_url,
                api_key=api_key,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                top_p=self.config.llm.top_p,
                timeout_seconds=self.config.llm.timeout_seconds,
                max_retries=self.config.llm.max_retries,
                concurrency_limit=self.config.llm.concurrency_limit,
            )
        plugin = self._create_plugin("llm", self.config.llm.provider, self.config.llm)
        if plugin is not None:
            return plugin
        raise ConfigurationError(f"Unsupported LLM provider: {self.config.llm.provider}")

    def _create_cache(self) -> Any:
        if self.config.cache.provider == "memory":
            return MemorySearchCache(
                ttl_seconds=self.config.cache.ttl_seconds,
                max_size=self.config.cache.max_size,
            )
        if self.config.cache.provider == "redis":
            if not self.config.cache.url:
                raise ConfigurationError("cache.url is required for Redis cache")
            return RedisSearchCache(
                self.config.cache.url,
                ttl_seconds=self.config.cache.ttl_seconds,
            )
        raise ConfigurationError(f"Unsupported cache provider: {self.config.cache.provider}")

    @staticmethod
    def _search_cache_key(
        query: str,
        top_k: int,
        filters: dict[str, Any],
        context: SecurityContext | None,
    ) -> str:
        payload = {
            "query": query,
            "top_k": top_k,
            "filters": filters,
            "security": context.fingerprint() if context else "public",
        }
        raw = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _create_plugin(self, capability: str, name: str, config: Any) -> Any | None:
        info = self.plugins.get(capability, name)
        if info is None:
            return None
        factory = getattr(info.plugin_class, "from_config", None)
        if factory:
            return factory(config)
        try:
            return info.plugin_class(config=config)
        except TypeError:
            return info.plugin_class()

    def _bundle_backup_manager(self) -> BundleBackupManager:
        sparse_path = Path(self.config.sparse_index.path)
        if not sparse_path.suffix:
            sparse_path = sparse_path / "index.json"
        files: dict[str, str | Path] = {
            "documents.db": self.config.storage.path,
            "vectors.db": self.config.vector_store.path,
            "bm25.json": sparse_path,
        }
        if self.graph_index:
            files["graph.json"] = self.config.graph.path
        return BundleBackupManager(
            files,
            sqlite_names={"documents.db", "vectors.db"},
        )

    def _reopen_local_stores(self) -> None:
        self.registry = DocumentRegistry(self.config.storage.path)
        self.vector_store = self._create_vector_store()
        self.sparse_index = BM25Index(path=self.config.sparse_index.path)
        if self.graph_index:
            self.graph_index = KnowledgeGraphIndex(
                self.config.graph.path,
                max_hops=self.config.graph.max_hops,
                min_entity_length=self.config.graph.min_entity_length,
            )
        self.retriever._vector_store = self.vector_store
        self.retriever._sparse_index = self.sparse_index
        self.retriever._graph_index = self.graph_index
        self.ingestion._registry = self.registry
        self.ingestion._vector_store = self.vector_store
        self.ingestion._sparse_index = self.sparse_index
        self.ingestion._graph_index = self.graph_index
        self.cache.clear()

    def _audit(
        self,
        event_type: str,
        *,
        tenant_id: str = "",
        user_id: str = "",
        resource_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        try:
            self.audit_sink.record(
                AuditEvent(
                    event_type=event_type,
                    tenant_id=tenant_id,
                    user_id=user_id,
                    resource_type="document" if resource_id else "",
                    resource_id=resource_id,
                    details=details or {},
                )
            )
        except Exception:
            logger.exception("Failed to write audit event %s", event_type)

    @staticmethod
    def _version() -> str:
        from ragzen import __version__

        return __version__

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
