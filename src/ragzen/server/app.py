"""FastAPI server application for RagZen (Server Mode).

Endpoints:
- POST /v1/documents
- POST /v1/documents/text
- DELETE /v1/documents/{document_id}
- POST /v1/search
- POST /v1/query
- POST /v1/query/stream (SSE)
- GET /health/live
- GET /health/ready
- GET /metrics
"""

from __future__ import annotations

import hmac
import json
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from ragzen import __version__
from ragzen.engine import RagZen
from ragzen.exceptions import (
    ConfigurationError,
    MissingOptionalDependencyError,
    TenantIsolationError,
)
from ragzen.models import SecurityContext
from ragzen.observability.metrics import global_metrics
from ragzen.server.schemas import (
    IngestRequest,
    IngestTextRequest,
    QueryApiRequest,
    SearchApiRequest,
)

logger = logging.getLogger("ragzen.server")


def create_app(engine: RagZen | None = None) -> Any:
    """Create FastAPI application.

    Raises:
        MissingOptionalDependencyError: If fastapi is not installed.
    """
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import PlainTextResponse, StreamingResponse
    except ImportError as e:
        raise MissingOptionalDependencyError("fastapi", "server", "RagZen Server Mode") from e

    rag_engine = engine or RagZen.local()
    if rag_engine.config.environment == "production" and not rag_engine.config.server.principals:
        raise ConfigurationError(
            "Production server mode requires at least one server.principals entry"
        )

    def resolve_context(
        requested: dict[str, Any], authorization: str | None
    ) -> SecurityContext | None:
        principals = rag_engine.config.server.principals
        if principals:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Bearer API key required")
            token = authorization.removeprefix("Bearer ").strip()
            principal = next(
                (
                    item
                    for item in principals
                    if hmac.compare_digest(token, item.api_key.get_secret_value())
                ),
                None,
            )
            if principal is None:
                raise HTTPException(status_code=401, detail="Invalid API key")
            if requested.get("tenant_id") not in (None, "", principal.tenant_id):
                raise HTTPException(status_code=403, detail="Cross-tenant context denied")
            return SecurityContext(
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                roles=principal.roles,
                departments=principal.departments,
                groups=principal.groups,
                permissions=principal.permissions,
            )
        if requested:
            return SecurityContext.model_validate(requested)
        if rag_engine.config.security.require_security_context:
            raise HTTPException(status_code=401, detail="Security context required")
        return None

    def validate_ingest_path(path: str) -> Path:
        resolved = Path(path).resolve()
        roots = [Path(root).resolve() for root in rag_engine.config.server.allowed_ingest_roots]
        if not roots:
            raise HTTPException(
                status_code=403,
                detail="Path ingestion is disabled; configure server.allowed_ingest_roots",
            )
        if not any(resolved == root or root in resolved.parents for root in roots):
            raise HTTPException(status_code=403, detail="Path is outside allowed ingest roots")
        return resolved

    app = FastAPI(
        title="RagZen API",
        version=__version__,
        description="Enterprise-grade, local-first RAG server for Python",
    )

    # CORS configuration (no wildcards in production)
    cors_origins = rag_engine.config.server.cors_origins or ["http://localhost:3000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health/live")
    async def health_live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def health_ready() -> dict[str, Any]:
        health = rag_engine.health()
        if not health.healthy:
            raise HTTPException(status_code=503, detail="Engine unhealthy")
        return health.model_dump(mode="json")

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        stats = rag_engine.stats()
        gauges = (
            "# TYPE ragzen_document_count gauge\n"
            f"ragzen_document_count {stats['document_count']}\n"
            "# TYPE ragzen_indexed_chunk_count gauge\n"
            f"ragzen_indexed_chunk_count {stats['indexed_chunk_count']}\n"
        )
        return gauges + global_metrics.render_prometheus()

    @app.get("/v1/documents")
    async def list_documents(
        tenant_id: str = "",
        limit: int = 100,
        offset: int = 0,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        context = resolve_context({"tenant_id": tenant_id} if tenant_id else {}, authorization)
        documents = rag_engine.list_documents(
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
            security_context=context,
        )
        return {"documents": [document.model_dump(mode="json") for document in documents]}

    @app.get("/v1/documents/{document_id}")
    async def get_document(
        document_id: str,
        tenant_id: str = "",
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        context = resolve_context({"tenant_id": tenant_id} if tenant_id else {}, authorization)
        document = rag_engine.get_document(document_id, security_context=context)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return document.model_dump(mode="json")

    @app.get("/v1/documents/{document_id}/versions")
    async def list_document_versions(
        document_id: str,
        tenant_id: str = "",
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        context = resolve_context({"tenant_id": tenant_id} if tenant_id else {}, authorization)
        versions = rag_engine.list_versions(document_id, security_context=context)
        if not versions:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"versions": [version.model_dump(mode="json") for version in versions]}

    @app.post("/v1/documents")
    async def ingest_document(
        req: IngestRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        try:
            context = resolve_context(req.security_context, authorization)
            path = validate_ingest_path(req.path)
            metadata = dict(req.metadata)
            if context:
                metadata["tenant_id"] = context.tenant_id
            job = await rag_engine.aadd(
                path,
                metadata=metadata,
                security_context=context,
                idempotency_key=req.idempotency_key,
            )
            return job.model_dump(mode="json")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/v1/documents/text")
    async def ingest_text(
        req: IngestTextRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        try:
            context = resolve_context(req.security_context, authorization)
            metadata = dict(req.metadata)
            if context:
                requested_tenant = metadata.get("tenant_id", context.tenant_id)
                if requested_tenant != context.tenant_id:
                    raise TenantIsolationError("Cross-tenant ingestion denied")
                metadata["tenant_id"] = context.tenant_id
            doc = rag_engine.add_text(
                req.text,
                metadata=metadata,
                security_context=context,
                idempotency_key=req.idempotency_key,
            )
            return doc.model_dump(mode="json")
        except HTTPException:
            raise
        except TenantIsolationError as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.delete("/v1/documents/{document_id}")
    async def delete_document(
        document_id: str,
        tenant_id: str = "",
        authorization: str | None = Header(default=None),
    ) -> dict[str, bool]:
        requested = {"tenant_id": tenant_id} if tenant_id else {}
        context = resolve_context(requested, authorization)
        effective_tenant = context.tenant_id if context else tenant_id
        deleted = rag_engine.delete(
            document_id,
            tenant_id=effective_tenant,
            security_context=context,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": True}

    @app.post("/v1/search")
    async def search_documents(
        req: SearchApiRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        try:
            context = resolve_context(req.security_context, authorization)
            results = await rag_engine.asearch(
                req.query,
                top_k=req.top_k,
                filters=req.filters,
                security_context=context,
            )
            return {"results": [r.model_dump(mode="json") for r in results]}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/v1/query")
    async def query_rag(
        req: QueryApiRequest, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        try:
            context = resolve_context(req.security_context, authorization)
            resp = await rag_engine.aask(
                req.query,
                filters=req.filters,
                security_context=context,
            )
            return resp.model_dump(mode="json")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/v1/query/stream")
    async def query_stream(
        req: QueryApiRequest, authorization: str | None = Header(default=None)
    ) -> StreamingResponse:
        context = resolve_context(req.security_context, authorization)

        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                async for chunk in rag_engine.stream(
                    req.query,
                    filters=req.filters,
                    security_context=context,
                ):
                    data = json.dumps({"token": chunk})
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err_data = json.dumps({"error": str(e)})
                yield f"data: {err_data}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app
