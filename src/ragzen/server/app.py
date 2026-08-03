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

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from ragzen.engine import RagZen
from ragzen.exceptions import MissingOptionalDependencyError
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
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
    except ImportError as e:
        raise MissingOptionalDependencyError("fastapi", "server", "RagZen Server Mode") from e

    rag_engine = engine or RagZen.local()

    app = FastAPI(
        title="RagZen API",
        version="0.1.0",
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

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return rag_engine.stats()

    @app.post("/v1/documents")
    async def ingest_document(req: IngestRequest) -> dict[str, Any]:
        try:
            job = await rag_engine.aadd(
                req.path,
                metadata=req.metadata,
                idempotency_key=req.idempotency_key,
            )
            return job.model_dump(mode="json")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/v1/documents/text")
    async def ingest_text(req: IngestTextRequest) -> dict[str, Any]:
        try:
            doc = rag_engine.add_text(
                req.text,
                metadata=req.metadata,
                idempotency_key=req.idempotency_key,
            )
            return doc.model_dump(mode="json")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.delete("/v1/documents/{document_id}")
    async def delete_document(document_id: str, tenant_id: str = "") -> dict[str, bool]:
        deleted = rag_engine.delete(document_id, tenant_id=tenant_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"deleted": True}

    @app.post("/v1/search")
    async def search_documents(req: SearchApiRequest) -> dict[str, Any]:
        try:
            results = await rag_engine.asearch(
                req.query,
                top_k=req.top_k,
                filters=req.filters,
                security_context=req.security_context,
            )
            return {"results": [r.model_dump(mode="json") for r in results]}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/v1/query")
    async def query_rag(req: QueryApiRequest) -> dict[str, Any]:
        try:
            resp = await rag_engine.aask(
                req.query,
                filters=req.filters,
                security_context=req.security_context,
            )
            return resp.model_dump(mode="json")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/v1/query/stream")
    async def query_stream(req: QueryApiRequest) -> StreamingResponse:
        async def event_generator() -> AsyncGenerator[str, None]:
            try:
                async for chunk in rag_engine.stream(
                    req.query,
                    filters=req.filters,
                    security_context=req.security_context,
                ):
                    data = json.dumps({"token": chunk})
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err_data = json.dumps({"error": str(e)})
                yield f"data: {err_data}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app
