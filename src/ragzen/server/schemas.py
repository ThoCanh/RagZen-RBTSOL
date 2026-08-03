"""Server request and response schemas for RagZen REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = ""


class IngestTextRequest(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = ""


class QueryApiRequest(BaseModel):
    query: str
    filters: dict[str, Any] = Field(default_factory=dict)
    security_context: dict[str, Any] = Field(default_factory=dict)
    top_k: int = 10
    stream: bool = False


class SearchApiRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: dict[str, Any] = Field(default_factory=dict)
    security_context: dict[str, Any] = Field(default_factory=dict)
