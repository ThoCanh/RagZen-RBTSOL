"""RagZen — Enterprise-grade, local-first RAG for Python.

Public API:
    - RagZen: Main entry point
    - SecurityContext: Permission-aware queries
    - RagResponse: Query results
    - SearchResult: Search results
    - Document: Document model
    - Chunk: Chunk model
"""

from __future__ import annotations

from ragzen.engine import RagZen
from ragzen.evaluation import RetrievalEvaluation, evaluate_retrieval
from ragzen.exceptions import (
    ConfigurationError,
    MissingOptionalDependencyError,
    PermissionDeniedError,
    RagZenError,
    SecurityContextRequiredError,
    SecurityError,
    TenantIsolationError,
)
from ragzen.models import (
    AccessControl,
    AuditEvent,
    Chunk,
    Citation,
    Document,
    DocumentVersion,
    HealthStatus,
    IngestionJob,
    QueryMetrics,
    QueryRequest,
    RagResponse,
    SearchResult,
    SecurityContext,
)

__version__ = "0.2.0"

__all__ = [
    "AccessControl",
    "AuditEvent",
    "Chunk",
    "Citation",
    "ConfigurationError",
    "Document",
    "DocumentVersion",
    "HealthStatus",
    "IngestionJob",
    "MissingOptionalDependencyError",
    "PermissionDeniedError",
    "QueryMetrics",
    "QueryRequest",
    "RagResponse",
    "RagZen",
    "RagZenError",
    "RetrievalEvaluation",
    "SearchResult",
    "SecurityContext",
    "SecurityContextRequiredError",
    "SecurityError",
    "TenantIsolationError",
    "__version__",
    "evaluate_retrieval",
]
