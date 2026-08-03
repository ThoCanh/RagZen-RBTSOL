# Changelog

All notable changes to RagZen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-03

### Added
- Restart-safe SQLite vector storage and persistent BM25/graph indexes.
- Qdrant provider, graph-assisted retrieval, weighted fusion and optional reranking.
- PDF, DOCX, XLSX, HTML and JSON loaders.
- Redis search cache, native provider streaming and Prometheus text metrics.
- Full local backup bundles, document version reads and context-manager lifecycle.
- Authenticated server principals and filesystem ingest allowlists.

### Fixed
- Mandatory ACL filters can no longer be overwritten by user filters.
- Tenant-scoped clear no longer removes other tenants' indexes.
- Updates preserve document IDs and content now persists in the registry.
- Configuration provider selections are now honored or fail fast.
- Packaging extras and integration examples now match the public API.
- Document and test dependency floors now exclude versions with known vulnerabilities.

## [0.1.0] - 2026-08-03

### Added
- Enterprise production-grade core domain models (Document, Chunk, SecurityContext, RagResponse).
- Multi-tenant isolation with fail-closed RBAC and ABAC.
- SQLite document registry with WAL mode and transactions.
- Recursive and fixed-size chunking supporting Vietnamese text.
- Dense, BM25 sparse, and Reciprocal Rank Fusion hybrid retrieval.
- OpenAI-compatible LLM provider with fallback chain and circuit breaker.
- Prompt injection screening and secret redaction.
- FastAPI Server mode with SSE streaming support.
- Full CLI interface (`ragzen`).
