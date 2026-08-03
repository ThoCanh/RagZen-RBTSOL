# Changelog

All notable changes to RagZen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
