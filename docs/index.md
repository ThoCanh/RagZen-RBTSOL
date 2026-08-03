# RagZen Documentation

Welcome to the official documentation for **RagZen** — an enterprise-grade, local-first, multi-tenant RAG (Retrieval-Augmented Generation) framework for Python.

## Core Highlights

- **Security by Default**: Multi-tenant data isolation, RBAC/ABAC fail-closed authorization, prompt injection detection, and PII redactor.
- **Local-First Speed**: Built-in SQLite WAL storage, BM25 Unicode/Vietnamese search, and in-memory cosine vector store with Reciprocal Rank Fusion (RRF).
- **Fault-Tolerant Resilience**: Circuit Breaker state machine (`CLOSED`, `OPEN`, `HALF_OPEN`) and Fallback LLM Provider chain.
- **Production Web Server**: FastAPI REST server, SSE real-time streaming (`/v1/query/stream`), Prometheus metrics (`/metrics`), and health probes.

## Installation

```bash
pip install ragzen
```

For the server and local semantic embedding models:

```bash
pip install "ragzen[server,local]"
```

## Quick Example

```python
from ragzen import RagZen, SecurityContext

rag = RagZen.local(storage_path="./data/ragzen_db")

# Ingest document
rag.add_text("Product refund period is 30 days.", metadata={"tenant_id": "company-a"})

# Query with SecurityContext
ctx = SecurityContext(tenant_id="company-a", user_id="user_101")
response = rag.ask("How many days for refund?", security_context=ctx)

print("Answer:", response.answer)
rag.close()
```
