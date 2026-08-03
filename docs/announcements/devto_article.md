# Introducing RagZen 0.2: a local-first RAG framework for Python

RagZen is an Apache-2.0 Python framework for permission-aware retrieval-augmented
generation. Version 0.2 provides a durable zero-config local stack and lets production
deployments replace individual providers.

Highlights:

- dense, BM25, hybrid, graph, and hybrid-graph retrieval;
- persistent SQLite vectors or Qdrant, with immutable tenant and ACL filters;
- document deduplication, version history, rollback-safe ingestion, and backup bundles;
- PDF, DOCX, XLSX, HTML, JSON, Markdown, CSV, and text loaders;
- OpenAI-compatible LLMs, Ollama, or dependency-free extractive generation;
- authenticated FastAPI deployment, real streaming, Prometheus metrics, and health probes;
- typed public APIs and retrieval evaluation helpers.

```bash
pip install ragzen
```

```python
from ragzen import RagZen, SecurityContext

with RagZen.local("./ragzen-data") as rag:
    rag.add_text("Refunds are available for 30 days.", metadata={"tenant_id": "acme"})
    context = SecurityContext(tenant_id="acme", user_id="demo")
    print(rag.ask("What is the refund period?", security_context=context).answer)
```

RagZen 0.2 is an alpha release. Pin the exact version and validate retrieval quality and
authorization policies against your own data before production use.
