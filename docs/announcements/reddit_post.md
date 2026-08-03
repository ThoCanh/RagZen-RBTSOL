# Post Title: [Project] RagZen – Open Source Local-First Multi-Tenant RAG Framework for Python (336 docs/sec, SQLite WAL + BM25 + Vector RRF)

Hey r/LocalLLaMA & r/Python!

We just open-sourced **RagZen**, an enterprise-grade local-first Python RAG framework built to address multi-tenant security isolation, local privacy, and high throughput without SaaS dependencies.

### 🌟 Key Highlights:
- **Local-First & Multi-Tenant**: Tenant isolation enforced at the database (SQLite WAL) and storage level with fail-closed RBAC/ABAC policies.
- **Hybrid Retrieval**: BM25 (Unicode & Vietnamese support) + Cosine Vector Similarity fused via Reciprocal Rank Fusion (RRF).
- **Fault-Tolerant Resilience**: Built-in Circuit Breakers and Fallback LLM provider chains.
- **REST & SSE Streaming**: Built-in FastAPI server with SSE streaming endpoints (`/v1/query/stream`) and Prometheus latency metrics (`/metrics`).
- **Zero Vulnerabilities**: 0 Bandit issues across 4.8k LOC, 173 passing unit/integration tests (85.04% branch coverage).

### 📦 Installation & Code Example:
`pip install ragzen`

```python
from ragzen import RagZen, SecurityContext

rag = RagZen.local(storage_path="./ragzen_db")
rag.add_text("Refund window is 30 days.", metadata={"tenant_id": "company-a"})

ctx = SecurityContext(tenant_id="company-a", user_id="user_101")
res = rag.ask("What is the refund window?", security_context=ctx)
print(res.answer)
```

GitHub: https://github.com/ThoCanh/RagZen-RBTSOL
PyPI: https://pypi.org/project/ragzen/

We'd love to get feedback from the community!
