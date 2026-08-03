# Introducing RagZen: Open Source Local-First Multi-Tenant RAG Framework for Python

Building Retrieval-Augmented Generation (RAG) applications for enterprise environments requires much more than simple vector search over PDF files. In production, enterprise applications demand **strict multi-tenant security boundaries**, **granular RBAC/ABAC authorization**, **local-first privacy with zero SaaS lock-in**, and **high fault tolerance**.

That is why we built **RagZen** — an enterprise-grade, local-first Python RAG framework available now on PyPI and GitHub.

---

## ⚡ Why RagZen?

1. **Multi-Tenant Security by Default**: Tenant boundaries are enforced directly at the storage layer (`SQLite` + `VectorStore` + `BM25`). Unauthorized cross-tenant queries are blocked fail-closed before any LLM inference occurs.
2. **Local-First Architecture**: Powered by SQLite in WAL (Write-Ahead Logging) mode, BM25 Unicode/Vietnamese lexical search, and high-speed in-memory vector storage with Reciprocal Rank Fusion (RRF).
3. **Resilience & Circuit Breakers**: Built-in `CircuitBreaker` state machine (`CLOSED`, `OPEN`, `HALF_OPEN`) and `FallbackLLMProvider` chain for automatic failover when LLM APIs time out or fail.
4. **Citation Validation**: Built-in verification engine maps output citations (`[Source X]`) directly to validated source metadata.
5. **Built-in FastAPI Server & SSE Streaming**: Includes a production REST server with real-time SSE streaming (`/v1/query/stream`), Prometheus metrics (`/metrics`), and health probes (`/health/live`, `/health/ready`).

---

## 🚀 Quickstart

```bash
pip install ragzen
```

### Python API

```python
from ragzen import RagZen, SecurityContext

# Initialize local RAG instance
rag = RagZen.local(storage_path="./data/ragzen_db")

# Ingest document restricted to tenant "finance_corp"
rag.add_text(
    text="Q4 Net profit increased by 14.2% based on audited accounts.",
    metadata={"tenant_id": "finance_corp", "roles": ["auditor"]}
)

# User Security Context
ctx = SecurityContext(
    tenant_id="finance_corp",
    user_id="user_101",
    roles=["auditor"]
)

# RAG Query with Citation Tracking
response = rag.ask("What is the Q4 net profit increase?", security_context=ctx)
print("Answer:", response.answer)
for c in response.citations:
    print("Source:", c.source_id)

rag.close()
```

---

## 📊 Empirical Benchmarks

- **Ingestion Throughput**: **336.9 documents / second**
- **Search Latency (P99)**: **< 8.0 milliseconds**
- **Test Suite**: **173 / 173 test cases passed (85.04% branch coverage)**
- **Security Audit**: **0 Vulnerabilities** detected by Bandit static analysis across 4,873 LOC.

---

## 🔗 Links

- **GitHub Repository**: [https://github.com/ThoCanh/RagZen-RBTSOL](https://github.com/ThoCanh/RagZen-RBTSOL)
- **PyPI Package**: [https://pypi.org/project/ragzen/](https://pypi.org/project/ragzen/)
- **License**: Apache 2.0 (Open Source)
