<p align="center">
  <img src="https://raw.githubusercontent.com/ThoCanh/RagZen-RBTSOL/main/assets/ragzen.png" alt="RagZen" width="560" />
</p>

# RagZen

RagZen is a local-first, permission-aware RAG framework for Python. It provides
durable ingestion, hybrid or graph-assisted retrieval, pluggable providers,
grounded generation, a CLI, and an optional FastAPI server.

> RagZen 0.2 is an alpha release. Its public API is usable, but production users
> should pin the exact version and validate their own retrieval and security policies.

## Features

- Persistent local mode: SQLite document registry, SQLite vectors, and durable BM25.
- Retrieval modes: `dense`, `sparse`, `hybrid`, `graph`, and `hybrid_graph`.
- Vector backends: zero-config SQLite, in-memory development backend, and Qdrant.
- RRF or weighted fusion and optional cross-encoder reranking.
- TXT, Markdown, CSV, JSON, HTML, PDF, DOCX, and XLSX ingestion.
- Tenant, department, role, group, permission, owner, and declared ABAC filtering.
- Idempotent ingestion, content deduplication, document versions, scoped deletion,
  complete local backup bundles, and restart-safe indexes.
- OpenAI-compatible providers including Ollama, plus a dependency-free extractive mode.
- Native async APIs, provider token streaming, Prometheus text metrics, health probes,
  API-key principals, audit events, Redis search cache, and plugin entry-point discovery.

## Install

```bash
pip install ragzen
```

Optional capabilities:

```bash
pip install "ragzen[local]"       # sentence-transformers and cross-encoder reranking
pip install "ragzen[documents]"   # PDF, DOCX and XLSX loaders
pip install "ragzen[qdrant]"      # Qdrant vector backend
pip install "ragzen[redis]"       # distributed search cache
pip install "ragzen[server]"      # FastAPI server
pip install "ragzen[all]"
```

## Zero-config quickstart

```python
from ragzen import RagZen, SecurityContext

with RagZen.local("./data/ragzen") as rag:
    document = rag.add_text(
        "The refund period is 30 days.",
        metadata={"tenant_id": "acme", "department": "support"},
    )

    context = SecurityContext(
        tenant_id="acme",
        user_id="user-1",
        departments=["support"],
    )

    results = rag.search("refund period", security_context=context)
    response = rag.ask("How long is the refund period?", security_context=context)
    print(document.document_id, results[0].content, response.answer)
```

Local mode uses durable SQLite vectors and a dependency-free extractive generator.
The same search results remain available after the process restarts.

## Semantic embeddings and Ollama

```yaml
# ragzen.yaml
embedding:
  provider: sentence_transformers
  model: sentence-transformers/all-MiniLM-L6-v2

llm:
  provider: ollama
  base_url: http://localhost:11434/v1
  model: llama3.2
  timeout_seconds: 60
```

```python
from ragzen import RagZen

rag = RagZen.from_config("ragzen.yaml")
```

## Qdrant and graph-assisted retrieval

```yaml
vector_store:
  provider: qdrant
  url: http://localhost:6333
  collection: company_documents

retrieval:
  mode: hybrid_graph
  fusion: rrf

graph:
  enabled: true
  path: .ragzen/graph.json
  max_hops: 2
```

The built-in graph index is a deterministic entity co-occurrence graph with chunk
provenance. Applications needing ontology extraction or a remote graph database can
inject a custom graph index or retriever.

## Server security

Server principals are configured on the server, not supplied by request bodies:

```yaml
environment: production
security:
  require_security_context: true
  fail_closed: true
  abac_keys: [region, clearance]

server:
  allowed_ingest_roots: [/srv/ragzen/imports]
  principals:
    - api_key: ${RAGZEN_API_KEY}
      tenant_id: acme
      user_id: service-account
      roles: [reader]
      departments: [support]
```

Start the server:

```bash
ragzen --config ragzen.yaml serve --host 0.0.0.0 --port 8000
```

Use `Authorization: Bearer <api-key>`. Production server mode refuses to start
without a configured principal. Filesystem ingestion is disabled unless
`allowed_ingest_roots` is configured.

The Docker Compose deployment uses the production config and Qdrant. Set
`RAGZEN_API_KEY` before running `docker compose`; startup fails if it is absent.

## CLI

```bash
ragzen init --path .ragzen
ragzen ingest ./documents --tenant acme
ragzen search "refund policy" --tenant acme
ragzen query "Summarize the refund policy" --tenant acme
ragzen stats
ragzen doctor
ragzen backup ./backups/ragzen
ragzen restore ./backups/ragzen.zip
```

## Provider plugins

Third-party packages can expose a class through the `ragzen.plugins` Python entry-point
group. Set `plugin_capability` to `embedding`, `vector_store`, or `llm`, and
`plugin_name` to the corresponding config provider name. A plugin may implement
`from_config(config)` or a constructor accepting `config=`.

## Quality gates

The repository CI runs Ruff, Mypy, Bandit, dependency auditing, the full pytest suite
with branch coverage, package build, Twine metadata validation, and a Python 3.11-3.13
matrix on Linux and Windows.
The dependency-free `evaluate_retrieval` helper provides Recall@K, reciprocal rank and
nDCG@K for application-specific evaluation sets.

See [documentation](docs/index.md), [architecture](ARCHITECTURE.md),
[security model](THREAT_MODEL.md), and [contributing guide](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
