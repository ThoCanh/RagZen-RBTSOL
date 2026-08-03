# RagZen Architecture

RagZen separates orchestration from provider contracts so applications can use the
zero-config local stack or replace individual components.

```text
Python API / CLI / FastAPI
          |
security context + immutable mandatory filters
          |
ingestion ---------------- retrieval ---------------- generation
  |                           |                           |
loader -> chunker          dense / BM25 / graph       LLM provider
  |                           |                           |
document registry          fusion -> reranker         citations
  |                           |
SQLite metadata       SQLite vectors or Qdrant
```

## Storage ownership

- `DocumentRegistry` owns document state, content, versions, idempotency metadata,
  retention metadata, and lifecycle status.
- Vector stores own embeddings plus retrieval payloads containing source provenance
  and access-control fields.
- BM25 and the graph index are durable local indexes and can be rebuilt from indexed
  payloads when a custom deployment requires it.
- Local backup bundles contain the document database, vector database, BM25 index,
  graph index, and a versioned manifest.

## Security boundary

The application authenticates a caller and constructs `SecurityContext`. User filters
cannot replace tenant or ACL fields. Every retrieval backend receives the immutable
tenant filter; built-in backends also evaluate department, role, group, permission,
owner and configured ABAC constraints before returning results.

Server request bodies are not an identity source when API principals are configured.
Production server mode requires a configured principal and restricts path ingestion to
explicit filesystem roots.

## Consistency

Ingestion validates one embedding per chunk. If any vector, sparse, or graph write
fails, RagZen removes partial index writes and marks the document failed. Updates keep
the document ID, archive the prior version, and only expose the new indexed version.

The built-in local stores use locks and SQLite WAL for safe concurrent access inside a
process. Large-scale multi-process workloads should use Qdrant and an external job
system through the provider interfaces.

## Extension model

Core component contracts use Python protocols and constructor injection. Installed
packages may register embedding, vector-store, or LLM providers through the
`ragzen.plugins` entry-point group.
