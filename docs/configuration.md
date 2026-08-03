# Configuration

Load a YAML file with `RagZen.from_config("ragzen.yaml")`. Exact `${VARIABLE}` values
are resolved from the environment; missing variables fail startup.

Important sections are `storage`, `embedding`, `vector_store`, `sparse_index`,
`retrieval`, `graph`, `reranker`, `llm`, `cache`, `chunking`, `security`, and `server`.
Provider names are validated and unsupported values fail fast.

For production, enable `security.require_security_context` and `security.fail_closed`,
declare any document ABAC keys in `security.abac_keys`, configure server principals,
and restrict `server.allowed_ingest_roots`.
