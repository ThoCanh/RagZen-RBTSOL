# Post template: RagZen 0.2, local-first hybrid RAG for Python

We are preparing RagZen 0.2, an Apache-2.0 Python framework for building local-first,
multi-tenant RAG applications.

The zero-config path persists documents, vectors, and BM25 locally. Deployments can
switch to Qdrant, sentence-transformer embeddings, OpenAI-compatible APIs or Ollama.
RagZen also includes graph-assisted retrieval, ACL/ABAC filters, document versions,
backup bundles, streaming FastAPI endpoints, Prometheus metrics, and retrieval metrics.

```bash
pip install ragzen
```

The release is alpha: users should pin versions and test security and retrieval policies
with their own evaluation set. Feedback and provider contributions are welcome.
