# Architecture

RagZen orchestrates loaders, chunkers, embedding providers, vector and sparse stores,
an optional graph index, fusion, reranking and generation behind the `RagZen` API.

Local mode persists documents and vectors in SQLite and persists BM25/graph indexes as
atomic JSON files. Qdrant is available for larger vector collections. See the root
`ARCHITECTURE.md` for storage ownership, consistency and extension details.
