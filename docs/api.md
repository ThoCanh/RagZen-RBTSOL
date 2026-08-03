# Python API

The stable entry point is `ragzen.RagZen`. Core operations include `add`, `add_text`,
`add_documents`, `search`, `ask`, `stream`, `get_document`, `list_documents`,
`list_versions`, `update`, `delete`, `clear`, `backup`, `restore`, `health`, and `stats`.

Async counterparts are available for ingestion, search and generation. `RagZen` is a
context manager and should be closed to flush indexes and provider resources.

`ragzen.evaluate_retrieval` computes Recall@K, reciprocal rank and nDCG@K for a
ranked list. `ragzen.evaluation.citation_precision` measures whether generated
citations refer to the supplied source set.
