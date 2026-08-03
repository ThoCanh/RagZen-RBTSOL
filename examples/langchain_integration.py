"""LangChain integration example for RagZen.

Shows how to wrap RagZen as a custom Retriever inside LangChain pipelines.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ragzen import RagZen, SecurityContext


class RagZenRetriever:
    """LangChain-compatible retriever adapter for RagZen."""

    def __init__(self, rag_engine: RagZen, security_context: SecurityContext | None = None) -> None:
        self.rag = rag_engine
        self.security_context = security_context

    def get_relevant_documents(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
        """Retrieve relevant documents for LangChain chain ingestion."""
        results = self.rag.search(
            query=query,
            top_k=top_k,
            security_context=self.security_context,
        )
        return [
            {
                "page_content": res.content,
                "metadata": res.metadata,
                "score": str(res.score),
            }
            for res in results
        ]


def main() -> None:
    print("=== RagZen LangChain Integration Demo ===")
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = Path(tmp_dir) / ".ragzen"
        rag = RagZen.local(storage_path=str(storage_path))

        # Ingest documents for tenant "acme_corp"
        rag.add_text(
            "LangChain is a framework for developing applications powered by language models.",
            metadata={"tenant_id": "acme_corp", "source": "langchain_doc.txt"},
        )
        rag.add_text(
            "RagZen provides local-first multi-tenant RAG security isolation for Python.",
            metadata={"tenant_id": "acme_corp", "source": "ragzen_doc.txt"},
        )

        # Create Security Context
        sec_ctx = SecurityContext(tenant_id="acme_corp", user_id="dev_01")

        # Instantiate LangChain Retriever Adapter
        retriever = RagZenRetriever(rag_engine=rag, security_context=sec_ctx)
        docs = retriever.get_relevant_documents("What is RagZen?")

        print(f"Retrieved {len(docs)} documents for LangChain:")
        for idx, doc in enumerate(docs, 1):
            print(f"[{idx}] {doc['page_content']} (Source: {doc['metadata'].get('source')})")

        rag.close()


if __name__ == "__main__":
    main()
