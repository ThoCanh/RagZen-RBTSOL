from __future__ import annotations

from pathlib import Path

import pytest

from ragzen import Document, RagZen, SecurityContext
from ragzen.cache import MemorySearchCache
from ragzen.config import RagZenConfig, RetrievalConfig, VectorStoreConfig
from ragzen.embeddings.mock import MockEmbeddingProvider
from ragzen.graph import KnowledgeGraphIndex
from ragzen.models import DocumentStatus, SearchResult
from ragzen.vectorstores.memory import InMemoryVectorStore
from ragzen.vectorstores.sqlite import SQLiteVectorStore


def test_memory_vector_store_complete_lifecycle() -> None:
    store = InMemoryVectorStore()
    assert not store.collection_exists("docs")
    store.create_collection("docs", dimensions=2)
    assert store.collection_exists("docs")
    store.batch_upsert(
        "docs",
        [
            ("a", [1.0, 0.0], {"tenant_id": "one", "kind": "policy"}),
            ("b", [0.0, 1.0], {"tenant_id": "two", "kind": "note"}),
        ],
    )
    assert store.count("docs") == 2
    assert store.search("docs", [1.0, 0.0], filters={"tenant_id": "one"})[0][0] == "a"
    assert store.search("missing", [1.0, 0.0]) == []
    assert store._cosine_similarity([1.0], [1.0, 2.0]) == 0.0
    assert store._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert store.delete("docs", "missing") is False
    assert store.delete("docs", "a") is True
    assert store.delete_by_filter("docs", {"tenant_id": "two"}) == 1
    store.upsert("docs", "c", [1.0, 1.0], {"tenant_id": "three"})
    store.clear("docs", filters={"tenant_id": "three"})
    assert store.count("docs") == 0
    assert store.health()
    store.clear()
    store.close()


def test_sqlite_vector_store_batch_delete_and_clear(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vectors.db")
    store.create_collection("docs", dimensions=2)
    assert store.collection_exists("docs")
    store.batch_upsert(
        "docs",
        [
            ("a", [1.0, 0.0], {"tenant_id": "one"}),
            ("b", [0.0, 1.0], {"tenant_id": "two"}),
        ],
    )
    assert store.delete("docs", "a")
    assert not store.delete("docs", "missing")
    assert store.delete_by_filter("docs", {"tenant_id": "two"}) == 1
    store.upsert("docs", "c", [1.0, 1.0], {"tenant_id": "three"})
    store.clear("docs", filters={"tenant_id": "three"})
    store.clear()
    assert store.health()
    store.close()


def test_memory_cache_expiry_disabled_and_clear() -> None:
    value = [SearchResult(chunk_id="c", document_id="d", content="x", score=1.0)]
    expiring = MemorySearchCache(ttl_seconds=0, max_size=1)
    expiring.set("key", value)
    assert expiring.get("key") is None
    disabled = MemorySearchCache(ttl_seconds=10, max_size=0)
    disabled.set("key", value)
    assert disabled.get("key") is None
    disabled.clear()


def test_mock_embedding_provider_paths() -> None:
    provider = MockEmbeddingProvider(dimensions=8)
    assert provider.dimensions == 8
    assert provider.model_name
    assert provider.embed([]) == []
    assert len(provider.embed(["hello"])[0]) == 8
    assert len(provider.embed_query("hello")) == 8
    assert provider.health_check()


def test_graph_replace_clear_and_empty_search(tmp_path: Path) -> None:
    graph = KnowledgeGraphIndex(tmp_path / "graph.json", max_hops=1)
    assert graph.search("unknown") == []
    graph.add("c1", "Alice builds Orion", {"tenant_id": "one", "document_id": "d1"})
    graph.add("c1", "Alice leads Orion", {"tenant_id": "one", "document_id": "d1"})
    graph.add("c2", "Bob builds Atlas", {"tenant_id": "two", "document_id": "d2"})
    assert graph.count() == 2
    graph.clear(filters={"tenant_id": "one"})
    assert graph.count() == 1
    assert not graph.remove("missing")
    graph.clear()
    assert graph.count() == 0


class WrongCountEmbedding:
    dimensions = 2
    model_name = "wrong-count"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return []

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0]

    def health_check(self) -> bool:
        return True


def test_ingestion_rolls_back_partial_indexes(tmp_path: Path) -> None:
    config = RagZenConfig.local_default(str(tmp_path))
    rag = RagZen(config=config, embedding=WrongCountEmbedding())
    with pytest.raises(ValueError, match="returned 0 vectors"):
        rag.add_text("rollback content", metadata={"tenant_id": "one"})
    documents = rag.registry.list_by_tenant("one")
    assert documents[0].status == DocumentStatus.FAILED
    assert rag.stats()["vector_store_count"] == 0
    rag.close()


def test_engine_memory_modes_document_api_and_dedup(tmp_path: Path) -> None:
    base = RagZenConfig.local_default(str(tmp_path))
    config = base.model_copy(
        update={
            "vector_store": VectorStoreConfig(provider="memory"),
            "retrieval": RetrievalConfig(mode="sparse"),
        }
    )
    rag = RagZen(config=config)
    context = SecurityContext(tenant_id="one")
    first = rag.add_text("same content", metadata={"tenant_id": "one"}, idempotency_key="key")
    second = rag.add_text(
        "different ignored content",
        metadata={"tenant_id": "one"},
        idempotency_key="key",
    )
    assert first.document_id == second.document_id
    duplicate = rag.add_text("same content", metadata={"tenant_id": "one"})
    assert duplicate.document_id == first.document_id
    assert rag.list_documents(security_context=context)
    assert rag.get_document(first.document_id, security_context=context)
    assert rag.search("same", security_context=context)
    assert not rag.delete("missing", tenant_id="one", security_context=context)
    rag.add_documents(
        [Document(tenant_id="one", content="second document")], security_context=context
    )
    assert rag.registry.count(tenant_id="one") == 2
    rag.clear(security_context=context)
    assert rag.registry.count(tenant_id="one") == 0
    rag.close()
