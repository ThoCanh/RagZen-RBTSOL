"""Regression coverage for persistence, security, graph and provider wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragzen import RagZen, SecurityContext
from ragzen.cache import MemorySearchCache
from ragzen.config import (
    ApiPrincipalConfig,
    GraphConfig,
    RagZenConfig,
    RetrievalConfig,
    SecurityConfig,
    ServerConfig,
)
from ragzen.graph import KnowledgeGraphIndex
from ragzen.loaders.documents import UniversalDocumentLoader
from ragzen.models import SearchResult
from ragzen.server.app import create_app
from ragzen.vectorstores.qdrant import QdrantVectorStore
from ragzen.vectorstores.sqlite import SQLiteVectorStore


def test_restart_keeps_searchable_indexes_and_versions(tmp_path: Path) -> None:
    context = SecurityContext(tenant_id="tenant-a")
    first = RagZen.local(str(tmp_path))
    document = first.add_text(
        "Durable retrieval survives a restart.", metadata={"tenant_id": "tenant-a"}
    )
    assert first.search("durable retrieval", security_context=context)
    first.close()

    second = RagZen.local(str(tmp_path))
    assert second.search("durable retrieval", security_context=context)
    assert second.update(
        document.document_id,
        "Durable retrieval now has version two.",
        security_context=context,
    )
    current = second.get_document(document.document_id, security_context=context)
    assert current is not None
    assert current.document_id == document.document_id
    assert current.version == 2
    assert [
        version.version
        for version in second.list_versions(document.document_id, security_context=context)
    ] == [1, 2]
    second.close()


def test_acl_filter_cannot_be_overridden_and_clear_is_scoped(tmp_path: Path) -> None:
    rag = RagZen.local(str(tmp_path))
    rag.add_text(
        "Human resources secret",
        metadata={"tenant_id": "tenant-a", "department": "hr"},
    )
    finance = SecurityContext(tenant_id="tenant-a", departments=["finance"])
    assert not rag.search("resources secret", security_context=finance)
    assert not rag.search(
        "resources secret",
        security_context=finance,
        filters={"departments": ["hr"]},
    )

    rag.add_text("Tenant B remains", metadata={"tenant_id": "tenant-b"})
    tenant_b = SecurityContext(tenant_id="tenant-b")
    rag.clear(tenant_id="tenant-a")
    assert rag.search("Tenant B", security_context=tenant_b)
    rag.close()


def test_graph_assisted_retrieval_persists(tmp_path: Path) -> None:
    config = RagZenConfig.local_default(str(tmp_path)).model_copy(
        update={
            "graph": GraphConfig(enabled=True, path=str(tmp_path / "graph.json")),
            "retrieval": RetrievalConfig(mode="hybrid_graph", final_top_k=5),
        }
    )
    rag = RagZen(config=config)
    rag.add_text(
        "Alice manages Project Orion. Project Orion uses Qdrant.",
        metadata={"tenant_id": "tenant-a"},
    )
    context = SecurityContext(tenant_id="tenant-a")
    results = rag.search("Alice Qdrant", security_context=context)
    assert results
    rag.close()

    reopened = RagZen(config=config)
    assert reopened.search("Alice Qdrant", security_context=context)
    reopened.close()


def test_local_backup_bundle_restores_all_indexes(tmp_path: Path) -> None:
    original_path = tmp_path / "original"
    rag = RagZen.local(str(original_path))
    context = SecurityContext(tenant_id="tenant-a")
    rag.add_text("Bundle backup content", metadata={"tenant_id": "tenant-a"})
    bundle = rag.backup(tmp_path / "backup")
    rag.clear(tenant_id="tenant-a")
    assert not rag.search("backup content", security_context=context)
    assert rag.restore(bundle)
    assert rag.search("backup content", security_context=context)
    rag.close()


def test_sqlite_vector_dimensions_and_filtering(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "vectors.db")
    store.upsert("docs", "c1", [1.0, 0.0], {"tenant_id": "a", "content": "A"})
    store.upsert("docs", "c2", [0.0, 1.0], {"tenant_id": "b", "content": "B"})
    results = store.search("docs", [1.0, 0.0], filters={"tenant_id": "a"})
    assert [result[0] for result in results] == ["c1"]
    with pytest.raises(Exception, match="expects 2 dimensions"):
        store.search("docs", [1.0])
    store.close()


def test_graph_index_and_memory_cache(tmp_path: Path) -> None:
    graph = KnowledgeGraphIndex(tmp_path / "graph.json")
    graph.add("c1", "Alice builds Orion", {"tenant_id": "a", "document_id": "d1"})
    assert graph.search("Alice", filters={"tenant_id": "a"})
    assert graph.remove_by_document_id("d1") == 1
    graph.close()

    cache = MemorySearchCache(ttl_seconds=60, max_size=1)
    value = [SearchResult(chunk_id="c", document_id="d", content="x", score=1.0)]
    cache.set("one", value)
    assert cache.get("one") == value
    cache.set("two", value)
    assert cache.get("one") is None
    cache.close()


def test_json_and_html_loaders(tmp_path: Path) -> None:
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps({"policy": "30 days"}), encoding="utf-8")
    html_path = tmp_path / "page.html"
    html_path.write_text("<h1>Policy</h1><script>ignore()</script><p>30 days</p>", encoding="utf-8")
    loader = UniversalDocumentLoader()
    assert "30 days" in loader.load(json_path)[0].content
    html = loader.load(html_path)[0].content
    assert "30 days" in html
    assert "ignore" not in html


def test_qdrant_local_adapter_enforces_acl(tmp_path: Path) -> None:
    store = QdrantVectorStore(path=str(tmp_path / "qdrant"))
    chunk_id = "11111111-1111-1111-1111-111111111111"
    store.upsert(
        "docs",
        chunk_id,
        [1.0, 0.0],
        {"tenant_id": "a", "content": "secret", "departments": ["hr"]},
    )
    base_filters = {
        "tenant_id": "a",
        "_security_user_id": "user",
        "_security_roles": [],
        "_security_groups": [],
        "_security_permissions": [],
        "_security_attributes": {},
        "_security_attribute_keys": [],
    }
    denied = store.search(
        "docs",
        [1.0, 0.0],
        filters={**base_filters, "_security_departments": ["finance"]},
    )
    allowed = store.search(
        "docs",
        [1.0, 0.0],
        filters={**base_filters, "_security_departments": ["hr"]},
    )
    assert denied == []
    assert allowed[0][0] == chunk_id
    store.close()


def test_production_server_uses_server_side_principal(tmp_path: Path) -> None:
    local = RagZenConfig.local_default(str(tmp_path))
    config = local.model_copy(
        update={
            "environment": "production",
            "security": SecurityConfig(require_security_context=True, fail_closed=True),
            "server": ServerConfig(
                principals=[
                    ApiPrincipalConfig(
                        api_key="secret-key",
                        tenant_id="tenant-a",
                        user_id="service",
                        departments=["hr"],
                    )
                ]
            ),
        }
    )
    engine = RagZen(config=config)
    client = TestClient(create_app(engine))
    headers = {"Authorization": "Bearer secret-key"}
    response = client.post(
        "/v1/documents/text",
        headers=headers,
        json={
            "text": "HR policy",
            "metadata": {"tenant_id": "tenant-a", "department": "hr"},
        },
    )
    assert response.status_code == 200
    denied = client.post(
        "/v1/search",
        headers=headers,
        json={"query": "HR", "security_context": {"tenant_id": "tenant-b"}},
    )
    assert denied.status_code == 403
    allowed = client.post(
        "/v1/search",
        headers=headers,
        json={"query": "HR", "security_context": {"tenant_id": "tenant-a"}},
    )
    assert allowed.status_code == 200
    assert allowed.json()["results"]
    engine.close()
