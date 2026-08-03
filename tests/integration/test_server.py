"""Integration tests for RagZen FastAPI server endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ragzen.engine import RagZen
from ragzen.server.app import create_app


def test_server_endpoints(tmp_path: object) -> None:
    storage_path = str(tmp_path)
    engine = RagZen.local(storage_path=storage_path)
    app = create_app(engine)
    client = TestClient(app)

    # Health live
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "live"}

    # Health ready
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["healthy"] is True

    # Metrics
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "document_count" in resp.json()

    # Ingest text
    resp = client.post(
        "/v1/documents/text",
        json={
            "text": "Báo cáo an toàn lao động năm 2026.",
            "metadata": {"tenant_id": "company-a"},
        },
    )
    assert resp.status_code == 200
    doc_data = resp.json()
    doc_id = doc_data["document_id"]

    # Search
    resp = client.post(
        "/v1/search",
        json={
            "query": "an toàn lao động",
            "security_context": {"tenant_id": "company-a"},
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) > 0

    # Query
    resp = client.post(
        "/v1/query",
        json={
            "query": "Báo cáo an toàn lao động ghi nhận gì?",
            "security_context": {"tenant_id": "company-a"},
        },
    )
    assert resp.status_code == 200
    assert "answer" in resp.json()

    # Delete
    resp = client.delete(f"/v1/documents/{doc_id}?tenant_id=company-a")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}
