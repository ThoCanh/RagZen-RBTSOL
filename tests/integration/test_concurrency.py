"""Integration tests for concurrent multi-tenant queries."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

from ragzen import RagZen, SecurityContext


def test_concurrent_queries(tmp_path: Path) -> None:
    rag = RagZen.local(storage_path=str(tmp_path))

    # Seed documents for 3 tenants
    for i in range(1, 4):
        tenant = f"tenant-{i}"
        rag.add_text(
            f"Báo cáo vận hành sản xuất của đơn vị {tenant}.",
            metadata={"tenant_id": tenant, "department": "ops"},
        )

    def worker(tenant_idx: int) -> bool:
        tenant_id = f"tenant-{(tenant_idx % 3) + 1}"
        ctx = SecurityContext(tenant_id=tenant_id, user_id=f"u-{tenant_idx}")

        results = rag.search("sản xuất", security_context=ctx)
        resp = rag.ask("Báo cáo vận hành thế nào?", security_context=ctx)

        # Confirm 100% tenant isolation under concurrency
        return all(r.metadata["tenant_id"] == tenant_id for r in results) and bool(resp.answer)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(30)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert all(results)
    rag.close()
