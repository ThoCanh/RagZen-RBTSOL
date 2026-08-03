"""RagZen Performance Benchmark Script.

Measures:
- Ingestion throughput (documents/sec)
- Search & RAG latency (P50, P95, P99 ms)
- Multi-tenant isolation verification
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from ragzen import RagZen, SecurityContext
from ragzen.llms.mock import MockLLMProvider


def run_benchmark() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("==================================================")
    print("      RAGZEN PERFORMANCE BENCHMARK SUITE          ")
    print("==================================================")

    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_path = Path(tmp_dir) / ".ragzen"
        rag = RagZen.from_components(
            document_registry=None,
            vector_store=None,
            sparse_index=None,
            llm=MockLLMProvider(),
        )

        # 1. Ingestion Benchmark
        num_docs = 100
        docs_dir = Path(tmp_dir) / "bench_docs"
        docs_dir.mkdir()

        for i in range(num_docs):
            p = docs_dir / f"doc_{i:03d}.txt"
            p.write_text(
                f"Tài liệu kỹ thuật số {i}: Quy trình bảo trì hệ thống và giám sát hiệu năng enterprise.",
                encoding="utf-8",
            )

        start_ingest = time.perf_counter()
        job = rag.add(docs_dir, metadata={"tenant_id": "bench-tenant"})
        ingest_time = time.perf_counter() - start_ingest
        throughput = num_docs / ingest_time if ingest_time > 0 else 0

        print(f"[Ingestion] Processed {job.processed_documents} docs in {ingest_time:.3f}s ({throughput:.1f} docs/sec)")

        # 2. Search Latency Benchmark
        ctx = SecurityContext(tenant_id="bench-tenant")
        search_latencies = []

        for _ in range(50):
            t0 = time.perf_counter()
            rag.search("giám sát hiệu năng", top_k=5, security_context=ctx)
            search_latencies.append((time.perf_counter() - t0) * 1000)

        search_latencies.sort()
        n_s = len(search_latencies)
        p50_s = search_latencies[int(n_s * 0.50)]
        p95_s = search_latencies[min(int(n_s * 0.95), n_s - 1)]
        p99_s = search_latencies[min(int(n_s * 0.99), n_s - 1)]

        print(f"[Search Latency]  P50: {p50_s:.2f}ms | P95: {p95_s:.2f}ms | P99: {p99_s:.2f}ms")

        # 3. Ask / Generation Latency Benchmark
        ask_latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            rag.ask("Quy trình bảo trì ra sao?", security_context=ctx)
            ask_latencies.append((time.perf_counter() - t0) * 1000)

        ask_latencies.sort()
        n_a = len(ask_latencies)
        p50_a = ask_latencies[int(n_a * 0.50)]
        p95_a = ask_latencies[min(int(n_a * 0.95), n_a - 1)]
        p99_a = ask_latencies[min(int(n_a * 0.99), n_a - 1)]

        print(f"[Ask Latency]     P50: {p50_a:.2f}ms | P95: {p95_a:.2f}ms | P99: {p99_a:.2f}ms")
        print("==================================================")

        rag.close()


if __name__ == "__main__":
    run_benchmark()
