"""Prometheus metrics collector for RagZen."""

from __future__ import annotations

from typing import Any


class MetricsCollector:
    """Collects counters and latency histograms for RagZen."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._latencies: dict[str, list[float]] = {}

    def increment(
        self, metric_name: str, value: int = 1, tags: dict[str, str] | None = None
    ) -> None:
        key = f"{metric_name}:{tags}" if tags else metric_name
        self._counters[key] = self._counters.get(key, 0) + value

    def observe_latency(self, metric_name: str, duration_seconds: float) -> None:
        if metric_name not in self._latencies:
            self._latencies[metric_name] = []
        self._latencies[metric_name].append(duration_seconds)

    def get_stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {"counters": self._counters, "latencies": {}}
        for name, values in self._latencies.items():
            if values:
                sorted_v = sorted(values)
                n = len(sorted_v)
                result["latencies"][name] = {
                    "count": n,
                    "avg_ms": (sum(sorted_v) / n) * 1000,
                    "p50_ms": sorted_v[int(n * 0.50)] * 1000,
                    "p95_ms": sorted_v[min(int(n * 0.95), n - 1)] * 1000,
                    "p99_ms": sorted_v[min(int(n * 0.99), n - 1)] * 1000,
                }
        return result


global_metrics = MetricsCollector()
