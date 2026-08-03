"""Prometheus metrics collector for RagZen."""

from __future__ import annotations

import threading
from typing import Any


class MetricsCollector:
    """Collects counters and latency histograms for RagZen."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._latencies: dict[str, list[float]] = {}
        self._lock = threading.RLock()
        self._max_samples = 10000

    def increment(
        self, metric_name: str, value: int = 1, tags: dict[str, str] | None = None
    ) -> None:
        key = f"{metric_name}:{tags}" if tags else metric_name
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def observe_latency(self, metric_name: str, duration_seconds: float) -> None:
        with self._lock:
            values = self._latencies.setdefault(metric_name, [])
            values.append(duration_seconds)
            if len(values) > self._max_samples:
                del values[: len(values) - self._max_samples]

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            latency_items = {name: list(values) for name, values in self._latencies.items()}
        result: dict[str, Any] = {"counters": counters, "latencies": {}}
        for name, values in latency_items.items():
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

    def render_prometheus(self) -> str:
        """Render counters and latency summaries in Prometheus text format."""
        stats = self.get_stats()
        lines: list[str] = []
        for name, value in stats["counters"].items():
            metric = str(name).replace(":", "_").replace("-", "_")
            lines.append(f"# TYPE {metric} counter")
            lines.append(f"{metric} {value}")
        for name, values in stats["latencies"].items():
            metric = name.replace("-", "_") + "_milliseconds"
            lines.append(f"# TYPE {metric} summary")
            for quantile in ("p50_ms", "p95_ms", "p99_ms"):
                q = {"p50_ms": "0.5", "p95_ms": "0.95", "p99_ms": "0.99"}[quantile]
                lines.append(f'{metric}{{quantile="{q}"}} {values[quantile]}')
            lines.append(f"{metric}_count {values['count']}")
        return "\n".join(lines) + "\n"


global_metrics = MetricsCollector()
