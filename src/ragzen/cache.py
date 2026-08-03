"""Security-aware cache backends used by RagZen retrieval."""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict

from ragzen.exceptions import MissingOptionalDependencyError
from ragzen.models import SearchResult


class MemorySearchCache:
    def __init__(self, *, ttl_seconds: int, max_size: int) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._items: OrderedDict[str, tuple[float, list[SearchResult]]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> list[SearchResult] | None:
        if self._ttl <= 0 or self._max_size <= 0:
            return None
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at < time.monotonic():
                self._items.pop(key, None)
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: list[SearchResult]) -> None:
        if self._ttl <= 0 or self._max_size <= 0:
            return
        with self._lock:
            self._items[key] = (time.monotonic() + self._ttl, value)
            self._items.move_to_end(key)
            while len(self._items) > self._max_size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def close(self) -> None:
        self.clear()


class RedisSearchCache:
    def __init__(self, url: str, *, ttl_seconds: int) -> None:
        try:
            import redis
        except ImportError as exc:
            raise MissingOptionalDependencyError("redis", "redis", "distributed cache") from exc
        self._client = redis.Redis.from_url(url)
        self._ttl = ttl_seconds
        self._prefix = "ragzen:search:"

    def get(self, key: str) -> list[SearchResult] | None:
        value = self._client.get(self._prefix + key)
        if value is None:
            return None
        raw = json.loads(value)
        return [SearchResult.model_validate(item) for item in raw]

    def set(self, key: str, value: list[SearchResult]) -> None:
        payload = json.dumps([item.model_dump(mode="json") for item in value])
        self._client.setex(self._prefix + key, self._ttl, payload)

    def clear(self) -> None:
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=self._prefix + "*")
            if keys:
                self._client.delete(*keys)
            if int(cursor) == 0:
                break

    def close(self) -> None:
        self._client.close()
