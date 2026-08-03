"""Persistent entity co-occurrence graph for graph-assisted retrieval."""

from __future__ import annotations

import json
import re
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from ragzen.security.filters import metadata_matches_filters


class KnowledgeGraphIndex:
    """A lightweight, deterministic graph index with source provenance.

    Entities are normalized terms and edges represent co-occurrence inside a
    chunk. It is intentionally provider-free and can later be replaced through
    dependency injection by a richer graph database implementation.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        max_hops: int = 2,
        min_entity_length: int = 3,
    ) -> None:
        self._path = Path(path) if path else None
        self._max_hops = max_hops
        self._min_entity_length = min_entity_length
        self._chunks: dict[str, dict[str, Any]] = {}
        self._entities: dict[str, set[str]] = defaultdict(set)
        self._edges: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        if self._path and self._path.exists():
            self._load()

    def _extract_entities(self, text: str) -> list[str]:
        tokens = re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)
        return sorted({token for token in tokens if len(token) >= self._min_entity_length})

    def add(self, chunk_id: str, content: str, metadata: dict[str, Any]) -> None:
        with self._lock:
            if chunk_id in self._chunks:
                self.remove(chunk_id)
            entities = self._extract_entities(content)
            self._chunks[chunk_id] = {
                "content": content,
                "metadata": metadata,
                "entities": entities,
            }
            for entity in entities:
                self._entities[entity].add(chunk_id)
            for index, entity in enumerate(entities):
                for neighbor in entities[index + 1 :]:
                    self._edges[entity].add(neighbor)
                    self._edges[neighbor].add(entity)
            self._persist()

    def remove(self, chunk_id: str) -> bool:
        with self._lock:
            chunk = self._chunks.pop(chunk_id, None)
            if chunk is None:
                return False
            for entity in chunk["entities"]:
                self._entities[entity].discard(chunk_id)
                if not self._entities[entity]:
                    self._entities.pop(entity, None)
            self._rebuild_edges()
            self._persist()
            return True

    def remove_by_document_id(self, document_id: str) -> int:
        with self._lock:
            ids = [
                chunk_id
                for chunk_id, item in self._chunks.items()
                if item["metadata"].get("document_id") == document_id
            ]
            for chunk_id in ids:
                chunk = self._chunks.pop(chunk_id)
                for entity in chunk["entities"]:
                    self._entities[entity].discard(chunk_id)
                    if not self._entities[entity]:
                        self._entities.pop(entity, None)
            self._rebuild_edges()
            self._persist()
            return len(ids)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        seeds = self._extract_entities(query)
        if not seeds:
            return []
        with self._lock:
            queue = deque((entity, 0) for entity in seeds if entity in self._entities)
            distances: dict[str, int] = {entity: 0 for entity, _ in queue}
            while queue:
                entity, distance = queue.popleft()
                if distance >= self._max_hops:
                    continue
                for neighbor in self._edges.get(entity, set()):
                    if neighbor not in distances:
                        distances[neighbor] = distance + 1
                        queue.append((neighbor, distance + 1))

            scores: dict[str, float] = defaultdict(float)
            for entity, distance in distances.items():
                weight = 1.0 / (distance + 1)
                if entity in seeds:
                    weight += 1.0
                for chunk_id in self._entities.get(entity, set()):
                    scores[chunk_id] += weight

            results = []
            for chunk_id, score in scores.items():
                item = self._chunks[chunk_id]
                metadata = dict(item["metadata"])
                if filters and not metadata_matches_filters(metadata, filters):
                    continue
                metadata["content"] = item["content"]
                results.append((chunk_id, score, metadata))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def clear(self, *, filters: dict[str, Any] | None = None) -> None:
        with self._lock:
            if filters:
                ids = [
                    chunk_id
                    for chunk_id, item in self._chunks.items()
                    if metadata_matches_filters(item["metadata"], filters)
                ]
                for chunk_id in ids:
                    self._chunks.pop(chunk_id, None)
                self._rebuild_entities()
            else:
                self._chunks.clear()
                self._entities.clear()
                self._edges.clear()
            self._persist()

    def close(self) -> None:
        with self._lock:
            self._persist()

    def _rebuild_entities(self) -> None:
        self._entities = defaultdict(set)
        for chunk_id, item in self._chunks.items():
            for entity in item["entities"]:
                self._entities[entity].add(chunk_id)
        self._rebuild_edges()

    def _rebuild_edges(self) -> None:
        self._edges = defaultdict(set)
        for item in self._chunks.values():
            entities = item["entities"]
            for index, entity in enumerate(entities):
                for neighbor in entities[index + 1 :]:
                    self._edges[entity].add(neighbor)
                    self._edges[neighbor].add(entity)

    def _persist(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"chunks": self._chunks}
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self._path)

    def _load(self) -> None:
        if self._path is None:
            return
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        self._chunks = payload.get("chunks", {})
        self._rebuild_entities()
