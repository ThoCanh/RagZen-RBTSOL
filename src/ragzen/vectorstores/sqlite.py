"""Persistent SQLite vector store for local RagZen deployments."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ragzen.exceptions import EmbeddingDimensionMismatchError, StorageError


class SQLiteVectorStore:
    """Small-to-medium persistent vector store using brute-force cosine search.

    This backend prioritizes zero-configuration durability and correctness. Qdrant
    remains the recommended backend for large collections and ANN search.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vector_collections (
                name TEXT PRIMARY KEY,
                dimensions INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS vectors (
                collection TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                vector TEXT NOT NULL,
                metadata TEXT NOT NULL,
                PRIMARY KEY (collection, chunk_id),
                FOREIGN KEY (collection) REFERENCES vector_collections(name)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_vectors_collection
                ON vectors(collection);
            """
        )
        self._conn.commit()

    def create_collection(self, name: str, *, dimensions: int = 0) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT dimensions FROM vector_collections WHERE name = ?", (name,)
            ).fetchone()
            if row and dimensions and row[0] not in (0, dimensions):
                raise EmbeddingDimensionMismatchError(
                    f"Collection '{name}' expects {row[0]} dimensions, got {dimensions}"
                )
            self._conn.execute(
                "INSERT OR IGNORE INTO vector_collections(name, dimensions) VALUES (?, ?)",
                (name, dimensions),
            )
            if row and row[0] == 0 and dimensions:
                self._conn.execute(
                    "UPDATE vector_collections SET dimensions = ? WHERE name = ?",
                    (dimensions, name),
                )
            self._conn.commit()

    def collection_exists(self, name: str) -> bool:
        with self._lock:
            return (
                self._conn.execute(
                    "SELECT 1 FROM vector_collections WHERE name = ?", (name,)
                ).fetchone()
                is not None
            )

    def _expected_dimensions(self, collection: str) -> int:
        row = self._conn.execute(
            "SELECT dimensions FROM vector_collections WHERE name = ?", (collection,)
        ).fetchone()
        return int(row[0]) if row else 0

    def upsert(
        self,
        collection: str,
        chunk_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        with self._lock:
            self.create_collection(collection, dimensions=len(vector))
            expected = self._expected_dimensions(collection)
            if expected and len(vector) != expected:
                raise EmbeddingDimensionMismatchError(
                    f"Collection '{collection}' expects {expected} dimensions, got {len(vector)}"
                )
            self._conn.execute(
                """
                INSERT INTO vectors(collection, chunk_id, vector, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(collection, chunk_id) DO UPDATE SET
                    vector = excluded.vector, metadata = excluded.metadata
                """,
                (collection, chunk_id, json.dumps(vector), json.dumps(metadata)),
            )
            self._conn.commit()

    def batch_upsert(
        self,
        collection: str,
        items: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        for chunk_id, vector, metadata in items:
            self.upsert(collection, chunk_id, vector, metadata)

    def delete(self, collection: str, chunk_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM vectors WHERE collection = ? AND chunk_id = ?",
                (collection, chunk_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def delete_by_filter(self, collection: str, filters: dict[str, Any]) -> int:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, metadata FROM vectors WHERE collection = ?", (collection,)
            ).fetchall()
            ids = [row[0] for row in rows if self._matches_filters(json.loads(row[1]), filters)]
            self._conn.executemany(
                "DELETE FROM vectors WHERE collection = ? AND chunk_id = ?",
                [(collection, chunk_id) for chunk_id in ids],
            )
            self._conn.commit()
            return len(ids)

    def search(
        self,
        collection: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        with self._lock:
            expected = self._expected_dimensions(collection)
            if expected and len(query_vector) != expected:
                raise EmbeddingDimensionMismatchError(
                    f"Collection '{collection}' expects {expected} dimensions, "
                    f"got {len(query_vector)}"
                )
            rows = self._conn.execute(
                "SELECT chunk_id, vector, metadata FROM vectors WHERE collection = ?",
                (collection,),
            ).fetchall()
        results = []
        for chunk_id, vector_json, metadata_json in rows:
            metadata = json.loads(metadata_json)
            if filters and not self._matches_filters(metadata, filters):
                continue
            vector = json.loads(vector_json)
            results.append((chunk_id, self._cosine_similarity(query_vector, vector), metadata))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]

    def count(self, collection: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE collection = ?", (collection,)
            ).fetchone()
            return int(row[0]) if row else 0

    def health(self) -> bool:
        try:
            with self._lock:
                self._conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def clear(
        self,
        collection: str | None = None,
        *,
        filters: dict[str, Any] | None = None,
    ) -> None:
        if filters and collection:
            self.delete_by_filter(collection, filters)
            return
        with self._lock:
            if collection:
                self._conn.execute("DELETE FROM vectors WHERE collection = ?", (collection,))
            else:
                self._conn.execute("DELETE FROM vectors")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error as exc:
                raise StorageError(f"Failed to close vector store: {exc}") from exc

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    @staticmethod
    def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        from ragzen.security.filters import metadata_matches_filters

        return metadata_matches_filters(metadata, filters)
