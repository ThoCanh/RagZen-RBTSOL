"""Local BM25 sparse index.

Implements BM25 ranking using term frequency / inverse document frequency
with configurable tokenization. Supports Vietnamese text.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("ragzen.sparse.bm25")


def _default_tokenize(text: str) -> list[str]:
    """Simple tokenizer that handles Unicode/Vietnamese text.

    Splits on whitespace and punctuation, lowercases tokens.
    """
    # Split on non-alphanumeric characters (Unicode-aware)
    tokens = re.findall(r"\w+", text.lower(), re.UNICODE)
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """Local BM25 sparse index.

    Supports:
    - Configurable tokenizer (not hardcoded to one model)
    - Vietnamese text
    - Persistent storage
    - Tenant-scoped search via metadata filtering
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Any = None,
        path: str | Path | None = None,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._tokenize = tokenizer or _default_tokenize
        raw_path = Path(path) if path else None
        self._path = (
            raw_path / "index.json" if raw_path is not None and not raw_path.suffix else raw_path
        )
        self._lock = threading.RLock()

        # Document data
        self._docs: dict[str, dict[str, Any]] = {}  # chunk_id -> metadata
        self._doc_tokens: dict[str, list[str]] = {}  # chunk_id -> tokens
        self._doc_freqs: dict[str, int] = {}  # term -> doc count
        self._avg_dl: float = 0.0
        if self._path and self._path.exists():
            self.load(self._path)

    def add(
        self,
        chunk_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a document to the index."""
        with self._lock:
            if chunk_id in self._doc_tokens:
                self._remove_in_memory(chunk_id)
            tokens = self._tokenize(content)
            self._doc_tokens[chunk_id] = tokens
            self._docs[chunk_id] = metadata or {}

            # Update document frequencies
            unique_terms = set(tokens)
            for term in unique_terms:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

            # Update average document length
            self._recalculate_average_length()
            self._persist()

    def remove(self, chunk_id: str) -> bool:
        """Remove a document chunk from the index."""
        with self._lock:
            if chunk_id not in self._doc_tokens:
                return False
            self._remove_in_memory(chunk_id)
            self._recalculate_average_length()
            self._persist()
            return True

    def _remove_in_memory(self, chunk_id: str) -> None:
        tokens = self._doc_tokens[chunk_id]
        unique_terms = set(tokens)
        for term in unique_terms:
            if term in self._doc_freqs:
                self._doc_freqs[term] -= 1
                if self._doc_freqs[term] <= 0:
                    del self._doc_freqs[term]

        del self._doc_tokens[chunk_id]
        del self._docs[chunk_id]

    def _recalculate_average_length(self) -> None:
        total_tokens = sum(len(t) for t in self._doc_tokens.values())
        self._avg_dl = total_tokens / len(self._doc_tokens) if self._doc_tokens else 0

    def remove_by_document_id(self, document_id: str) -> int:
        """Remove all chunks belonging to a document_id.

        Returns number of removed chunks.
        """
        with self._lock:
            to_remove = [
                cid for cid, meta in self._docs.items() if meta.get("document_id") == document_id
            ]
            for cid in to_remove:
                self._remove_in_memory(cid)
            self._recalculate_average_length()
            self._persist()
            return len(to_remove)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search the index using BM25 scoring.

        Args:
            query: Search query text.
            top_k: Number of results.
            filters: Metadata filters (applied BEFORE scoring).

        Returns:
            List of (chunk_id, score, metadata) sorted by score desc.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        n = len(self._doc_tokens)
        if n == 0:
            return []

        scores: list[tuple[str, float, dict[str, Any]]] = []

        with self._lock:
            items = list(self._doc_tokens.items())
            docs = dict(self._docs)
            n = len(items)
        for chunk_id, doc_tokens in items:
            # Apply filters FIRST
            metadata = docs.get(chunk_id, {})
            if filters and not self._matches_filters(metadata, filters):
                continue

            score = self._compute_bm25(query_tokens, doc_tokens, n)
            if score > 0:
                scores.append((chunk_id, score, metadata))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _compute_bm25(self, query_tokens: list[str], doc_tokens: list[str], n: int) -> float:
        """Compute BM25 score for a document."""
        doc_len = len(doc_tokens)
        tf = Counter(doc_tokens)
        score = 0.0

        for term in query_tokens:
            if term not in tf:
                continue

            df = self._doc_freqs.get(term, 0)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            term_freq = tf[term]
            numerator = term_freq * (self._k1 + 1)
            denominator = term_freq + self._k1 * (
                1 - self._b + self._b * doc_len / max(self._avg_dl, 1)
            )
            score += idf * (numerator / denominator)

        return score

    def count(self) -> int:
        """Return number of indexed documents."""
        with self._lock:
            return len(self._doc_tokens)

    def save(self, path: str | Path) -> None:
        """Save index to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "docs": self._docs,
            "doc_tokens": self._doc_tokens,
            "doc_freqs": self._doc_freqs,
            "avg_dl": self._avg_dl,
            "k1": self._k1,
            "b": self._b,
        }
        temp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data), encoding="utf-8")
        temp_path.replace(save_path)

    def _persist(self) -> None:
        if self._path:
            self.save(self._path)

    def load(self, path: str | Path) -> None:
        """Load index from disk."""
        load_path = Path(path)
        if not load_path.exists():
            return
        with self._lock:
            data = json.loads(load_path.read_text(encoding="utf-8"))
            self._docs = data["docs"]
            self._doc_tokens = data["doc_tokens"]
            self._doc_freqs = data["doc_freqs"]
            self._avg_dl = data["avg_dl"]

    def clear(self, *, filters: dict[str, Any] | None = None) -> None:
        """Clear the index."""
        with self._lock:
            if filters:
                to_remove = [
                    chunk_id
                    for chunk_id, metadata in self._docs.items()
                    if self._matches_filters(metadata, filters)
                ]
                for chunk_id in to_remove:
                    self._remove_in_memory(chunk_id)
                self._recalculate_average_length()
            else:
                self._docs.clear()
                self._doc_tokens.clear()
                self._doc_freqs.clear()
                self._avg_dl = 0.0
            self._persist()

    def close(self) -> None:
        """Persist the index before shutdown."""
        with self._lock:
            self._persist()

    @staticmethod
    def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches all filter criteria."""
        from ragzen.security.filters import metadata_matches_filters

        return metadata_matches_filters(metadata, filters)
