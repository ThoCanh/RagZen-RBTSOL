"""Qdrant vector-store adapter."""

from __future__ import annotations

from typing import Any

from ragzen.exceptions import MissingOptionalDependencyError
from ragzen.security.filters import metadata_matches_filters


class QdrantVectorStore:
    """Qdrant-backed vector store with tenant filtering and ACL verification."""

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        api_key: str = "",
        timeout_seconds: float = 10.0,
        path: str = "",
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                "qdrant-client", "qdrant", "Qdrant vector store"
            ) from exc
        kwargs: dict[str, Any] = {"timeout": timeout_seconds}
        if path:
            kwargs["path"] = path
        else:
            kwargs["url"] = url
            if api_key:
                kwargs["api_key"] = api_key
        self._client = QdrantClient(**kwargs)

    def create_collection(self, name: str, *, dimensions: int = 0) -> None:
        if self.collection_exists(name):
            return
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:  # pragma: no cover - guarded by constructor
            raise MissingOptionalDependencyError("qdrant-client", "qdrant") from exc
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimensions, distance=Distance.COSINE),
        )

    def collection_exists(self, name: str) -> bool:
        return bool(self._client.collection_exists(name))

    def upsert(
        self,
        collection: str,
        chunk_id: str,
        vector: list[float],
        metadata: dict[str, Any],
    ) -> None:
        from qdrant_client.models import PointStruct

        self.create_collection(collection, dimensions=len(vector))
        self._client.upsert(
            collection_name=collection,
            points=[PointStruct(id=chunk_id, vector=vector, payload=metadata)],
            wait=True,
        )

    def batch_upsert(
        self,
        collection: str,
        items: list[tuple[str, list[float], dict[str, Any]]],
    ) -> None:
        if not items:
            return
        from qdrant_client.models import PointStruct

        self.create_collection(collection, dimensions=len(items[0][1]))
        self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(id=chunk_id, vector=vector, payload=metadata)
                for chunk_id, vector, metadata in items
            ],
            wait=True,
        )

    def delete(self, collection: str, chunk_id: str) -> bool:
        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=[chunk_id]),
            wait=True,
        )
        return True

    def _qdrant_filter(self, filters: dict[str, Any] | None) -> Any:
        if not filters:
            return None
        from qdrant_client.models import (
            FieldCondition,
            Filter,
            IsEmptyCondition,
            MatchAny,
            MatchValue,
            PayloadField,
        )

        must: list[Any] = []
        if "tenant_id" in filters:
            must.append(
                FieldCondition(key="tenant_id", match=MatchValue(value=filters["tenant_id"]))
            )
        for key, expected in filters.items():
            if key == "tenant_id" or key.startswith("_security"):
                continue
            match = (
                MatchAny(any=expected) if isinstance(expected, list) else MatchValue(value=expected)
            )
            must.append(FieldCondition(key=key, match=match))

        permissions = filters.get("_security_permissions", [])
        has_security_filters = any(key.startswith("_security") for key in filters)
        if has_security_filters and "*" not in permissions:
            acl_must: list[Any] = []
            acl_pairs = (
                ("departments", "_security_departments"),
                ("roles", "_security_roles"),
                ("groups", "_security_groups"),
                ("permissions", "_security_permissions"),
            )
            for payload_key, filter_key in acl_pairs:
                granted = filters.get(filter_key, [])
                if payload_key == "departments" and "all" in granted:
                    continue
                conditions: list[Any] = [IsEmptyCondition(is_empty=PayloadField(key=payload_key))]
                if granted:
                    conditions.append(FieldCondition(key=payload_key, match=MatchAny(any=granted)))
                acl_must.append(Filter(should=conditions))
            granted_attributes = filters.get("_security_attributes", {})
            for key in filters.get("_security_attribute_keys", []):
                value = granted_attributes.get(key)
                conditions = [IsEmptyCondition(is_empty=PayloadField(key=f"attributes.{key}"))]
                if value is not None:
                    conditions.append(
                        FieldCondition(
                            key=f"attributes.{key}",
                            match=MatchValue(value=value),
                        )
                    )
                acl_must.append(Filter(should=conditions))
            acl_filter = Filter(must=acl_must)
            user_id = filters.get("_security_user_id", "")
            if user_id:
                must.append(
                    Filter(
                        should=[
                            FieldCondition(key="owner_id", match=MatchValue(value=user_id)),
                            acl_filter,
                        ]
                    )
                )
            else:
                must.append(acl_filter)
        return Filter(must=must)

    def delete_by_filter(self, collection: str, filters: dict[str, Any]) -> int:
        from qdrant_client.models import FilterSelector

        matched = self._scroll_all(collection, filters)
        if not matched:
            return 0
        if set(filters) == {"tenant_id"}:
            self._client.delete(
                collection_name=collection,
                points_selector=FilterSelector(filter=self._qdrant_filter(filters)),
                wait=True,
            )
        else:
            from qdrant_client.models import PointIdsList

            ids = [point_id for point_id, _ in matched]
            self._client.delete(
                collection_name=collection,
                points_selector=PointIdsList(points=ids),
                wait=True,
            )
        return len(matched)

    def search(
        self,
        collection: str,
        query_vector: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        query_filter = self._qdrant_filter(filters)
        limit = top_k
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=collection,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            points = response.points
        else:  # pragma: no cover - compatibility with older qdrant-client
            points = self._client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        results = []
        for point in points:
            metadata = dict(point.payload or {})
            if filters and not metadata_matches_filters(metadata, filters):
                continue
            results.append((str(point.id), float(point.score), metadata))
            if len(results) >= top_k:
                break
        return results

    def _scroll_all(
        self, collection: str, filters: dict[str, Any]
    ) -> list[tuple[Any, dict[str, Any]]]:
        points, _ = self._client.scroll(
            collection_name=collection,
            scroll_filter=self._qdrant_filter(filters),
            limit=10000,
            with_payload=True,
        )
        return [
            (point.id, dict(point.payload or {}))
            for point in points
            if metadata_matches_filters(dict(point.payload or {}), filters)
        ]

    def count(self, collection: str) -> int:
        if not self.collection_exists(collection):
            return 0
        result = self._client.count(collection_name=collection, exact=True)
        return int(result.count)

    def health(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def clear(
        self,
        collection: str | None = None,
        *,
        filters: dict[str, Any] | None = None,
    ) -> None:
        if not collection:
            for item in self._client.get_collections().collections:
                self._client.delete_collection(item.name)
            return
        if filters:
            self.delete_by_filter(collection, filters)
        elif self.collection_exists(collection):
            self._client.delete_collection(collection)

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            close()
