"""SQLite-backed document registry with WAL mode and transactions.

Provides persistent storage for document metadata with full CRUD,
tenant-scoped queries, content hash tracking, and version management.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragzen.exceptions import StorageError, TransactionError
from ragzen.models import AccessControl, Document, DocumentStatus, RetentionPolicy

logger = logging.getLogger("ragzen.storage.documents")

_SCHEMA_VERSION = 1

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    name TEXT NOT NULL DEFAULT 'initial',
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL DEFAULT 1,
    tenant_id TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT '',
    source_uri TEXT NOT NULL DEFAULT '',
    file_name TEXT NOT NULL DEFAULT '',
    mime_type TEXT NOT NULL DEFAULT 'text/plain',
    page_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    indexed_at TEXT,
    document_type TEXT NOT NULL DEFAULT '',
    access_control TEXT,
    retention_policy TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    idempotency_key TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_tenant
    ON documents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash
    ON documents(tenant_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_idempotency
    ON documents(idempotency_key)
    WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_documents_type
    ON documents(tenant_id, document_type);
"""


class DocumentRegistry:
    """SQLite-backed document registry.

    Uses WAL mode for concurrent read access and provides
    transactional guarantees for write operations.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Create a new SQLite connection with WAL mode."""
        conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _initialize(self) -> None:
        """Create tables if they don't exist."""
        try:
            self._conn.executescript(_CREATE_TABLES)
            # Check if schema version exists
            cur = self._conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO schema_version (version, name, applied_at) VALUES (?, ?, ?)",
                    (_SCHEMA_VERSION, "initial_schema", datetime.now(UTC).isoformat()),
                )
            self._conn.commit()
        except sqlite3.Error as e:
            msg = f"Failed to initialize document registry: {e}"
            raise StorageError(msg) from e

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for transactional operations."""
        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            yield cursor
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            msg = f"Transaction failed: {e}"
            raise TransactionError(msg) from e

    def save(self, document: Document, *, idempotency_key: str = "") -> None:
        """Save or update a document in the registry.

        Args:
            document: The document to save.
            idempotency_key: Optional key for duplicate detection.
        """
        now = datetime.now(UTC).isoformat()
        ac_json = (
            document.access_control.model_dump_json()
            if document.access_control
            else None
        )
        rp_json = (
            document.retention_policy.model_dump_json()
            if document.retention_policy
            else None
        )

        with self.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (
                    document_id, version, tenant_id, content_hash,
                    metadata, source, source_uri, file_name,
                    mime_type, page_count, created_at, updated_at,
                    indexed_at, document_type, access_control,
                    retention_policy, status, idempotency_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    version = excluded.version,
                    content_hash = excluded.content_hash,
                    metadata = excluded.metadata,
                    source = excluded.source,
                    source_uri = excluded.source_uri,
                    file_name = excluded.file_name,
                    mime_type = excluded.mime_type,
                    page_count = excluded.page_count,
                    updated_at = excluded.updated_at,
                    indexed_at = excluded.indexed_at,
                    document_type = excluded.document_type,
                    access_control = excluded.access_control,
                    retention_policy = excluded.retention_policy,
                    status = excluded.status
                """,
                (
                    document.document_id,
                    document.version,
                    document.tenant_id,
                    document.content_hash or document.compute_content_hash(),
                    json.dumps(document.metadata),
                    document.source,
                    document.source_uri,
                    document.file_name,
                    document.mime_type,
                    document.page_count,
                    document.created_at.isoformat(),
                    now,
                    document.indexed_at.isoformat() if document.indexed_at else None,
                    document.document_type,
                    ac_json,
                    rp_json,
                    document.status.value,
                    idempotency_key or None,
                ),
            )

    def get(self, document_id: str, *, tenant_id: str = "") -> Document | None:
        """Get a document by ID, optionally scoped to a tenant.

        Args:
            document_id: The document ID.
            tenant_id: If provided, enforce tenant scope.

        Returns:
            Document or None if not found.
        """
        if tenant_id:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE document_id = ? AND tenant_id = ?",
                (document_id, tenant_id),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_document(row)

    def find_by_idempotency_key(
        self, key: str, *, tenant_id: str = ""
    ) -> Document | None:
        """Find a document by idempotency key."""
        if tenant_id:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE idempotency_key = ? AND tenant_id = ?",
                (key, tenant_id),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE idempotency_key = ?",
                (key,),
            ).fetchone()

        return self._row_to_document(row) if row else None

    def find_by_content_hash(
        self, content_hash: str, tenant_id: str
    ) -> Document | None:
        """Find a document by content hash within a tenant."""
        row = self._conn.execute(
            "SELECT * FROM documents WHERE content_hash = ? AND tenant_id = ?",
            (content_hash, tenant_id),
        ).fetchone()
        return self._row_to_document(row) if row else None

    def list_by_tenant(
        self,
        tenant_id: str,
        *,
        status: DocumentStatus | None = None,
        document_type: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        """List documents for a tenant with optional filtering."""
        query = "SELECT * FROM documents WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]

        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        if document_type:
            query += " AND document_type = ?"
            params.append(document_type)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_document(row) for row in rows]

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        tenant_id: str = "",
    ) -> bool:
        """Update document status. Returns True if document was found."""
        now = datetime.now(UTC).isoformat()
        params: list[Any] = [status.value, now, document_id]
        query = "UPDATE documents SET status = ?, updated_at = ? WHERE document_id = ?"
        if tenant_id:
            query += " AND tenant_id = ?"
            params.append(tenant_id)

        with self.transaction() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount > 0

    def delete(self, document_id: str, *, tenant_id: str = "") -> bool:
        """Delete a document from the registry. Returns True if deleted."""
        params: list[Any] = [document_id]
        query = "DELETE FROM documents WHERE document_id = ?"
        if tenant_id:
            query += " AND tenant_id = ?"
            params.append(tenant_id)

        with self.transaction() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount > 0

    def count(self, *, tenant_id: str = "") -> int:
        """Count documents, optionally by tenant."""
        if tenant_id:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM documents WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    def clear(self, *, tenant_id: str = "") -> int:
        """Delete all documents, optionally scoped to a tenant. Returns count."""
        if tenant_id:
            with self.transaction() as cursor:
                cursor.execute(
                    "DELETE FROM documents WHERE tenant_id = ?", (tenant_id,)
                )
                return cursor.rowcount
        else:
            with self.transaction() as cursor:
                cursor.execute("DELETE FROM documents")
                return cursor.rowcount

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        """Convert a database row to a Document model."""
        ac = None
        if row["access_control"]:
            ac = AccessControl.model_validate_json(row["access_control"])

        rp = None
        if row["retention_policy"]:
            rp = RetentionPolicy.model_validate_json(row["retention_policy"])

        indexed_at = None
        if row["indexed_at"]:
            indexed_at = datetime.fromisoformat(row["indexed_at"])

        return Document(
            document_id=row["document_id"],
            version=row["version"],
            tenant_id=row["tenant_id"],
            content_hash=row["content_hash"],
            metadata=json.loads(row["metadata"]),
            source=row["source"],
            source_uri=row["source_uri"],
            file_name=row["file_name"],
            mime_type=row["mime_type"],
            page_count=row["page_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            indexed_at=indexed_at,
            document_type=row["document_type"],
            access_control=ac,
            retention_policy=rp,
            status=DocumentStatus(row["status"]),
        )
