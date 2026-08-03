"""Database migration engine for RagZen.

Manages SQLite schema versioning, applying migrations sequentially,
and verifying schema integrity.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ragzen.exceptions import StorageError

logger = logging.getLogger("ragzen.storage.migrations")


@dataclass(frozen=True)
class MigrationStep:
    """Represents a single database schema migration step."""

    version: int
    name: str
    up_sql: str
    python_handler: Callable[[sqlite3.Connection], None] | None = None


# Registry of versioned migrations
_MIGRATIONS: list[MigrationStep] = [
    MigrationStep(
        version=1,
        name="initial_schema",
        up_sql="""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
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

        CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);
        CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(tenant_id, status);
        CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(tenant_id, content_hash);
        CREATE INDEX IF NOT EXISTS idx_documents_idempotency
            ON documents(idempotency_key) WHERE idempotency_key IS NOT NULL;
        """,
    ),
    MigrationStep(
        version=2,
        name="add_document_versioning_and_audit",
        up_sql="""
        CREATE TABLE IF NOT EXISTS document_versions (
            version_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            tenant_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_doc_versions ON document_versions(document_id, version);
        """,
    ),
]


class MigrationEngine:
    """SQLite migration manager for RagZen."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def current_version(self) -> int:
        """Get the current applied schema version."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if not cur.fetchone():
                return 0
            cur = conn.execute("SELECT MAX(version) FROM schema_version")
            row = cur.fetchone()
            return row[0] if (row and row[0] is not None) else 0
        finally:
            conn.close()

    def plan(self) -> list[MigrationStep]:
        """Return migrations that need to be applied."""
        current = self.current_version()
        return [m for m in _MIGRATIONS if m.version > current]

    def apply(self) -> int:
        """Apply all pending migrations. Returns count of applied migrations."""
        pending = self.plan()
        if not pending:
            logger.info("Database schema is up to date (version %d)", self.current_version())
            return 0

        conn = sqlite3.connect(str(self.db_path))
        applied_count = 0

        # Ensure schema_version has name column if upgraded from legacy schema
        try:
            cur = conn.execute("PRAGMA table_info(schema_version)")
            cols = [row[1] for row in cur.fetchall()]
            if "name" not in cols:
                conn.execute("ALTER TABLE schema_version ADD COLUMN name TEXT NOT NULL DEFAULT 'initial'")
                conn.commit()
        except sqlite3.Error:
            pass

        try:
            for step in pending:
                logger.info("Applying migration v%d (%s)...", step.version, step.name)
                conn.execute("BEGIN IMMEDIATE")
                conn.executescript(step.up_sql)
                if step.python_handler:
                    step.python_handler(conn)

                conn.execute(
                    "INSERT INTO schema_version (version, name, applied_at) VALUES (?, ?, ?)",
                    (step.version, step.name, datetime.now(UTC).isoformat()),
                )
                conn.commit()
                applied_count += 1
            return applied_count
        except sqlite3.Error as e:
            conn.rollback()
            msg = f"Migration failed at step v{pending[applied_count].version}: {e}"
            raise StorageError(msg) from e
        finally:
            conn.close()

    def status(self) -> dict[str, object]:
        """Return migration status summary."""
        current = self.current_version()
        pending = self.plan()
        return {
            "current_version": current,
            "target_version": _MIGRATIONS[-1].version if _MIGRATIONS else 0,
            "pending_count": len(pending),
            "pending_versions": [m.version for m in pending],
        }
