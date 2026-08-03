"""Unit tests for MigrationEngine."""

from __future__ import annotations

from pathlib import Path

from ragzen.storage.migrations import MigrationEngine


class TestMigrationEngine:
    def test_migration_lifecycle(self, tmp_path: Path) -> None:
        db_path = tmp_path / "migrate.db"
        engine = MigrationEngine(db_path)

        assert engine.current_version() == 0
        pending = engine.plan()
        assert len(pending) == 2

        # Apply migrations
        applied = engine.apply()
        assert applied == 2
        assert engine.current_version() == 2

        # Status check
        st = engine.status()
        assert st["current_version"] == 2
        assert st["pending_count"] == 0

        # Idempotent apply
        assert engine.apply() == 0
