"""Unit tests for BackupManager."""

from __future__ import annotations

from pathlib import Path

from ragzen.models import Document
from ragzen.storage.backup import BackupManager
from ragzen.storage.documents import DocumentRegistry


class TestBackupManager:
    def test_backup_and_restore(self, tmp_path: Path) -> None:
        db_path = tmp_path / "orig.db"
        reg = DocumentRegistry(db_path)
        reg.save(Document(tenant_id="t1", content="Backup test content"))
        reg.close()

        bm = BackupManager(db_path)
        backup_gz = bm.backup(tmp_path / "backup.db.gz", compress=True)
        assert backup_gz.exists()

        # Restore into new location
        restored_db_path = tmp_path / "restored.db"
        bm_restored = BackupManager(restored_db_path)
        res = bm_restored.restore(backup_gz, verify=True)
        assert res is True

        reg_restored = DocumentRegistry(restored_db_path)
        assert reg_restored.count(tenant_id="t1") == 1
        reg_restored.close()
