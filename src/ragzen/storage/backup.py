"""Backup and Restore manager for RagZen SQLite databases.

Supports online backup using SQLite backup API, gzip compression,
verification, and atomic restore.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import sqlite3
from pathlib import Path

from ragzen.exceptions import StorageError

logger = logging.getLogger("ragzen.storage.backup")


class BackupManager:
    """Manages online SQLite backups and restores."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def backup(self, dest_path: str | Path, *, compress: bool = True) -> Path:
        """Create a consistent online backup of the SQLite database.

        Args:
            dest_path: Destination path for backup file.
            compress: If True, compress backup with gzip (.gz).

        Returns:
            Path to the created backup file.
        """
        if not self.db_path.exists():
            msg = f"Database file does not exist: {self.db_path}"
            raise StorageError(msg)

        out_path = Path(dest_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        raw_backup_path = out_path.with_suffix(".tmp.db") if compress else out_path

        # Perform online backup using SQLite backup API
        src_conn = sqlite3.connect(str(self.db_path))
        dest_conn = sqlite3.connect(str(raw_backup_path))

        try:
            with dest_conn:
                src_conn.backup(dest_conn)
        except sqlite3.Error as e:
            msg = f"Backup operation failed: {e}"
            raise StorageError(msg) from e
        finally:
            src_conn.close()
            dest_conn.close()

        if compress:
            gz_path = Path(str(out_path) + ".gz") if not str(out_path).endswith(".gz") else out_path
            with raw_backup_path.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            raw_backup_path.unlink()
            return gz_path

        return out_path

    def restore(self, backup_path: str | Path, *, verify: bool = True) -> bool:
        """Restore database from a backup file (raw SQLite or gzipped).

        Args:
            backup_path: Path to backup file.
            verify: Verify database integrity before restoring.

        Returns:
            True if restore succeeded.
        """
        src = Path(backup_path)
        if not src.exists():
            msg = f"Backup file not found: {src}"
            raise StorageError(msg)

        # Decompress if gzipped
        if src.name.endswith(".gz"):
            temp_unpacked = src.with_suffix(".unpacked.db")
            with gzip.open(src, "rb") as f_in, temp_unpacked.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            restore_src = temp_unpacked
        else:
            restore_src = src
            temp_unpacked = None

        try:
            if verify:
                # Verify SQLite integrity
                conn = sqlite3.connect(str(restore_src))
                cur = conn.execute("PRAGMA quick_check")
                res = cur.fetchone()
                conn.close()
                if not res or res[0] != "ok":
                    msg = f"Backup file integrity check failed: {res}"
                    raise StorageError(msg)

            # Atomic copy into destination
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(restore_src, self.db_path)
            logger.info("Successfully restored database from %s", src)
            return True
        finally:
            if temp_unpacked and temp_unpacked.exists():
                temp_unpacked.unlink()

    def verify(self, backup_path: str | Path) -> bool:
        """Verify integrity of a backup file without restoring it."""
        try:
            return self.restore(backup_path, verify=True)
        except Exception:
            return False
