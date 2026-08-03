"""Unit tests for RagZen CLI."""

from __future__ import annotations

import contextlib
from pathlib import Path

from ragzen.cli.main import main


class TestCLI:
    """CLI test suite."""

    def test_cli_version(self, capsys: object) -> None:
        with contextlib.suppress(SystemExit):
            main(["--version"])

    def test_cli_init(self, tmp_path: Path) -> None:
        target = tmp_path / "test_ragzen_dir"
        ret = main(["init", "--path", str(target)])
        assert ret == 0
        assert target.exists()

    def test_cli_stats(self) -> None:
        ret = main(["stats"])
        assert ret == 0

    def test_cli_health(self) -> None:
        ret = main(["health"])
        assert ret == 0

    def test_cli_doctor(self) -> None:
        ret = main(["doctor"])
        assert ret == 0

    def test_cli_config_validate(self) -> None:
        ret = main(["config", "validate"])
        assert ret == 0

    def test_cli_migrate(self) -> None:
        ret = main(["migrate", "status"])
        assert ret == 0

    def test_cli_ingest_and_query(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.txt"
        f.write_text("Quy trình kỹ thuật vận hành nhà máy.", encoding="utf-8")

        # Ingest
        ret_ingest = main(["ingest", str(f), "--tenant", "company-a"])
        assert ret_ingest == 0

        # Query
        ret_query = main(["query", "nhà máy", "--tenant", "company-a"])
        assert ret_query == 0

        # Search
        ret_search = main(["search", "nhà máy", "--tenant", "company-a"])
        assert ret_search == 0
