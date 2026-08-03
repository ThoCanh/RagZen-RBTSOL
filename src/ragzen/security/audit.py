"""Audit logging for compliance and debugging.

Provides the AuditSink protocol and a default file-based implementation
that writes structured audit events as JSON lines.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from ragzen.models import AuditEvent

logger = logging.getLogger("ragzen.security.audit")


@runtime_checkable
class AuditSink(Protocol):
    """Protocol for audit event sinks.

    Implement this protocol to integrate with your organization's
    audit infrastructure (SIEM, log aggregator, database, etc.).
    """

    def record(self, event: AuditEvent) -> None:
        """Record an audit event.

        Args:
            event: The audit event to record.
        """
        ...

    def flush(self) -> None:
        """Flush any buffered events."""
        ...


class LogAuditSink:
    """Audit sink that writes events to Python's logging system.

    Uses structured JSON format for machine-parseable audit logs.
    """

    def __init__(self, logger_name: str = "ragzen.audit") -> None:
        self._logger = logging.getLogger(logger_name)

    def record(self, event: AuditEvent) -> None:
        """Record an audit event via Python logging."""
        event_dict = event.model_dump(mode="json")
        self._logger.info(json.dumps(event_dict, default=str))

    def flush(self) -> None:
        """Flush handlers."""
        for handler in self._logger.handlers:
            handler.flush()


class FileAuditSink:
    """Audit sink that writes JSON-line events to a file.

    Each event is written as a single JSON line for easy parsing.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")

    def record(self, event: AuditEvent) -> None:
        """Append an audit event to the file."""
        event_dict = event.model_dump(mode="json")
        self._file.write(json.dumps(event_dict, default=str) + "\n")

    def flush(self) -> None:
        """Flush the file buffer."""
        self._file.flush()

    def close(self) -> None:
        """Close the file."""
        self._file.flush()
        self._file.close()


class InMemoryAuditSink:
    """In-memory audit sink for testing."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, event: AuditEvent) -> None:
        """Store event in memory."""
        self.events.append(event)

    def flush(self) -> None:
        """No-op for in-memory sink."""

    def clear(self) -> None:
        """Clear all recorded events."""
        self.events.clear()
