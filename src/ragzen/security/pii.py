"""PII detection interface.

This module provides an INTERFACE for PII detection. The built-in
implementation uses basic regex patterns and is NOT claimed to be
100% accurate. For production use, integrate a dedicated PII service.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger("ragzen.security.pii")


@dataclass(frozen=True)
class PIIMatch:
    """A detected PII occurrence."""

    pii_type: str
    start: int
    end: int
    confidence: float


@dataclass(frozen=True)
class PIIDetectionResult:
    """Result of PII detection scan."""

    has_pii: bool
    matches: list[PIIMatch] = field(default_factory=list)
    message: str = ""


@runtime_checkable
class PIIDetector(Protocol):
    """Protocol for PII detection implementations.

    WARNING: No PII detector is 100% accurate. This interface provides
    a hook for PII scanning, but the implementation should be chosen
    based on your compliance requirements.
    """

    def detect(self, text: str) -> PIIDetectionResult:
        """Scan text for PII.

        Args:
            text: The text to scan.

        Returns:
            PIIDetectionResult.
        """
        ...


class BasicPIIDetector:
    """Basic regex-based PII detector.

    WARNING: This implementation uses simple regex patterns and will
    have both false positives and false negatives. It is NOT suitable
    as the sole PII detection mechanism for regulatory compliance.
    For production, use a dedicated NER-based or ML-based PII service.

    Detects patterns that may match:
    - Email addresses
    - Phone numbers
    - Social Security Numbers (SSN)
    - Credit card numbers
    - IP addresses
    """

    _PATTERNS: list[tuple[str, str]] = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
        (r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b", "phone"),
        (r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", "ssn"),
        (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "credit_card"),
        (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "ip_address"),
    ]

    def __init__(self) -> None:
        self._compiled = [
            (re.compile(pattern), pii_type) for pattern, pii_type in self._PATTERNS
        ]

    def detect(self, text: str) -> PIIDetectionResult:
        """Scan text for basic PII patterns.

        Args:
            text: Text to scan.

        Returns:
            PIIDetectionResult with matches found.
        """
        matches: list[PIIMatch] = []

        for compiled, pii_type in self._compiled:
            for match in compiled.finditer(text):
                matches.append(
                    PIIMatch(
                        pii_type=pii_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.6,  # Low confidence for regex-based detection
                    )
                )

        return PIIDetectionResult(
            has_pii=len(matches) > 0,
            matches=matches,
            message=f"Found {len(matches)} potential PII occurrence(s)" if matches else "",
        )
