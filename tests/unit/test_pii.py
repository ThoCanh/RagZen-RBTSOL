"""Unit tests for PII detection module."""

from __future__ import annotations

from ragzen.security.pii import BasicPIIDetector


class TestPIIDetector:
    def setup_method(self) -> None:
        self.detector = BasicPIIDetector()

    def test_detect_email(self) -> None:
        res = self.detector.detect("Contact us at test.user@company.com for support.")
        assert res.has_pii is True
        assert any(m.pii_type == "email" for m in res.matches)

    def test_detect_phone(self) -> None:
        res = self.detector.detect("Call 555-123-4567 immediately.")
        assert res.has_pii is True
        assert any(m.pii_type == "phone" for m in res.matches)

    def test_detect_ssn(self) -> None:
        res = self.detector.detect("SSN: 123-45-6789.")
        assert res.has_pii is True
        assert any(m.pii_type == "ssn" for m in res.matches)

    def test_clean_text(self) -> None:
        res = self.detector.detect("This is plain text with no sensitive data.")
        assert res.has_pii is False
        assert len(res.matches) == 0
