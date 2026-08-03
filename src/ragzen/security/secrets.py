"""Secret management and redaction.

Provides secure handling of secrets: redaction from logs, environment-based
retrieval, and a provider interface for vault integration.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("ragzen.security.secrets")

# Patterns that look like secrets in log output
_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(secret|password|passwd|token|credential)\s*[:=]\s*\S+"),
    re.compile(r"(?i)(authorization|bearer)\s*[:=]?\s*\S+"),
    re.compile(r"(?i)(private[_-]?key)\s*[:=]\s*\S+"),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI-style keys
    re.compile(r"(?i)eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}"),  # JWT
]

_REDACTED = "***REDACTED***"


def redact_secrets(text: str) -> str:
    """Redact potential secrets from text.

    Args:
        text: Text that may contain secrets.

    Returns:
        Text with secrets replaced by ***REDACTED***.
    """
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(_REDACTED, result)
    return result


def redact_dict(data: dict[str, Any], sensitive_keys: set[str] | None = None) -> dict[str, Any]:
    """Redact sensitive values from a dictionary.

    Args:
        data: Dictionary potentially containing secrets.
        sensitive_keys: Set of key names to redact. Defaults to common secret keys.

    Returns:
        New dictionary with sensitive values redacted.
    """
    if sensitive_keys is None:
        sensitive_keys = {
            "api_key",
            "apikey",
            "api-key",
            "secret",
            "password",
            "passwd",
            "token",
            "authorization",
            "private_key",
            "secret_key",
            "access_token",
            "refresh_token",
            "credentials",
        }

    result: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower().replace("-", "_")
        if key_lower in sensitive_keys:
            result[key] = _REDACTED
        elif isinstance(value, dict):
            result[key] = redact_dict(value, sensitive_keys)
        elif isinstance(value, str):
            result[key] = redact_secrets(value)
        else:
            result[key] = value
    return result


def safe_log_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Safely prepare a config dictionary for logging.

    Redacts all known sensitive fields.

    Args:
        config_dict: Configuration as a dictionary.

    Returns:
        Safe-to-log version.
    """
    return redact_dict(config_dict)
