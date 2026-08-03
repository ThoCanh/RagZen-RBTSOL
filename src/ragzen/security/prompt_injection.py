"""Prompt injection detection.

This is a SUPPLEMENTARY defense layer. Permission filters remain mandatory.
No prompt injection detector is 100% reliable — this module uses heuristic
pattern matching to flag suspicious inputs for additional review.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("ragzen.security.prompt_injection")


@dataclass(frozen=True)
class InjectionCheckResult:
    """Result of a prompt injection check."""

    is_suspicious: bool
    confidence: float  # 0.0 to 1.0
    matched_patterns: list[str] = field(default_factory=list)
    message: str = ""


class PromptInjectionDetector:
    """Heuristic-based prompt injection detector.

    WARNING: This detector uses pattern matching and is NOT a complete
    defense against prompt injection. It should be used as one layer
    in a defense-in-depth strategy. Permission filters and context
    isolation remain the primary security controls.

    Checks for:
    - System prompt override attempts
    - Role-playing instructions
    - Instruction override patterns
    - Data exfiltration patterns
    - Delimiter injection
    """

    # Patterns that may indicate prompt injection attempts
    _PATTERNS: list[tuple[str, float, str]] = [
        # (pattern, confidence_weight, description)
        (
            r"(?i)\b(ignore|disregard|forget)\b.{0,30}\b(previous|above|prior|all)\b.{0,30}\b(instructions?|rules?|prompts?|constraints?)\b",
            0.9,
            "instruction_override",
        ),
        (
            r"(?i)\b(you are|act as|pretend to be|roleplay as"
            r"|behave as)\b.{0,50}\b(assistant|AI|bot|helper|system)\b",
            0.7,
            "role_override",
        ),
        (
            r"(?i)\b(system\s*prompt|system\s*message|system\s*instruction)\b",
            0.6,
            "system_prompt_reference",
        ),
        (
            r"(?i)\b(reveal|show|display|print|output)\b.{0,30}\b(system|hidden|secret|internal|private)\b.{0,30}\b(prompt|instruction|data|information)\b",
            0.8,
            "data_exfiltration",
        ),
        (
            r"(?i)\b(do not|don't|never)\b.{0,20}\b(follow|obey|listen)"
            r"\b.{0,20}\b(rules?|instructions?|guidelines?)\b",
            0.8,
            "rule_negation",
        ),
        (
            r"(?i)```\s*system\b",
            0.7,
            "code_block_system",
        ),
        (
            r"(?i)\b(new\s+instruction|override\s+instruction|replace\s+instruction)\b",
            0.85,
            "instruction_replacement",
        ),
        (
            r"(?i)\[\s*(?:SYSTEM|INST|ASSISTANT)\s*\]",
            0.75,
            "delimiter_injection",
        ),
        (
            r"(?i)<\|(?:im_start|im_end|system|user|assistant)\|>",
            0.8,
            "chat_template_injection",
        ),
        (
            r"(?i)\b(translate|convert|transform)\b.{0,20}\b(into|to)\b.{0,20}\b(base64|hex|binary|encoded)\b",
            0.5,
            "encoding_evasion",
        ),
    ]

    def __init__(self, *, custom_patterns: list[tuple[str, float, str]] | None = None) -> None:
        """Initialize the detector.

        Args:
            custom_patterns: Additional patterns as (regex, confidence, description) tuples.
        """
        self._patterns = list(self._PATTERNS)
        if custom_patterns:
            self._patterns.extend(custom_patterns)
        # Pre-compile patterns
        self._compiled: list[tuple[re.Pattern[str], float, str]] = [
            (re.compile(pattern), conf, desc)
            for pattern, conf, desc in self._patterns
        ]

    def check(self, text: str) -> InjectionCheckResult:
        """Check text for potential prompt injection patterns.

        Args:
            text: The text to check (query or document content).

        Returns:
            InjectionCheckResult with detection results.
        """
        if not text.strip():
            return InjectionCheckResult(is_suspicious=False, confidence=0.0)

        matched: list[str] = []
        max_confidence = 0.0

        for compiled, confidence, description in self._compiled:
            if compiled.search(text):
                matched.append(description)
                max_confidence = max(max_confidence, confidence)

        is_suspicious = max_confidence >= 0.5

        if is_suspicious:
            logger.warning(
                "Prompt injection detected (confidence=%.2f, patterns=%s)",
                max_confidence,
                matched,
            )

        return InjectionCheckResult(
            is_suspicious=is_suspicious,
            confidence=max_confidence,
            matched_patterns=matched,
            message=f"Detected {len(matched)} suspicious pattern(s)" if matched else "",
        )

    def check_document(self, content: str) -> InjectionCheckResult:
        """Check document content for embedded injection attempts.

        Documents may contain indirect prompt injection — malicious
        instructions embedded in document text that attempt to
        hijack the LLM when the content is used as context.

        Args:
            content: Document content to scan.

        Returns:
            InjectionCheckResult.
        """
        return self.check(content)
