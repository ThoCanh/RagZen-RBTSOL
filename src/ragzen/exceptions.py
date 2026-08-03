"""RagZen exception hierarchy.

All exceptions inherit from RagZenError to allow catching any framework error.
Exceptions are organized by domain: configuration, security, provider, ingestion,
retrieval, generation, storage, and dependency management.
"""

from __future__ import annotations


class RagZenError(Exception):
    """Base exception for all RagZen errors."""

    def __init__(self, message: str, *, details: dict[str, object] | None = None) -> None:
        self.details = details or {}
        super().__init__(message)


# --- Configuration ---


class ConfigurationError(RagZenError):
    """Invalid or missing configuration."""


class SecretNotFoundError(ConfigurationError):
    """A required secret was not found in any secret provider."""


# --- Security ---


class SecurityError(RagZenError):
    """Base for all security-related errors."""


class TenantIsolationError(SecurityError):
    """Cross-tenant data access attempted."""


class PermissionDeniedError(SecurityError):
    """User lacks required permissions for the requested operation."""


class SecurityContextRequiredError(SecurityError):
    """Operation requires a SecurityContext but none was provided."""


class PromptInjectionDetectedError(SecurityError):
    """Potential prompt injection detected in input."""


# --- Provider ---


class ProviderError(RagZenError):
    """Error from an external provider (LLM, embedding, vector store, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        retriable: bool = False,
        details: dict[str, object] | None = None,
    ) -> None:
        self.provider = provider
        self.retriable = retriable
        super().__init__(message, details=details)


class ProviderTimeoutError(ProviderError):
    """Provider did not respond within the configured timeout."""

    def __init__(
        self,
        message: str = "Provider timed out",
        *,
        provider: str = "",
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, provider=provider, retriable=True, details=details)


class ProviderUnavailableError(ProviderError):
    """Provider is down or unreachable."""

    def __init__(
        self,
        message: str = "Provider unavailable",
        *,
        provider: str = "",
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, provider=provider, retriable=True, details=details)


class CircuitBreakerOpenError(ProviderError):
    """Circuit breaker is open due to repeated failures."""

    def __init__(
        self,
        message: str = "Circuit breaker is open",
        *,
        provider: str = "",
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, provider=provider, retriable=False, details=details)


class AllProvidersFailedError(ProviderError):
    """All providers in a fallback chain failed."""

    def __init__(
        self,
        message: str = "All providers failed",
        *,
        provider: str = "fallback_chain",
        errors: list[Exception] | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        self.errors = errors or []
        super().__init__(message, provider=provider, retriable=False, details=details)


# --- Ingestion ---


class IngestionError(RagZenError):
    """Error during document ingestion."""


class DuplicateDocumentError(IngestionError):
    """Document with the same idempotency key or content hash already exists."""


class UnsupportedFileTypeError(IngestionError):
    """File type is not supported or not in the MIME allowlist."""


class FileTooLargeError(IngestionError):
    """File exceeds the configured size limit."""


class DocumentValidationError(IngestionError):
    """Document failed validation checks."""


# --- Retrieval ---


class RetrievalError(RagZenError):
    """Error during retrieval phase."""


class CollectionNotFoundError(RetrievalError):
    """Vector store collection does not exist."""


class EmbeddingDimensionMismatchError(RetrievalError):
    """Query embedding dimensions don't match the index dimensions."""


# --- Generation ---


class GenerationError(RagZenError):
    """Error during LLM generation phase."""


class TokenBudgetExceededError(GenerationError):
    """Context or prompt exceeds the configured token budget."""


class CitationValidationError(GenerationError):
    """One or more citations in the response are invalid."""


class InsufficientEvidenceError(GenerationError):
    """Not enough context to generate a reliable answer."""


# --- Storage ---


class StorageError(RagZenError):
    """Error in the storage layer."""


class MigrationError(StorageError):
    """Database migration failed."""


class TransactionError(StorageError):
    """Transaction commit or rollback failed."""


# --- Dependencies ---


class MissingOptionalDependencyError(RagZenError):
    """An optional dependency is required but not installed.

    Provides a helpful install instruction to the user.
    """

    def __init__(self, package: str, extra: str, feature: str = "") -> None:
        self.package = package
        self.extra = extra
        self.feature = feature
        feature_msg = f" for {feature}" if feature else ""
        message = (
            f"'{package}' is required{feature_msg} but not installed.\n"
            f"Install with: pip install \"ragzen[{extra}]\""
        )
        super().__init__(message)
