"""Tests for RagZen exception hierarchy."""

from __future__ import annotations

import pytest

from ragzen.exceptions import (
    AllProvidersFailedError,
    CircuitBreakerOpenError,
    CollectionNotFoundError,
    ConfigurationError,
    DuplicateDocumentError,
    FileTooLargeError,
    GenerationError,
    IngestionError,
    MigrationError,
    MissingOptionalDependencyError,
    PermissionDeniedError,
    PromptInjectionDetectedError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RagZenError,
    RetrievalError,
    SecurityContextRequiredError,
    SecurityError,
    StorageError,
    TenantIsolationError,
    TokenBudgetExceededError,
    TransactionError,
    UnsupportedFileTypeError,
)


class TestExceptionHierarchy:
    """Test that all exceptions inherit correctly from RagZenError."""

    def test_all_exceptions_inherit_from_ragzen_error(self) -> None:
        exceptions = [
            ConfigurationError("test"),
            SecurityError("test"),
            TenantIsolationError("test"),
            PermissionDeniedError("test"),
            SecurityContextRequiredError("test"),
            PromptInjectionDetectedError("test"),
            ProviderError("test"),
            ProviderTimeoutError(),
            ProviderUnavailableError(),
            CircuitBreakerOpenError(),
            AllProvidersFailedError(),
            IngestionError("test"),
            DuplicateDocumentError("test"),
            UnsupportedFileTypeError("test"),
            FileTooLargeError("test"),
            RetrievalError("test"),
            CollectionNotFoundError("test"),
            GenerationError("test"),
            TokenBudgetExceededError("test"),
            StorageError("test"),
            MigrationError("test"),
            TransactionError("test"),
            MissingOptionalDependencyError("pkg", "extra"),
        ]

        for exc in exceptions:
            assert isinstance(exc, RagZenError), f"{type(exc).__name__} must inherit RagZenError"

    def test_security_exceptions_inherit_from_security_error(self) -> None:
        sec_exceptions = [
            TenantIsolationError("test"),
            PermissionDeniedError("test"),
            SecurityContextRequiredError("test"),
            PromptInjectionDetectedError("test"),
        ]
        for exc in sec_exceptions:
            assert isinstance(exc, SecurityError)

    def test_provider_exceptions_inherit_from_provider_error(self) -> None:
        prov_exceptions = [
            ProviderTimeoutError(),
            ProviderUnavailableError(),
            CircuitBreakerOpenError(),
            AllProvidersFailedError(),
        ]
        for exc in prov_exceptions:
            assert isinstance(exc, ProviderError)


class TestRagZenError:
    """Test base error features."""

    def test_message(self) -> None:
        err = RagZenError("something broke")
        assert str(err) == "something broke"

    def test_details(self) -> None:
        err = RagZenError("error", details={"key": "value"})
        assert err.details == {"key": "value"}

    def test_details_default_empty(self) -> None:
        err = RagZenError("error")
        assert err.details == {}

    def test_catch_any_ragzen_error(self) -> None:
        with pytest.raises(RagZenError):
            raise TenantIsolationError("cross-tenant access")


class TestProviderError:
    """Test provider error specifics."""

    def test_provider_name(self) -> None:
        err = ProviderError("fail", provider="ollama")
        assert err.provider == "ollama"

    def test_retriable_flag(self) -> None:
        err = ProviderTimeoutError(provider="llm")
        assert err.retriable is True

        err2 = CircuitBreakerOpenError(provider="llm")
        assert err2.retriable is False

    def test_all_providers_failed_stores_errors(self) -> None:
        inner_errors = [ValueError("e1"), TimeoutError("e2")]
        err = AllProvidersFailedError(errors=inner_errors)
        assert len(err.errors) == 2


class TestMissingOptionalDependencyError:
    """Test dependency error messaging."""

    def test_message_format(self) -> None:
        err = MissingOptionalDependencyError("qdrant-client", "qdrant", "Qdrant vector store")
        assert "qdrant-client" in str(err)
        assert 'pip install "ragzen[qdrant]"' in str(err)
        assert "Qdrant vector store" in str(err)

    def test_message_without_feature(self) -> None:
        err = MissingOptionalDependencyError("redis", "redis")
        assert "redis" in str(err)
        assert 'pip install "ragzen[redis]"' in str(err)

    def test_attributes(self) -> None:
        err = MissingOptionalDependencyError("pkg", "extra", "feat")
        assert err.package == "pkg"
        assert err.extra == "extra"
        assert err.feature == "feat"
