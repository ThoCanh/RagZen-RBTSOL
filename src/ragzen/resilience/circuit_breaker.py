"""Circuit Breaker pattern for provider resilience.

States:
- CLOSED: Normal operation, calls pass through.
- OPEN: Failure threshold exceeded, calls fail fast.
- HALF_OPEN: Trial period after recovery timeout, probing for recovery.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypeVar

from ragzen.exceptions import ProviderError

logger = logging.getLogger("ragzen.resilience.circuit_breaker")

T = TypeVar("T")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(ProviderError):
    """Raised when call is rejected by an open circuit breaker."""

    def __init__(self, name: str, state: CircuitState, reset_seconds: float) -> None:
        super().__init__(
            f"Circuit breaker '{name}' is {state.value.upper()}. Service temporarily unavailable.",
            provider=name,
            retriable=True,
        )
        self.name = name
        self.state = state
        self.reset_seconds = reset_seconds


class CircuitBreaker:
    """Circuit Breaker protecting against cascading failures."""

    def __init__(
        self,
        name: str = "default",
        *,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 10.0,
        expected_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.expected_exceptions = expected_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._success_count = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state."""
        if (
            self._state == CircuitState.OPEN
            and time.time() - self._last_failure_time >= self.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            logger.info("Circuit breaker '%s' entering HALF_OPEN probe state", self.name)
        return self._state

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute a call through the circuit breaker.

        Raises:
            CircuitBreakerError: If circuit is OPEN.
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            reset_in = max(0.0, self.recovery_timeout - (time.time() - self._last_failure_time))
            raise CircuitBreakerError(self.name, current_state, reset_in)

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exceptions as e:
            self._on_failure()
            raise e

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker '%s' recovered! Transitioning to CLOSED.", self.name)
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker '%s' OPENED after %d consecutive failures",
                self.name,
                self._failure_count,
            )

    def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
