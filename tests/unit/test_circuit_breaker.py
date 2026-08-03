"""Unit tests for CircuitBreaker."""

from __future__ import annotations

import time

import pytest

from ragzen.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitState


class TestCircuitBreaker:
    def test_circuit_breaker_transitions(self) -> None:
        cb = CircuitBreaker("test-cb", failure_threshold=2, recovery_timeout_seconds=0.2)
        assert cb.state == CircuitState.CLOSED

        def failing_func() -> None:
            msg = "fail"
            raise RuntimeError(msg)

        # 1st failure
        with pytest.raises(RuntimeError):
            cb.call(failing_func)
        assert cb.state == CircuitState.CLOSED

        # 2nd failure -> OPEN
        with pytest.raises(RuntimeError):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN

        # 3rd call rejected fast without calling func
        with pytest.raises(CircuitBreakerError) as exc_info:
            cb.call(failing_func)
        assert exc_info.value.state == CircuitState.OPEN

        # Wait recovery timeout -> HALF_OPEN
        time.sleep(0.25)
        assert cb.state == CircuitState.HALF_OPEN

        # Successful call in HALF_OPEN resets to CLOSED
        def success_func() -> str:
            return "ok"

        res = cb.call(success_func)
        assert res == "ok"
        assert cb.state == CircuitState.CLOSED
