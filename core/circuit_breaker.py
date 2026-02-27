"""
Async Circuit Breaker

Implements a production‑grade circuit breaker pattern for async Python code.
Supports three states: CLOSED, OPEN, HALF_OPEN.
Includes failure counting, automatic recovery after timeout, and thread‑safe locking.

Also provides a registry (`CircuitBreakerRegistry`) to manage named breakers,
ensuring isolation per endpoint.
"""

import asyncio
import time
import logging
import inspect
from enum import Enum
from typing import Callable, Awaitable, TypeVar, Optional, AsyncGenerator, Dict

T = TypeVar("T")  # Return type of the protected function


class BreakerState(str, Enum):
    """Enum representing the three possible states of the circuit breaker."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Exception raised when the circuit breaker is OPEN and refuses calls."""
    pass


class AsyncCircuitBreaker:
    """
    Asynchronous circuit breaker.

    Typical usage:
        breaker = AsyncCircuitBreaker(failure_threshold=5, recovery_timeout=60)
        result = await breaker.call(my_async_function)

    Also supports streaming via `run_generator_protected()`, context manager,
    and decorator syntax.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        logger: Optional[logging.Logger] = None,
        metrics_callback: Optional[Callable[[str, Optional[dict]], None]] = None,
    ):
        """
        Initialize the circuit breaker.

        :param failure_threshold: Number of consecutive failures before opening.
        :param recovery_timeout: Seconds after which an open circuit transitions to half‑open.
        :param half_open_max_calls: Max number of test calls allowed in half‑open state.
        :param logger: Optional logger for structured logging. If None, a default logger is used.
        :param metrics_callback: Optional function called with metric names and optional metadata.
                                 Can accept either one argument (name) or two (name, metadata).
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = BreakerState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0

        self._lock = asyncio.Lock()

        # Optional logging and metrics
        self._logger = logger or logging.getLogger(__name__)
        self._metrics_callback = metrics_callback

    # ----------------------------------------------------------------------
    # Public properties and inspection methods
    # ----------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current state of the circuit breaker (readable string)."""
        return self._state.value

    @property
    def failure_count(self) -> int:
        """Current number of consecutive failures recorded."""
        return self._failure_count

    async def is_open(self) -> bool:
        """
        Check whether the circuit is currently open (rejects calls).
        This method automatically attempts recovery if the timeout has passed.
        """
        async with self._lock:
            await self._maybe_recover()
            return self._state == BreakerState.OPEN

    async def reset(self) -> None:
        """
        Manually force the circuit breaker back to CLOSED state.
        Use with caution – typically for emergency recovery or administrative actions.
        """
        async with self._lock:
            self._state = BreakerState.CLOSED
            self._failure_count = 0
            self._opened_at = None
            self._half_open_calls = 0
            self._logger.info(
                "Circuit breaker manually reset to CLOSED",
                extra={"event": "breaker_reset"}
            )

    # ----------------------------------------------------------------------
    # Public recording methods (for external use)
    # ----------------------------------------------------------------------

    async def record_failure(self) -> None:
        """Public wrapper for recording a failure."""
        async with self._lock:
            await self._record_failure()
            self._emit_metric("circuit.failure")

    async def record_success(self) -> None:
        """Public wrapper for recording a success (closes the breaker if half‑open)."""
        async with self._lock:
            await self._record_success()
            self._emit_metric("circuit.success")

    # ----------------------------------------------------------------------
    # Core internal state management
    # ----------------------------------------------------------------------

    async def _maybe_recover(self) -> None:
        """
        Transition from OPEN to HALF_OPEN if the recovery timeout has elapsed.
        Must be called while holding the lock.
        """
        if self._state == BreakerState.OPEN:
            if self._opened_at is not None and (time.monotonic() - self._opened_at) >= self.recovery_timeout:
                self._state = BreakerState.HALF_OPEN
                self._half_open_calls = 0
                self._logger.info(
                    "Circuit breaker transitioned to HALF_OPEN after timeout",
                    extra={"event": "breaker_half_open", "timeout": self.recovery_timeout}
                )

    async def _record_failure(self) -> None:
        """
        Record a failure and possibly open the circuit.
        Must be called while holding the lock.
        """
        self._failure_count += 1
        self._logger.debug(
            "Failure recorded",
            extra={"event": "breaker_failure", "failure_count": self._failure_count}
        )

        if self._state == BreakerState.HALF_OPEN:
            # Any failure in half‑open state immediately re‑opens the circuit
            self._open()
            return

        if self._failure_count >= self.failure_threshold:
            self._open()

    def _open(self) -> None:
        """Open the circuit (transition to OPEN state)."""
        self._state = BreakerState.OPEN
        self._opened_at = time.monotonic()
        self._failure_count = 0
        self._logger.warning(
            "Circuit breaker opened",
            extra={"event": "breaker_open", "threshold": self.failure_threshold}
        )
        self._emit_metric("circuit.open")

    async def _record_success(self) -> None:
        """
        Record a success and close the circuit if in half‑open.
        Must be called while holding the lock.
        """
        if self._state in (BreakerState.HALF_OPEN, BreakerState.OPEN):
            self._state = BreakerState.CLOSED
            self._logger.info(
                "Circuit breaker closed after successful call",
                extra={"event": "breaker_closed"}
            )

        self._failure_count = 0
        self._half_open_calls = 0

    # ----------------------------------------------------------------------
    # Metrics helper
    # ----------------------------------------------------------------------

    def _emit_metric(self, name: str, metadata: Optional[dict] = None) -> None:
        """Safely call the metrics callback, handling both single‑ and two‑argument signatures."""
        if not self._metrics_callback:
            return
        try:
            # Attempt two‑argument call first, fall back to one argument
            try:
                self._metrics_callback(name, metadata or {})
            except TypeError:
                self._metrics_callback(name)
        except Exception:
            # Never let metrics callback break the circuit breaker
            self._logger.debug("Metrics callback raised an exception", exc_info=True)

    # ----------------------------------------------------------------------
    # Core `call()` method (for coroutines)
    # ----------------------------------------------------------------------

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """
        Execute the provided async function under circuit breaker protection.

        :param func: Async function to call.
        :return: Result of the function call.
        :raises CircuitBreakerOpenError: If the circuit is OPEN.
        :raises Exception: Any exception raised by `func` is propagated.
        """
        async with self._lock:
            await self._maybe_recover()

            if self._state == BreakerState.OPEN:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")

            if self._state == BreakerState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError("Circuit breaker HALF_OPEN limit reached")
                self._half_open_calls += 1

        # Execute the protected function outside the lock to avoid holding it during I/O
        try:
            result = await func()
        except Exception:
            # Failure path
            async with self._lock:
                await self._record_failure()
                self._emit_metric("circuit.failure")
            raise
        else:
            # Success path
            async with self._lock:
                await self._record_success()
                self._emit_metric("circuit.success")
            return result

    # ----------------------------------------------------------------------
    # Generator protection (for streaming)
    # ----------------------------------------------------------------------

    async def run_generator_protected(
        self, agen_factory: Callable[[], AsyncGenerator]
    ) -> AsyncGenerator:
        """
        Protect an async generator produced by `agen_factory()`.

        Usage:
            async for token in breaker.run_generator_protected(lambda: my_streaming_gen(...)):
                ...
        """
        async with self._lock:
            await self._maybe_recover()
            if self._state == BreakerState.OPEN:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
            if self._state == BreakerState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError("Circuit breaker HALF_OPEN limit reached")
                self._half_open_calls += 1

        agen = agen_factory()
        try:
            async for item in agen:
                yield item
        except asyncio.CancelledError:
            # Treat cancellation as failure, record it, then re-raise
            async with self._lock:
                await self._record_failure()
                self._emit_metric("circuit.failure", {"cancel": True})
            self._logger.info(
                "Circuit breaker recorded cancellation as failure",
                extra={"event": "breaker_cancelled"}
            )
            raise
        except Exception:
            async with self._lock:
                await self._record_failure()
                self._emit_metric("circuit.failure")
            raise
        else:
            async with self._lock:
                await self._record_success()
                self._emit_metric("circuit.success")
        finally:
            # Best-effort close
            try:
                await agen.aclose()
            except Exception:
                pass

    # ----------------------------------------------------------------------
    # Async context manager support
    # ----------------------------------------------------------------------

    async def __aenter__(self):
        async with self._lock:
            await self._maybe_recover()
            if self._state == BreakerState.OPEN:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
            if self._state == BreakerState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError("Circuit breaker HALF_OPEN limit reached")
                self._half_open_calls += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc is None:
            await self.record_success()
        else:
            await self.record_failure()
        return False  # Do not suppress exceptions

    # ----------------------------------------------------------------------
    # Decorator support
    # ----------------------------------------------------------------------

    def __call__(self, func):
        """
        Decorator that protects both coroutines and async generator functions.

        For coroutine functions, it wraps with `call()`.
        For async generator functions, it wraps with `run_generator_protected()`.
        """
        if inspect.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                return await self.call(lambda: func(*args, **kwargs))
            return wrapper

        if inspect.isasyncgenfunction(func):
            def gen_wrapper(*args, **kwargs):
                agen_factory = lambda: func(*args, **kwargs)
                return self.run_generator_protected(agen_factory)
            return gen_wrapper

        # For synchronous functions, return the original (or optionally raise)
        return func


# ----------------------------------------------------------------------
# Registry for managing multiple named circuit breakers
# ----------------------------------------------------------------------

class CircuitBreakerRegistry:
    """
    Thread‑safe registry that holds AsyncCircuitBreaker instances keyed by name.
    Ensures isolation per endpoint.
    """

    def __init__(self):
        self._breakers: Dict[str, AsyncCircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        logger: Optional[logging.Logger] = None,
        metrics_callback: Optional[Callable[[str, Optional[dict]], None]] = None,
    ) -> AsyncCircuitBreaker:
        """
        Retrieve a circuit breaker by name. If it does not exist, create one with the given parameters.
        The creation is atomic and uses the registry's lock.
        """
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = AsyncCircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                    half_open_max_calls=half_open_max_calls,
                    logger=logger,
                    metrics_callback=metrics_callback,
                )
            return self._breakers[name]

    async def remove(self, name: str) -> bool:
        """Remove a breaker from the registry. Returns True if it existed."""
        async with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                return True
            return False

    async def clear(self) -> None:
        """Remove all breakers."""
        async with self._lock:
            self._breakers.clear()


# Global default registry instance (optional, but convenient)
_default_registry = CircuitBreakerRegistry()


async def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    half_open_max_calls: int = 1,
    logger: Optional[logging.Logger] = None,
    metrics_callback: Optional[Callable[[str, Optional[dict]], None]] = None,
) -> AsyncCircuitBreaker:
    """
    Convenience function to retrieve a named circuit breaker from the default global registry.
    """
    return await _default_registry.get(
        name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        half_open_max_calls=half_open_max_calls,
        logger=logger,
        metrics_callback=metrics_callback,
    )


# ----------------------------------------------------------------------
# Example usage (can be removed in production)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    async def example():
        import random

        # Using the registry to get isolated breakers per endpoint
        breaker_a = await get_circuit_breaker("api-a", failure_threshold=2, recovery_timeout=3)
        breaker_b = await get_circuit_breaker("api-b", failure_threshold=5, recovery_timeout=10)

        async def flaky_call(name: str):
            if random.random() < 0.6:
                raise ValueError(f"Failure from {name}")
            return f"OK from {name}"

        for i in range(5):
            try:
                res_a = await breaker_a.call(lambda: flaky_call("A"))
                print(f"A: {res_a}")
            except Exception as e:
                print(f"A failed: {e}")

            try:
                res_b = await breaker_b.call(lambda: flaky_call("B"))
                print(f"B: {res_b}")
            except Exception as e:
                print(f"B failed: {e}")

            await asyncio.sleep(0.5)

    asyncio.run(example())

async def is_open(name: str) -> bool:
    """
    Convenience helper to check whether a named breaker is open.
    Used by modules that only need a read check.
    """
    breaker = await get_circuit_breaker(name)
    return await breaker.is_open()
