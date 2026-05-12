from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, TypeVar

import httpx

T = TypeVar("T")

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}


def _is_retryable(exc: BaseException) -> bool:
    """Return True for transient API errors that warrant a retry."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS_CODES
    return False


async def with_exponential_backoff(
    coro_factory: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> T:
    """
    Retry `coro_factory()` up to max_retries times with exponential back-off.
    Re-raises the last exception if all retries are exhausted.
    """
    last_exc: BaseException = RuntimeError("No attempts made")
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except BaseException as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == max_retries:
                raise
            delay = base_delay * (2**attempt)
            await asyncio.sleep(delay)
    raise last_exc  # unreachable but satisfies type checkers
