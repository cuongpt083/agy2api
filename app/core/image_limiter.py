import asyncio
import os
import weakref
from typing import Optional, Tuple

DEFAULT_CONCURRENCY = 2
DEFAULT_TIMEOUT_S = 60.0
POLL_INTERVAL_S = 2.0

# Store per event loop: loop -> (Semaphore, limit)
_SEMAPHORES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_WAITING_COUNTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def get_image_concurrency_limit() -> int:
    val = os.environ.get("AGY_IMAGE_MAX_CONCURRENCY", "").strip()
    if val.isdigit() and int(val) > 0:
        return int(val)
    return DEFAULT_CONCURRENCY


def get_image_queue_timeout() -> float:
    val = os.environ.get("AGY_IMAGE_QUEUE_TIMEOUT_S", "").strip()
    try:
        f = float(val)
        if f > 0:
            return f
    except ValueError:
        pass
    return DEFAULT_TIMEOUT_S


def get_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    limit = get_image_concurrency_limit()
    cached = _SEMAPHORES.get(loop)
    if cached is None or cached[1] != limit:
        sem = asyncio.Semaphore(limit)
        _SEMAPHORES[loop] = (sem, limit)
        return sem
    return cached[0]


class ImageSlotAcquisition:
    """Async context manager to safely acquire an image job slot."""

    def __init__(self, timeout: Optional[float] = None) -> None:
        self.timeout = timeout if timeout is not None else get_image_queue_timeout()
        self.acquired = False
        self._sem: Optional[asyncio.Semaphore] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def __aenter__(self) -> "ImageSlotAcquisition":
        self._loop = asyncio.get_running_loop()
        self._sem = get_semaphore()
        _WAITING_COUNTS[self._loop] = _WAITING_COUNTS.get(self._loop, 0) + 1
        try:
            acquire_task = asyncio.create_task(self._sem.acquire())
            try:
                await asyncio.wait_for(asyncio.shield(acquire_task), timeout=self.timeout)
                self.acquired = True
                return self
            except (asyncio.TimeoutError, asyncio.CancelledError):
                if acquire_task.done() and not acquire_task.cancelled() and not acquire_task.exception():
                    # Acquired just as the timeout/cancel fired: release to prevent permit leak
                    self._sem.release()
                else:
                    acquire_task.cancel()
                raise
        finally:
            current = _WAITING_COUNTS.get(self._loop, 1)
            _WAITING_COUNTS[self._loop] = max(0, current - 1)

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.acquired and self._sem is not None:
            self._sem.release()
            self.acquired = False


def get_image_waiting_count() -> int:
    try:
        loop = asyncio.get_running_loop()
        return _WAITING_COUNTS.get(loop, 0)
    except RuntimeError:
        return 0


def reset_limiter_state() -> None:
    """Reset limiter state (primarily for test teardown)."""
    _SEMAPHORES.clear()
    _WAITING_COUNTS.clear()
