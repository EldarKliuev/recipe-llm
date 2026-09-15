import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from math import ceil
from time import perf_counter

from app.errors import QueueFullError
from app.metrics import Metrics

log = logging.getLogger(__name__)


@dataclass
class Slot:
    """A held GPU slot. The caller fills in the outcome; the gate reports it to metrics."""

    queue_wait_ms: int
    retries: int = 0
    failed: bool = False


class GpuGate:
    """One model on one GPU answers serially, so requests are serialised here rather than
    left to pile up as timeouts.

    The semaphore is per-process, which is why the service runs with a single worker.
    """

    def __init__(self, *, max_concurrency: int, queue_wait_sec: float, metrics: Metrics) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._queue_wait_sec = queue_wait_sec
        self._metrics = metrics

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[Slot]:
        queue_started = perf_counter()
        self._metrics.enter_queue()
        try:
            async with asyncio.timeout(self._queue_wait_sec):
                await self._semaphore.acquire()
        except TimeoutError as exc:
            log.warning(
                "queue wait exceeded",
                extra={"queue_wait_sec": self._queue_wait_sec},
            )
            raise QueueFullError(
                "the model is busy, retry shortly",
                retry_after_sec=max(1, ceil(self._queue_wait_sec)),
            ) from exc
        finally:
            self._metrics.leave_queue()

        slot = Slot(queue_wait_ms=int((perf_counter() - queue_started) * 1000))
        self._metrics.start_run()
        run_started = perf_counter()
        try:
            yield slot
        except BaseException:
            slot.failed = True
            raise
        finally:
            self._metrics.finish_run(
                latency_ms=(perf_counter() - run_started) * 1000,
                retries=slot.retries,
                failed=slot.failed,
            )
            self._semaphore.release()
