import asyncio

import pytest

from app.errors import QueueFullError
from app.metrics import Metrics
from app.usecases.gate import GpuGate
from tests.conftest import ADAPT_PAYLOAD, ADAPTED, FakeProvider


async def test_slots_are_serialised_with_concurrency_one():
    gate = GpuGate(max_concurrency=1, queue_wait_sec=5, metrics=Metrics())
    order: list[str] = []

    async def worker(name: str) -> None:
        async with gate.slot():
            order.append(f"{name}-in")
            await asyncio.sleep(0.02)
            order.append(f"{name}-out")

    await asyncio.gather(worker("a"), worker("b"))

    assert order in (
        ["a-in", "a-out", "b-in", "b-out"],
        ["b-in", "b-out", "a-in", "a-out"],
    )


async def test_queue_wait_timeout_raises_queue_full():
    gate = GpuGate(max_concurrency=1, queue_wait_sec=0.05, metrics=Metrics())

    async def hold() -> None:
        async with gate.slot():
            await asyncio.sleep(0.3)

    holder = asyncio.create_task(hold())
    await asyncio.sleep(0.01)

    with pytest.raises(QueueFullError) as raised:
        async with gate.slot():
            pass

    assert raised.value.status_code == 503
    assert raised.value.retry_after_sec >= 1
    await holder


async def test_slot_is_released_after_a_failure():
    gate = GpuGate(max_concurrency=1, queue_wait_sec=0.5, metrics=Metrics())

    with pytest.raises(RuntimeError):
        async with gate.slot():
            raise RuntimeError("boom")

    async with gate.slot() as slot:
        assert slot.queue_wait_ms >= 0


async def test_metrics_track_queue_and_repairs():
    metrics = Metrics()
    gate = GpuGate(max_concurrency=1, queue_wait_sec=0.5, metrics=metrics)

    async with gate.slot() as slot:
        assert metrics.in_flight == 1
        slot.retries = 1

    snapshot = metrics.snapshot()
    assert snapshot["runs_total"] == 1
    assert snapshot["runs_with_repair"] == 1
    assert snapshot["repair_rate"] == 1.0
    assert metrics.in_flight == 0
    assert metrics.queue_depth == 0


async def test_api_returns_503_with_retry_after_when_the_queue_is_full(make_async_client):
    provider = FakeProvider([ADAPTED, ADAPTED], delay_s=0.4)

    async with make_async_client(provider) as client:
        first, second = await asyncio.gather(
            client.post("/v1/adapt", json=ADAPT_PAYLOAD),
            client.post("/v1/adapt", json=ADAPT_PAYLOAD),
        )

    by_status = {response.status_code: response for response in (first, second)}
    assert set(by_status) == {200, 503}
    busy = by_status[503]
    assert busy.headers["Retry-After"] == "1"
    assert busy.json()["error"]["code"] == "queue_full"
