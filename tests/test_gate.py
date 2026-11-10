import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.deps import get_gate
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


class CountingProvider(FakeProvider):
    """Records the peak number of calls in flight at once."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.in_flight = 0
        self.peak = 0

    async def complete_json(self, *args: Any, **kwargs: Any) -> Any:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            return await super().complete_json(*args, **kwargs)
        finally:
            self.in_flight -= 1


async def test_api_runs_up_to_max_concurrency_requests_in_parallel(make_app, metrics):
    gate = GpuGate(max_concurrency=2, queue_wait_sec=5, metrics=metrics)
    provider = CountingProvider([ADAPTED] * 3, delay_s=0.2)
    app = make_app(provider)
    app.dependency_overrides[get_gate] = lambda: gate

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://llm-service") as client:
        responses = await asyncio.gather(
            *(client.post("/v1/adapt", json=ADAPT_PAYLOAD) for _ in range(3))
        )

    assert [response.status_code for response in responses] == [200, 200, 200]
    # Two slots run side by side; the third request waits for one of them, never a third slot.
    assert provider.peak == 2


class HangingProvider(FakeProvider):
    """Answers after `delay_s` and records whether the call was cancelled on the way."""

    def __init__(self, script: list[Any], *, delay_s: float) -> None:
        super().__init__(script, delay_s=delay_s)
        self.cancelled = 0
        self.started = asyncio.Event()

    async def complete_json(self, system, user, schema, timeout_s):
        self.started.set()
        try:
            return await super().complete_json(system, user, schema, timeout_s)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise


@asynccontextmanager
async def running_server(app: FastAPI) -> AsyncIterator[int]:
    """A real uvicorn on a free port: ASGITransport cannot reproduce a dropped connection."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None, lifespan="off")
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve())
    assert await wait_until(lambda: server.started, timeout_s=5)
    try:
        yield server.servers[0].sockets[0].getsockname()[1]
    finally:
        server.should_exit = True
        await serving


async def send_post(port: int, path: str, payload: dict[str, Any]) -> asyncio.StreamWriter:
    """Send a complete request over a raw socket, which the test can close at any moment."""
    body = json.dumps(payload).encode()
    _, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        f"POST {path} HTTP/1.1\r\nHost: llm-service\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    await writer.drain()
    return writer


async def hang_up(writer: asyncio.StreamWriter) -> None:
    writer.close()
    await writer.wait_closed()


async def wait_until(condition: Callable[[], bool], timeout_s: float = 1.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            return False
        await asyncio.sleep(0.01)
    return True


async def test_dropped_client_frees_the_slot_for_the_next_request(make_app, metrics):
    provider = HangingProvider([ADAPTED, ADAPTED], delay_s=0.6)

    async with running_server(make_app(provider)) as port:
        writer = await send_post(port, "/v1/adapt", ADAPT_PAYLOAD)
        await asyncio.wait_for(provider.started.wait(), timeout=2)
        await hang_up(writer)

        assert await wait_until(lambda: metrics.in_flight == 0, timeout_s=0.3)
        assert provider.cancelled == 1

        provider.delay_s = 0
        async with AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            response = await client.post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 200
    assert len(provider.calls) == 2


async def test_client_dropped_while_queued_never_reaches_the_model(make_app, gate, metrics):
    provider = HangingProvider([ADAPTED], delay_s=0)

    async with running_server(make_app(provider)) as port:
        async with gate.slot():
            writer = await send_post(port, "/v1/adapt", ADAPT_PAYLOAD)
            assert await wait_until(lambda: metrics.queue_depth == 1)
            await hang_up(writer)

            assert await wait_until(lambda: metrics.queue_depth == 0, timeout_s=0.1)

        await asyncio.sleep(0.05)

    assert provider.calls == []


async def test_connected_client_still_gets_the_answer_on_a_real_server(make_app, metrics):
    provider = HangingProvider([ADAPTED], delay_s=0.1)

    async with running_server(make_app(provider)) as port:
        async with AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            response = await client.post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["recipe"]["base_recipe_id"] == ADAPTED["base_recipe_id"]
    assert provider.cancelled == 0
    assert metrics.runs_failed == 0


async def test_dropped_parse_client_frees_the_slot(make_app, metrics):
    provider = HangingProvider([{"items": []}], delay_s=0.6)

    async with running_server(make_app(provider)) as port:
        writer = await send_post(port, "/v1/parse-ingredients", {"lines": ["2 cloves garlic"]})
        await asyncio.wait_for(provider.started.wait(), timeout=2)
        await hang_up(writer)

        assert await wait_until(lambda: metrics.in_flight == 0, timeout_s=0.3)

    assert provider.cancelled == 1
