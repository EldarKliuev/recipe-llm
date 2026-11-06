import asyncio
from collections.abc import Coroutine
from typing import Any

from fastapi import Request

from app.errors import ClientDisconnectedError


async def cancel_on_disconnect[T](request: Request, work: Coroutine[Any, Any, T]) -> T:
    """Run `work`, cancelling it as soon as the client hangs up.

    Nobody is left to read the answer, yet the run would keep the only GPU slot and turn the
    next requests into 503. Cancelling releases the slot through `GpuGate` and closes the
    connection to the provider. The body is already read by now, so the next ASGI message can
    only be the disconnect.
    """
    work_task = asyncio.create_task(work)
    watcher = asyncio.create_task(_wait_for_disconnect(request))
    try:
        await asyncio.wait({work_task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if work_task.done():
            return work_task.result()
        work_task.cancel()
        await asyncio.wait({work_task})
        raise ClientDisconnectedError("client disconnected before the answer was ready")
    finally:
        watcher.cancel()
        work_task.cancel()


async def _wait_for_disconnect(request: Request) -> None:
    while (await request.receive())["type"] != "http.disconnect":
        pass
