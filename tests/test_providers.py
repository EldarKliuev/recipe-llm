import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from app.errors import ProviderTimeoutError
from app.providers import CloudProvider, LMStudioProvider, MalformedModelJSON, base
from tests.conftest import ADAPTED


def wire_provider(provider_cls, *, message: dict) -> tuple[object, list[dict]]:
    """A real provider and SDK over a fake transport, to see the body that goes on the wire."""
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        choice = {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [choice],
            },
        )

    provider = provider_cls(base_url="http://llm.test/v1", api_key="k", model="m", max_tokens=16)
    provider._client = AsyncOpenAI(
        base_url="http://llm.test/v1",
        api_key="k",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return provider, sent


async def test_lmstudio_turns_reasoning_off_so_the_answer_lands_in_content():
    provider, sent = wire_provider(LMStudioProvider, message={"content": json.dumps(ADAPTED)})

    data, _ = await provider.complete_json("s", "u", {"name": "x", "schema": {}}, 5)

    assert sent[0]["reasoning_effort"] == "none"
    assert data == ADAPTED


async def test_cloud_does_not_send_reasoning_effort():
    provider, sent = wire_provider(CloudProvider, message={"content": json.dumps(ADAPTED)})

    await provider.complete_json("s", "u", {"name": "x", "schema": {}}, 5)

    assert "reasoning_effort" not in sent[0]


async def test_answer_only_in_reasoning_content_is_malformed_not_accepted():
    provider, _ = wire_provider(
        LMStudioProvider,
        message={"content": "", "reasoning_content": json.dumps(ADAPTED)},
    )

    with pytest.raises(MalformedModelJSON):
        await provider.complete_json("s", "u", {"name": "x", "schema": {}}, 5)


async def test_silent_provider_on_list_models_is_a_provider_timeout(monkeypatch):
    """A real socket that accepts and never answers: the case a mocked timeout cannot prove."""
    monkeypatch.setattr(base, "LIST_MODELS_TIMEOUT_SEC", 0.1)

    async def accept_and_stay_silent(reader, writer) -> None:
        await reader.read()
        writer.close()

    server = await asyncio.start_server(accept_and_stay_silent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    provider = LMStudioProvider(
        base_url=f"http://127.0.0.1:{port}/v1", api_key="k", model="m", max_tokens=8
    )
    try:
        with pytest.raises(ProviderTimeoutError):
            # The guard fails the test instead of waiting out the SDK's 600 s default.
            await asyncio.wait_for(provider.list_models(), timeout=3)
    finally:
        await provider.aclose()
        server.close()
