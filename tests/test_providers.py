import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from app.config import Settings
from app.errors import ProviderTimeoutError, ProviderUnavailableError
from app.providers import (
    CloudProvider,
    LMStudioProvider,
    MalformedModelJSON,
    base,
    create_provider,
)
from tests.conftest import ADAPTED

SCHEMA = {"name": "x", "schema": {}}


def completion(message: dict, *, choices: bool = True) -> httpx.Response:
    choice = {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}
    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [choice] if choices else [],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        },
    )


def wire_provider(
    provider_cls,
    *,
    message: dict | None = None,
    respond: Callable[[httpx.Request], httpx.Response] | None = None,
) -> tuple[object, list[httpx.Request]]:
    """A real provider and SDK over a fake transport, to see what goes on the wire.

    The transport is swapped with with_options, so the client keeps the settings the
    provider's own constructor gave it (max_retries above all).
    """
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return respond(request) if respond else completion(message or {})

    provider = provider_cls(base_url="http://llm.test/v1", api_key="k", model="m", max_tokens=16)
    provider._client = provider._client.with_options(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    return provider, sent


def body(request: httpx.Request) -> dict:
    return json.loads(request.content)


async def test_lmstudio_turns_reasoning_off_so_the_answer_lands_in_content():
    provider, sent = wire_provider(LMStudioProvider, message={"content": json.dumps(ADAPTED)})

    data, _ = await provider.complete_json("s", "u", SCHEMA, 5)

    assert body(sent[0])["reasoning_effort"] == "none"
    assert data == ADAPTED


async def test_cloud_does_not_send_reasoning_effort():
    provider, sent = wire_provider(CloudProvider, message={"content": json.dumps(ADAPTED)})

    await provider.complete_json("s", "u", SCHEMA, 5)

    assert "reasoning_effort" not in body(sent[0])


async def test_answer_only_in_reasoning_content_is_malformed_not_accepted():
    provider, _ = wire_provider(
        LMStudioProvider,
        message={"content": "", "reasoning_content": json.dumps(ADAPTED)},
    )

    with pytest.raises(MalformedModelJSON):
        await provider.complete_json("s", "u", SCHEMA, 5)


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


async def test_request_carries_the_schema_token_limit_and_the_per_call_timeout():
    """The runner hands each attempt what is left of the run budget; it must reach the socket."""
    provider, sent = wire_provider(CloudProvider, message={"content": json.dumps(ADAPTED)})

    _, meta = await provider.complete_json("sys", "usr", SCHEMA, 7.5)

    sent_body = body(sent[0])
    assert sent_body["response_format"] == {"type": "json_schema", "json_schema": SCHEMA}
    assert sent_body["max_tokens"] == 16
    assert sent_body["model"] == "m"
    assert sent_body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert sent[0].extensions["timeout"]["read"] == 7.5
    assert (meta.provider, meta.prompt_tokens, meta.completion_tokens) == ("cloud", 11, 7)


@pytest.mark.parametrize("status", [400, 500])
async def test_http_error_is_not_retried_and_reads_as_unavailable(status):
    """Retries belong to repair and to the caller; core-api shows 503 as "model busy"."""
    provider, sent = wire_provider(
        LMStudioProvider, respond=lambda _: httpx.Response(status, json={"error": "boom"})
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await provider.complete_json("s", "u", SCHEMA, 5)

    assert len(sent) == 1
    assert exc_info.value.status_code == 503
    assert f"HTTP {status}" in exc_info.value.message


async def test_read_timeout_is_a_provider_timeout():
    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no answer", request=request)

    provider, sent = wire_provider(LMStudioProvider, respond=stall)

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await provider.complete_json("s", "u", SCHEMA, 5)

    assert len(sent) == 1
    assert exc_info.value.status_code == 504


async def test_unreachable_model_is_unavailable():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider, _ = wire_provider(LMStudioProvider, respond=refuse)

    with pytest.raises(ProviderUnavailableError):
        await provider.complete_json("s", "u", SCHEMA, 5)


async def test_json_wrapped_in_fences_and_prose_is_extracted():
    text = f"Here you go:\n```json\n{json.dumps(ADAPTED)}\n```\nEnjoy!"
    provider, _ = wire_provider(LMStudioProvider, message={"content": text})

    data, _ = await provider.complete_json("s", "u", SCHEMA, 5)

    assert data == ADAPTED


async def test_no_choices_is_malformed_with_usage_kept():
    provider, _ = wire_provider(LMStudioProvider, respond=lambda _: completion({}, choices=False))

    with pytest.raises(MalformedModelJSON) as exc_info:
        await provider.complete_json("s", "u", SCHEMA, 5)

    assert exc_info.value.meta.completion_tokens == 7


async def test_json_that_is_not_an_object_is_malformed():
    provider, _ = wire_provider(LMStudioProvider, message={"content": "[1, 2]"})

    with pytest.raises(MalformedModelJSON) as exc_info:
        await provider.complete_json("s", "u", SCHEMA, 5)

    assert exc_info.value.raw == "[1, 2]"


async def test_list_models_returns_ids():
    page = {
        "object": "list",
        "data": [{"id": "a", "object": "model", "created": 0, "owned_by": "o"}],
    }
    provider, _ = wire_provider(LMStudioProvider, respond=lambda _: httpx.Response(200, json=page))

    assert await provider.list_models() == ["a"]


async def test_list_models_error_status_is_unavailable():
    provider, _ = wire_provider(
        LMStudioProvider, respond=lambda _: httpx.Response(503, json={"error": "loading"})
    )

    with pytest.raises(ProviderUnavailableError):
        await provider.list_models()


@pytest.mark.parametrize(
    ("name", "provider_cls"), [("lmstudio", LMStudioProvider), ("cloud", CloudProvider)]
)
async def test_create_provider_picks_the_backend_from_settings(name, provider_cls):
    settings = Settings(
        _env_file=None,
        llm_provider=name,
        llm_base_url="http://llm.test/v1",
        llm_api_key="k",
        llm_model="some-model",
        llm_max_tokens=99,
    )

    provider = create_provider(settings)
    try:
        assert type(provider) is provider_cls
        assert provider.model == "some-model"
        assert provider._max_tokens == 99
        assert provider._client.max_retries == 0
        assert str(provider._client.base_url) == "http://llm.test/v1/"
    finally:
        await provider.aclose()
