import asyncio
import json
from time import perf_counter

import pytest

from app.domain.schemas import AdaptedRecipe
from app.errors import ProviderTimeoutError, ProviderUnavailableError
from app.usecases.runner import run_structured
from tests.conftest import ADAPT_PAYLOAD, ADAPTED, FakeProvider


def test_adapt_returns_recipe_and_run_metadata(make_client):
    provider = FakeProvider([ADAPTED])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["recipe"]["base_recipe_id"] == 1001
    assert body["recipe"]["substitutions"][0]["replace_with"] == "chili flakes"
    meta = body["meta"]
    assert meta == {
        "provider": "fake",
        "model": "fake-model",
        "prompt_version": meta["prompt_version"],
        "latency_ms": 7,
        "queue_wait_ms": meta["queue_wait_ms"],
        "retries": 0,
        "prompt_tokens": 100,
        "completion_tokens": 50,
    }
    assert response.headers["X-Request-ID"]

    # The model is asked for constrained JSON, not asked politely in the prompt.
    schema = provider.calls[0]["schema"]
    assert schema["name"] == "adapted_recipe"
    assert schema["strict"] is True


def test_request_id_from_core_api_is_echoed_back(make_client):
    response = make_client(FakeProvider([ADAPTED])).post(
        "/v1/adapt", json=ADAPT_PAYLOAD, headers={"X-Request-ID": "core-api-42"}
    )

    assert response.headers["X-Request-ID"] == "core-api-42"
    assert response.status_code == 200


def test_unparsable_output_is_repaired_on_retry(make_client):
    provider = FakeProvider(["not json at all {", ADAPTED])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["meta"]["retries"] == 1
    assert response.json()["meta"]["latency_ms"] == 14  # both attempts counted
    repair_prompt = provider.calls[1]["user"]
    assert "was rejected" in repair_prompt
    assert "not json at all {" in repair_prompt


def test_schema_violation_is_repaired_on_retry(make_client):
    provider = FakeProvider([{"title": "No id here", "steps": ["do things"]}, ADAPTED])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["meta"]["retries"] == 1
    assert "base_recipe_id" in provider.calls[1]["user"]


def test_invented_base_recipe_id_is_rejected_and_repaired(make_client):
    invented = {**ADAPTED, "base_recipe_id": 999999}
    provider = FakeProvider([invented, ADAPTED])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["recipe"]["base_recipe_id"] == 1001
    assert "999999 is not one of the candidate recipes" in provider.calls[1]["user"]


def test_persistently_invalid_output_gives_422_for_core_api_to_degrade(make_client):
    provider = FakeProvider(["{", "{", "{"])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_model_output"
    assert "repair attempts" in body["error"]["message"]
    assert len(provider.calls) == 3  # one attempt plus two repairs


def test_model_that_keeps_inventing_ids_eventually_fails(make_client):
    invented = {**ADAPTED, "base_recipe_id": 4242}
    provider = FakeProvider([invented, invented, invented])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 422
    assert "4242" in json.dumps(response.json()["error"]["detail"])


def test_provider_timeout_is_surfaced_as_504(make_client):
    provider = FakeProvider([ProviderTimeoutError("model did not answer within 5s")])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "provider_timeout"


def test_slow_invalid_answers_stop_at_the_run_budget_with_422(make_client, settings):
    settings.llm_run_budget_sec = 0.3
    provider = FakeProvider(["{", "{", "{"], delay_s=0.2)

    started = perf_counter()
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)
    elapsed = perf_counter() - started

    assert response.status_code == 422
    assert "not json" in response.json()["error"]["detail"]["reason"]
    assert len(provider.calls) == 2  # the second attempt ran out of budget, no third one
    assert elapsed < 0.5  # three full attempts would take 0.6


def test_hanging_model_stops_at_the_run_budget_with_504(make_client, settings):
    settings.llm_run_budget_sec = 0.05
    provider = FakeProvider([ADAPTED], delay_s=1.0)

    started = perf_counter()
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "provider_timeout"
    assert perf_counter() - started < 0.5


def test_repair_gets_only_what_is_left_of_the_budget_and_still_works(make_client, settings):
    settings.llm_run_budget_sec = 1.0
    provider = FakeProvider(["{", ADAPTED], delay_s=0.2)

    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["meta"]["retries"] == 1
    assert provider.calls[0]["timeout_s"] <= 1.0
    assert provider.calls[1]["timeout_s"] <= 0.8


def test_attempt_timeout_still_caps_each_attempt_under_a_larger_budget(make_client, settings):
    settings.llm_timeout_sec = 2
    provider = FakeProvider([ADAPTED])

    make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert provider.calls[0]["timeout_s"] == 2


def test_timeout_after_an_invalid_answer_gives_422_not_504(make_client):
    provider = FakeProvider(["{", ProviderTimeoutError("model did not answer within 5s")])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 422
    assert "not json" in response.json()["error"]["detail"]["reason"]
    assert len(provider.calls) == 2


def test_unreachable_provider_after_an_invalid_answer_stays_503(make_client):
    provider = FakeProvider(["{", ProviderUnavailableError("cannot reach the model")])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 503


async def test_queue_wait_counts_against_the_run_budget(make_async_client, settings):
    settings.llm_run_budget_sec = 0.25
    provider = FakeProvider([ADAPTED, ADAPTED], delay_s=0.15)

    async with make_async_client(provider) as client:
        responses = await asyncio.gather(
            client.post("/v1/adapt", json=ADAPT_PAYLOAD),
            client.post("/v1/adapt", json=ADAPT_PAYLOAD),
        )

    assert sorted(response.status_code for response in responses) == [200, 504]
    assert provider.calls[1]["timeout_s"] < 0.2  # 0.15 of its 0.25 went on the queue


async def test_no_attempt_starts_once_the_queue_used_up_the_budget(gate):
    provider = FakeProvider([ADAPTED, ADAPTED], delay_s=0.15)

    def run(budget_s: float):
        return run_structured(
            provider=provider,
            gate=gate,
            output_model=AdaptedRecipe,
            schema_name="adapted_recipe",
            system="system",
            user="user",
            timeout_s=5,
            budget_s=budget_s,
        )

    holder = asyncio.create_task(run(5))
    await asyncio.sleep(0.01)

    with pytest.raises(ProviderTimeoutError):
        await run(0.05)

    await holder
    assert len(provider.calls) == 1


def test_unreachable_provider_is_surfaced_as_503(make_client):
    provider = FakeProvider([ProviderUnavailableError("cannot reach the model")])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "provider_unavailable"


def test_malformed_request_is_400_not_422(make_client):
    # 422 is reserved for "the model failed", so core-api can tell the two apart.
    response = make_client(FakeProvider([ADAPTED])).post("/v1/adapt", json={"recipes": []})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
