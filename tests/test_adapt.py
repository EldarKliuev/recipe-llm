import asyncio
import json
import logging
from time import perf_counter

import pytest

from app.domain.prompts import PROMPT_VERSION
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


@pytest.mark.parametrize("steps", [[], [""], ["  ", "\n"]], ids=["none", "blank", "whitespace"])
def test_empty_steps_are_rejected_and_repaired(make_client, steps):
    # core-api rejects such an answer as empty_steps and starts a whole new generation.
    provider = FakeProvider([{**ADAPTED, "steps": steps}, ADAPTED])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["meta"]["retries"] == 1
    assert response.json()["recipe"]["steps"] == ADAPTED["steps"]
    assert "steps must not be empty" in provider.calls[1]["user"]


def test_one_blank_step_among_real_ones_is_not_worth_a_repair(make_client):
    # core-api drops blank steps itself, so a repair here would only spend GPU time.
    steps = ["Boil the pasta.", "", "Toss and serve."]
    provider = FakeProvider([{**ADAPTED, "steps": steps}])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["meta"]["retries"] == 0
    assert len(provider.calls) == 1


def test_step_minutes_are_returned_alongside_steps(make_client):
    response = make_client(FakeProvider([ADAPTED])).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["recipe"]["step_minutes"] == [10, None, None]


@pytest.mark.parametrize("minutes", [[10], [10, None, None, 5]], ids=["short", "long"])
def test_step_minutes_of_the_wrong_length_are_repaired(make_client, minutes):
    # core-api pairs minutes with steps by index, so a shifted list would put a timer
    # on the wrong step.
    provider = FakeProvider([{**ADAPTED, "step_minutes": minutes}, ADAPTED])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 200
    assert response.json()["recipe"]["step_minutes"] == ADAPTED["step_minutes"]
    assert response.json()["meta"]["retries"] == 1
    assert "step_minutes must have one entry per step" in provider.calls[1]["user"]


def test_step_minutes_that_never_match_the_steps_give_422(make_client):
    bad = {**ADAPTED, "step_minutes": [10]}
    provider = FakeProvider([bad, bad, bad])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 422
    assert len(provider.calls) == 3
    assert "step_minutes" in response.json()["error"]["detail"]["reason"]


def test_persistently_invalid_output_gives_422_for_core_api_to_degrade(make_client):
    provider = FakeProvider(["{", "{", "{"])
    response = make_client(provider).post("/v1/adapt", json=ADAPT_PAYLOAD)

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_model_output"
    assert "repair attempts" in body["error"]["message"]
    assert len(provider.calls) == 3  # one attempt plus two repairs
    # core-api stores these in generation_runs: failed runs are the ones worth analysing.
    meta = body["error"]["detail"]["meta"]
    assert meta == {
        "provider": "fake",
        "model": "fake-model",
        "prompt_version": PROMPT_VERSION,
        "latency_ms": 21,  # all three answers counted
        "queue_wait_ms": meta["queue_wait_ms"],
        "retries": 2,
        "prompt_tokens": 300,
        "completion_tokens": 150,
    }


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
    detail = response.json()["error"]["detail"]
    assert "not json" in detail["reason"]
    assert len(provider.calls) == 2
    # The repair attempt that timed out still counts; only the answered one has latency and tokens.
    assert detail["meta"]["retries"] == 1
    assert detail["meta"]["latency_ms"] == 7
    assert detail["meta"]["prompt_tokens"] == 100


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


def test_unexpected_error_keeps_the_error_format_request_id_and_log_line(make_client, caplog):
    # A bug, not a model or provider failure: nothing in the service maps it to a status.
    provider = FakeProvider([RuntimeError("bug in our code")])

    with caplog.at_level(logging.INFO):
        response = make_client(provider).post(
            "/v1/adapt", json=ADAPT_PAYLOAD, headers={"X-Request-ID": "core-api-42"}
        )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "core-api-42"
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert body["request_id"] == "core-api-42"
    # The exception text stays in our log; core-api gets only the code.
    assert "bug in our code" not in response.text

    crash = next(r for r in caplog.records if r.getMessage() == "unhandled error")
    assert crash.exc_info and isinstance(crash.exc_info[1], RuntimeError)
    handled = next(r for r in caplog.records if r.getMessage() == "request handled")
    assert handled.status == 500
    assert handled.request_id == "core-api-42"


def test_unexpected_error_frees_the_slot_for_the_next_request(make_client, gate):
    client = make_client(FakeProvider([RuntimeError("bug in our code"), ADAPTED]))

    assert client.post("/v1/adapt", json=ADAPT_PAYLOAD).status_code == 500
    assert client.post("/v1/adapt", json=ADAPT_PAYLOAD).status_code == 200
