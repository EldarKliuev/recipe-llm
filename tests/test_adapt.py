import json

from app.errors import ProviderTimeoutError, ProviderUnavailableError
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
