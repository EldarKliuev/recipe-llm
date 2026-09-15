from app.errors import ProviderUnavailableError
from tests.conftest import ADAPT_PAYLOAD, ADAPTED, FakeProvider


def test_health_is_independent_of_the_model(make_client):
    provider = FakeProvider([], models=[])
    response = make_client(provider).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model"] == "fake-model"


def test_ready_requires_the_configured_model_to_be_loaded(make_client):
    response = make_client(FakeProvider([], models=["fake-model"])).get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_reports_a_different_loaded_model(make_client):
    response = make_client(FakeProvider([], models=["some-other-model"])).get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "model_not_loaded"
    assert body["available"] == ["some-other-model"]


def test_ready_reports_an_unreachable_provider(make_client):
    class Down(FakeProvider):
        async def list_models(self) -> list[str]:
            raise ProviderUnavailableError("cannot list provider models")

    response = make_client(Down([])).get("/ready")

    assert response.status_code == 503
    assert response.json()["reason"] == "provider_unavailable"


def test_metrics_lite_counts_runs_and_repairs(make_client):
    client = make_client(FakeProvider(["{", ADAPTED]))
    assert client.post("/v1/adapt", json=ADAPT_PAYLOAD).status_code == 200

    body = client.get("/metrics-lite").json()

    assert body["runs_total"] == 1
    assert body["runs_with_repair"] == 1
    assert body["repair_rate"] == 1.0
    assert body["queue_depth"] == 0
    assert body["in_flight"] == 0
    assert body["max_concurrency"] == 1
    assert body["latency_ms_p95"] is not None
