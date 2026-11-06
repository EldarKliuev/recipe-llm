import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.deps import get_gate, get_metrics, get_provider, get_settings
from app.main import create_app
from app.metrics import Metrics
from app.providers.base import LLMMeta, MalformedModelJSON
from app.usecases.gate import GpuGate

ADAPTED = {
    "base_recipe_id": 1001,
    "title": "Garlic Oil Spaghetti",
    "steps": ["Boil the pasta.", "Warm garlic in oil.", "Toss and serve."],
    "used_ingredients": ["spaghetti", "garlic", "olive oil"],
    "missing_ingredients": ["parsley"],
    "substitutions": [
        {"missing": "parsley", "replace_with": "chili flakes", "note": "Adds the same lift."}
    ],
}

ADAPT_PAYLOAD = {
    "recipes": [
        {
            "id": 1001,
            "title": "Spaghetti Aglio e Olio",
            "ingredients": ["spaghetti", "garlic", "olive oil", "parsley"],
            "steps": ["Boil pasta.", "Fry garlic.", "Combine."],
        },
        {
            "id": 1002,
            "title": "Cacio e Pepe",
            "ingredients": ["spaghetti", "pecorino", "black pepper"],
            "steps": ["Boil pasta.", "Make cheese cream.", "Toss."],
        },
    ],
    "inventory": ["spaghetti", "garlic", "olive oil", "chili flakes"],
    "prefs": {"allergies": ["peanut"], "dislikes": [], "servings": 2},
}


class FakeProvider:
    """Replays a scripted sequence of model answers.

    Each item is a dict (returned as the model body), a raw string (parsed like real output,
    so malformed text exercises the repair path) or an exception to raise.
    """

    name = "fake"
    model = "fake-model"

    def __init__(self, script: list[Any], *, delay_s: float = 0.0, models: list[str] | None = None):
        self.script = list(script)
        self.delay_s = delay_s
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self._models = models if models is not None else ["fake-model"]

    async def complete_json(
        self, system: str, user: str, schema: dict[str, Any], timeout_s: float
    ) -> tuple[dict[str, Any], LLMMeta]:
        self.calls.append(
            {"system": system, "user": user, "schema": schema, "timeout_s": timeout_s}
        )
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        meta = LLMMeta(
            provider=self.name,
            model=self.model,
            latency_ms=7,
            prompt_tokens=100,
            completion_tokens=50,
        )
        item = self.script.pop(0) if self.script else {}
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            try:
                data = json.loads(item)
            except json.JSONDecodeError as exc:
                raise MalformedModelJSON(f"not json: {exc}", raw=item, meta=meta) from exc
            return data, meta
        return item, meta

    async def list_models(self) -> list[str]:
        return self._models

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="lmstudio",
        llm_model="fake-model",
        llm_timeout_sec=5,
        llm_run_budget_sec=5,
        llm_max_concurrency=1,
        llm_queue_wait_sec=0.2,
        llm_max_tokens=256,
    )


@pytest.fixture
def metrics() -> Metrics:
    return Metrics()


@pytest.fixture
def gate(settings: Settings, metrics: Metrics) -> GpuGate:
    return GpuGate(
        max_concurrency=settings.llm_max_concurrency,
        queue_wait_sec=settings.llm_queue_wait_sec,
        metrics=metrics,
    )


@pytest.fixture
def make_app(settings: Settings, metrics: Metrics, gate: GpuGate):
    def _make(provider: FakeProvider) -> FastAPI:
        app = create_app()
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_provider] = lambda: provider
        app.dependency_overrides[get_gate] = lambda: gate
        app.dependency_overrides[get_metrics] = lambda: metrics
        return app

    return _make


@pytest.fixture
def make_client(make_app):
    def _make(provider: FakeProvider) -> TestClient:
        # No context manager on purpose: the lifespan would build a real provider.
        return TestClient(make_app(provider))

    return _make


@pytest.fixture
def make_async_client(make_app):
    """Runs the app in the test's own event loop, so concurrent requests share the gate."""

    @asynccontextmanager
    async def _make(provider: FakeProvider) -> AsyncIterator[AsyncClient]:
        transport = ASGITransport(app=make_app(provider))
        async with AsyncClient(transport=transport, base_url="http://llm-service") as client:
            yield client

    return _make


@pytest.fixture
def client(make_client) -> Iterator[TestClient]:
    yield make_client(FakeProvider([ADAPTED]))
