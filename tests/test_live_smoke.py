"""Smoke test against a real LM Studio server.

Excluded from the default run (see addopts in pyproject). Run it by hand after changing the
model, the prompt or the provider:

    LLM_MODEL=<loaded model> pytest -m live
"""

import pytest

from app.config import get_settings
from app.domain.schemas import AdaptRequest
from app.metrics import Metrics
from app.providers import create_provider
from app.usecases.adapt import adapt_recipe
from app.usecases.gate import GpuGate
from tests.conftest import ADAPT_PAYLOAD

pytestmark = pytest.mark.live


async def test_live_adapt_against_lm_studio():
    settings = get_settings()
    provider = create_provider(settings)
    gate = GpuGate(
        max_concurrency=settings.llm_max_concurrency,
        queue_wait_sec=settings.llm_queue_wait_sec,
        metrics=Metrics(),
    )
    try:
        models = await provider.list_models()
        assert settings.llm_model in models, f"{settings.llm_model} is not loaded: {models}"

        response = await adapt_recipe(
            AdaptRequest.model_validate(ADAPT_PAYLOAD),
            provider=provider,
            gate=gate,
            timeout_s=settings.llm_timeout_sec,
            budget_s=settings.llm_run_budget_sec,
        )
    finally:
        await provider.aclose()

    assert response.recipe.base_recipe_id in {1001, 1002}
    assert response.recipe.steps
    assert response.meta.latency_ms > 0
