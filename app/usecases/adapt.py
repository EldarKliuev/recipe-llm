import logging

from app.domain.prompts import ADAPT_SYSTEM, build_adapt_user
from app.domain.schemas import AdaptedRecipe, AdaptRequest, AdaptResponse
from app.providers.base import LLMProvider
from app.usecases.gate import GpuGate
from app.usecases.runner import run_structured

log = logging.getLogger(__name__)


async def adapt_recipe(
    request: AdaptRequest,
    *,
    provider: LLMProvider,
    gate: GpuGate,
    timeout_s: float,
) -> AdaptResponse:
    allowed_ids = {recipe.id for recipe in request.recipes}

    def check(recipe: AdaptedRecipe) -> str | None:
        if recipe.base_recipe_id not in allowed_ids:
            return (
                f"base_recipe_id {recipe.base_recipe_id} is not one of the candidate "
                f"recipes {sorted(allowed_ids)}"
            )
        if not recipe.steps:
            return "steps must not be empty"
        return None

    recipe, meta = await run_structured(
        provider=provider,
        gate=gate,
        output_model=AdaptedRecipe,
        schema_name="adapted_recipe",
        system=ADAPT_SYSTEM,
        user=build_adapt_user(request),
        timeout_s=timeout_s,
        check=check,
    )
    log.info(
        "recipe adapted",
        extra={
            "base_recipe_id": recipe.base_recipe_id,
            "candidates": len(request.recipes),
            "inventory_size": len(request.inventory),
            "missing_count": len(recipe.missing_ingredients),
            "retries": meta.retries,
            "latency_ms": meta.latency_ms,
        },
    )
    return AdaptResponse(recipe=recipe, meta=meta)
