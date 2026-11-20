import logging
import re

from app.domain.prompts import ADAPT_SYSTEM, build_adapt_user, language_reminder
from app.domain.schemas import AdaptedRecipe, AdaptRequest, AdaptResponse
from app.providers.base import LLMProvider
from app.usecases.gate import GpuGate
from app.usecases.runner import run_structured

log = logging.getLogger(__name__)

_CYRILLIC = re.compile("[\u0400-\u04ff]")


async def adapt_recipe(
    request: AdaptRequest,
    *,
    provider: LLMProvider,
    gate: GpuGate,
    timeout_s: float,
    budget_s: float,
) -> AdaptResponse:
    allowed_ids = {recipe.id for recipe in request.recipes}

    def check(recipe: AdaptedRecipe) -> str | None:
        if recipe.base_recipe_id not in allowed_ids:
            return (
                f"base_recipe_id {recipe.base_recipe_id} is not one of the candidate "
                f"recipes {sorted(allowed_ids)}"
            )
        # Same rule as core-api: it drops blank steps and rejects the answer only when
        # nothing is left, so a single blank step is not worth a repair attempt.
        if not any(step.strip() for step in recipe.steps):
            return "steps must not be empty"
        # core-api pairs minutes with steps by index; a shifted list puts a timer on
        # the wrong step. Value ranges are core-api's call, it drops the field instead.
        if len(recipe.step_minutes) != len(recipe.steps):
            return (
                f"step_minutes must have one entry per step: got {len(recipe.step_minutes)} "
                f"for {len(recipe.steps)} steps"
            )
        # core-api matches names against an English catalog and allergen rules, so a
        # translated name would be rejected there as an off-catalog product.
        names = [
            *recipe.used_ingredients,
            *recipe.missing_ingredients,
            *(name for sub in recipe.substitutions for name in (sub.missing, sub.replace_with)),
        ]
        translated = [name for name in names if _CYRILLIC.search(name)]
        if translated:
            # Without the reminder the repaired answer comes back entirely in English.
            return (
                f"ingredient names must stay in English as given in the inventory and "
                f"recipes, whatever the language: {translated}. "
                f"{language_reminder(request.language)}"
            )
        return None

    recipe, meta = await run_structured(
        provider=provider,
        gate=gate,
        output_model=AdaptedRecipe,
        schema_name="adapted_recipe",
        system=ADAPT_SYSTEM,
        user=build_adapt_user(request),
        timeout_s=timeout_s,
        budget_s=budget_s,
        check=check,
    )
    log.info(
        "recipe adapted",
        extra={
            "base_recipe_id": recipe.base_recipe_id,
            "candidates": len(request.recipes),
            "language": request.language,
            "inventory_size": len(request.inventory),
            "missing_count": len(recipe.missing_ingredients),
            "retries": meta.retries,
            "latency_ms": meta.latency_ms,
        },
    )
    return AdaptResponse(recipe=recipe, meta=meta)
