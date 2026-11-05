import logging

from app.domain.prompts import PARSE_SYSTEM, build_parse_user
from app.domain.schemas import (
    ParsedIngredients,
    ParseIngredientsRequest,
    ParseIngredientsResponse,
)
from app.providers.base import LLMProvider
from app.usecases.gate import GpuGate
from app.usecases.runner import run_structured

log = logging.getLogger(__name__)


async def parse_ingredients(
    request: ParseIngredientsRequest,
    *,
    provider: LLMProvider,
    gate: GpuGate,
    timeout_s: float,
    budget_s: float,
) -> ParseIngredientsResponse:
    """Long tail of ingestion: lines that rules and CRF in core-api could not resolve."""
    expected = len(request.lines)

    def check(parsed: ParsedIngredients) -> str | None:
        if len(parsed.items) != expected:
            return f"expected {expected} items, got {len(parsed.items)}"
        for index, (item, line) in enumerate(zip(parsed.items, request.lines, strict=True)):
            if item.raw.strip() != line.strip():
                return f"item {index} `raw` must repeat the input line verbatim"
            if not item.name.strip():
                return f"item {index} has an empty `name`"
        return None

    parsed, meta = await run_structured(
        provider=provider,
        gate=gate,
        output_model=ParsedIngredients,
        schema_name="parsed_ingredients",
        system=PARSE_SYSTEM,
        user=build_parse_user(request),
        timeout_s=timeout_s,
        budget_s=budget_s,
        check=check,
    )
    log.info(
        "ingredient lines parsed",
        extra={"lines": expected, "retries": meta.retries, "latency_ms": meta.latency_ms},
    )
    return ParseIngredientsResponse(items=parsed.items, meta=meta)
