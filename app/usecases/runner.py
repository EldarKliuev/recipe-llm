import json
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from app.domain.prompts import PROMPT_VERSION, build_repair_user
from app.domain.schemas import RunMeta, strict_json_schema
from app.errors import ModelOutputInvalidError
from app.providers.base import LLMMeta, LLMProvider, MalformedModelJSON
from app.usecases.gate import GpuGate, Slot

log = logging.getLogger(__name__)

# One straight attempt plus at most this many repair attempts.
MAX_REPAIR_ATTEMPTS = 2


async def run_structured[T: BaseModel](
    *,
    provider: LLMProvider,
    gate: GpuGate,
    output_model: type[T],
    schema_name: str,
    system: str,
    user: str,
    timeout_s: float,
    check: Callable[[T], str | None] | None = None,
) -> tuple[T, RunMeta]:
    """Ask the model for JSON matching `output_model`, repairing invalid answers.

    `check` guards semantics we can cheaply verify here (for instance that the model kept to
    the supplied recipe ids). Returning a message from it triggers the same repair loop as a
    schema violation. Everything else — allergens, real inventory — is core-api's job.
    """
    async with gate.slot() as slot:
        return await _attempt_loop(
            provider=provider,
            slot=slot,
            output_model=output_model,
            schema_name=schema_name,
            system=system,
            user=user,
            timeout_s=timeout_s,
            check=check,
        )


async def _attempt_loop[T: BaseModel](
    *,
    provider: LLMProvider,
    slot: Slot,
    output_model: type[T],
    schema_name: str,
    system: str,
    user: str,
    timeout_s: float,
    check: Callable[[T], str | None] | None,
) -> tuple[T, RunMeta]:
    schema = strict_json_schema(output_model, schema_name)
    current_user = user
    total_latency_ms = 0
    prompt_tokens = 0
    completion_tokens = 0
    last_error = "unknown"
    last_raw = ""

    for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        slot.retries = attempt
        try:
            data, meta = await provider.complete_json(
                system=system, user=current_user, schema=schema, timeout_s=timeout_s
            )
        except MalformedModelJSON as exc:
            total_latency_ms, prompt_tokens, completion_tokens = _accumulate(
                exc.meta, total_latency_ms, prompt_tokens, completion_tokens
            )
            last_error, last_raw = exc.message, exc.raw
            log.warning(
                "model output unparsable",
                extra={"attempt": attempt, "reason": exc.message},
            )
            current_user = build_repair_user(user, last_raw, last_error)
            continue

        total_latency_ms, prompt_tokens, completion_tokens = _accumulate(
            meta, total_latency_ms, prompt_tokens, completion_tokens
        )

        try:
            value = output_model.model_validate(data)
        except ValidationError as exc:
            last_error = _format_validation_error(exc)
            last_raw = _dump(data)
            log.warning(
                "model output failed schema validation",
                extra={"attempt": attempt, "reason": last_error},
            )
            current_user = build_repair_user(user, last_raw, last_error)
            continue

        problem = check(value) if check else None
        if problem:
            last_error = problem
            last_raw = _dump(data)
            log.warning(
                "model output failed semantic check",
                extra={"attempt": attempt, "reason": problem},
            )
            current_user = build_repair_user(user, last_raw, last_error)
            continue

        return value, RunMeta(
            provider=provider.name,
            model=meta.model,
            prompt_version=PROMPT_VERSION,
            latency_ms=total_latency_ms,
            queue_wait_ms=slot.queue_wait_ms,
            retries=attempt,
            prompt_tokens=prompt_tokens or None,
            completion_tokens=completion_tokens or None,
        )

    raise ModelOutputInvalidError(
        f"model output still invalid after {MAX_REPAIR_ATTEMPTS} repair attempts",
        detail={"reason": last_error, "raw_excerpt": last_raw[:500]},
    )


def _accumulate(
    meta: LLMMeta, latency_ms: int, prompt_tokens: int, completion_tokens: int
) -> tuple[int, int, int]:
    return (
        latency_ms + meta.latency_ms,
        prompt_tokens + (meta.prompt_tokens or 0),
        completion_tokens + (meta.completion_tokens or 0),
    )


def _format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
        for error in exc.errors()[:5]
    )


def _dump(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)
