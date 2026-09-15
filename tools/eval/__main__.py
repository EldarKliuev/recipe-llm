"""Eval harness: run the fixed fixture set through the configured model and write CSV.

    docker compose run --rm llm-service python -m tools.eval

Mandatory whenever LLM_MODEL or PROMPT_VERSION changes; without numbers, swapping a model
or a prompt is guesswork. Fixtures carry their own retrieval set, so Postgres is not needed.
"""

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.domain.prompts import PROMPT_VERSION
from app.domain.schemas import AdaptRequest
from app.errors import ServiceError
from app.metrics import Metrics, percentile
from app.providers import create_provider
from app.usecases.adapt import adapt_recipe
from app.usecases.gate import GpuGate
from tools.eval.scoring import count_allergen_violations, count_out_of_inventory

FIXTURES = Path(__file__).parent / "fixtures" / "cases.json"
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class CaseResult:
    case_id: str
    status: str
    base_recipe_id: int | None = None
    retries: int | None = None
    latency_ms: int | None = None
    valid_first_attempt: int = 0
    allergen_violations: list[str] = field(default_factory=list)
    out_of_inventory: list[str] = field(default_factory=list)
    missing_count: int | None = None
    steps_count: int | None = None
    error: str = ""


async def run_case(case: dict, *, provider, gate, timeout_s: float) -> CaseResult:
    request = AdaptRequest.model_validate(case)
    try:
        response = await adapt_recipe(request, provider=provider, gate=gate, timeout_s=timeout_s)
    except ServiceError as exc:
        return CaseResult(case_id=case["id"], status=exc.code, error=exc.message)

    recipe, meta = response.recipe, response.meta
    return CaseResult(
        case_id=case["id"],
        status="ok",
        base_recipe_id=recipe.base_recipe_id,
        retries=meta.retries,
        latency_ms=meta.latency_ms,
        valid_first_attempt=int(meta.retries == 0),
        allergen_violations=count_allergen_violations(recipe, request.prefs.allergies),
        out_of_inventory=count_out_of_inventory(recipe, request.inventory),
        missing_count=len(recipe.missing_ingredients),
        steps_count=len(recipe.steps),
    )


def summarise(results: list[CaseResult], *, provider_name: str, model: str) -> dict[str, object]:
    done = [result for result in results if result.status == "ok"]
    latencies = sorted(result.latency_ms for result in done if result.latency_ms is not None)
    refusals = len(results) - len(done)
    return {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": provider_name,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "cases": len(results),
        "refusals": refusals,
        "refusal_rate": _rate(refusals, len(results)),
        "first_attempt_json_rate": _rate(
            sum(result.valid_first_attempt for result in results), len(results)
        ),
        "allergen_violations": sum(len(result.allergen_violations) for result in results),
        "cases_with_allergen_violation": sum(1 for result in results if result.allergen_violations),
        "out_of_inventory_total": sum(len(result.out_of_inventory) for result in results),
        "cases_with_out_of_inventory": sum(1 for result in results if result.out_of_inventory),
        "latency_ms_median": int(percentile(latencies, 0.5)) if latencies else "",
        "latency_ms_p95": int(percentile(latencies, 0.95)) if latencies else "",
    }


def _rate(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def write_csv(results: list[CaseResult], summary: dict[str, object], stamp: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases_path = RESULTS_DIR / f"{stamp}_cases.csv"
    with cases_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["model", "prompt_version", *(field_name for field_name in asdict(results[0]))]
        )
        for result in results:
            row = asdict(result)
            row["allergen_violations"] = "|".join(result.allergen_violations)
            row["out_of_inventory"] = "|".join(result.out_of_inventory)
            writer.writerow([summary["model"], summary["prompt_version"], *row.values()])

    summary_path = RESULTS_DIR / "summary.csv"
    is_new = not summary_path.exists()
    with summary_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        if is_new:
            writer.writeheader()
        writer.writerow(summary)
    return cases_path


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tools.eval", description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    args = parser.parse_args(argv)

    cases = json.loads(args.fixtures.read_text())
    if args.limit:
        cases = cases[: args.limit]

    settings = get_settings()
    metrics = Metrics(window=len(cases) or 1)
    provider = create_provider(settings)
    gate = GpuGate(
        max_concurrency=settings.llm_max_concurrency,
        # Eval runs sequentially and must not give up on a queue that is never contended.
        queue_wait_sec=max(settings.llm_queue_wait_sec, settings.llm_timeout_sec),
        metrics=metrics,
    )

    print(
        f"eval: {len(cases)} cases, provider={settings.llm_provider}, "
        f"model={settings.llm_model}, prompt_version={PROMPT_VERSION}",
        flush=True,
    )
    results: list[CaseResult] = []
    try:
        for index, case in enumerate(cases, start=1):
            result = await run_case(
                case, provider=provider, gate=gate, timeout_s=settings.llm_timeout_sec
            )
            results.append(result)
            print(
                f"[{index}/{len(cases)}] {result.case_id}: {result.status} "
                f"retries={result.retries} latency_ms={result.latency_ms} "
                f"allergens={len(result.allergen_violations)} "
                f"out_of_inventory={len(result.out_of_inventory)}",
                flush=True,
            )
    finally:
        await provider.aclose()

    summary = summarise(results, provider_name=settings.llm_provider, model=settings.llm_model)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    cases_path = write_csv(results, summary, stamp)

    print("\nsummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nper-case CSV: {cases_path}")
    print(f"appended run to: {RESULTS_DIR / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
