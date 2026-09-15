import json

from app.domain.schemas import AdaptRequest
from tools.eval.__main__ import FIXTURES, CaseResult, summarise

CASES = json.loads(FIXTURES.read_text())


def test_thirty_fixtures_exist_and_have_unique_ids():
    assert len(CASES) == 30
    assert len({case["id"] for case in CASES}) == 30


def test_every_fixture_is_a_valid_adapt_request():
    for case in CASES:
        request = AdaptRequest.model_validate(case)
        assert 1 <= len(request.recipes) <= 5
        assert request.inventory
        for recipe in request.recipes:
            assert recipe.ingredients and recipe.steps


def test_recipe_ids_are_unique_across_fixtures():
    ids = [recipe["id"] for case in CASES for recipe in case["recipes"]]
    assert len(ids) == len(set(ids))


def test_summary_reports_rates_over_all_cases_including_refusals():
    results = [
        CaseResult(case_id="a", status="ok", retries=0, latency_ms=100, valid_first_attempt=1),
        CaseResult(
            case_id="b",
            status="ok",
            retries=1,
            latency_ms=300,
            allergen_violations=["butter"],
            out_of_inventory=["parsley"],
        ),
        CaseResult(case_id="c", status="invalid_model_output", error="gave up"),
    ]

    summary = summarise(results, provider_name="lmstudio", model="m")

    assert summary["cases"] == 3
    assert summary["refusals"] == 1
    assert summary["refusal_rate"] == round(1 / 3, 4)
    assert summary["first_attempt_json_rate"] == round(1 / 3, 4)
    assert summary["allergen_violations"] == 1
    assert summary["out_of_inventory_total"] == 1
    assert summary["latency_ms_median"] == 200
