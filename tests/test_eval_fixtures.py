import csv
import json

import pytest

import tools.eval.__main__ as eval_main
from app.domain.prompts import PROMPT_VERSION
from app.domain.schemas import AdaptRequest
from tests.conftest import ADAPT_PAYLOAD, ADAPTED, FakeProvider
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


@pytest.fixture
def eval_env(tmp_path, monkeypatch, settings):
    """Points main() at a fake provider and a temporary results directory."""
    results_dir = tmp_path / "results"
    monkeypatch.setattr(eval_main, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(eval_main, "get_settings", lambda: settings)

    def _use(provider: FakeProvider) -> None:
        monkeypatch.setattr(eval_main, "create_provider", lambda _settings: provider)

    return tmp_path, results_dir, _use


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


async def test_eval_run_writes_comparable_csvs_and_counts_refusals(eval_env):
    tmp_path, results_dir, use_provider = eval_env
    fixtures = tmp_path / "cases.json"
    fixtures.write_text(
        json.dumps([{"id": "good", **ADAPT_PAYLOAD}, {"id": "broken", **ADAPT_PAYLOAD}])
    )
    # First case answers at once; second never produces JSON, so repair is exhausted.
    provider = FakeProvider([ADAPTED, "not json", "not json", "not json"])
    use_provider(provider)

    assert await eval_main.main(["--fixtures", str(fixtures)]) == 0

    assert len(provider.calls) == 4
    assert provider.closed

    [cases_path] = results_dir.glob("*_cases.csv")
    rows = _read_csv(cases_path)
    assert [row["case_id"] for row in rows] == ["good", "broken"]
    assert all(row["model"] == "fake-model" for row in rows)
    assert all(row["prompt_version"] == PROMPT_VERSION for row in rows)
    good, broken = rows
    assert good["status"] == "ok"
    assert good["base_recipe_id"] == "1001"
    assert good["valid_first_attempt"] == "1"
    assert broken["status"] == "invalid_model_output"
    assert broken["valid_first_attempt"] == "0"

    [summary] = _read_csv(results_dir / "summary.csv")
    assert summary["model"] == "fake-model"
    assert summary["prompt_version"] == PROMPT_VERSION
    assert summary["cases"] == "2"
    assert summary["refusals"] == "1"
    assert summary["refusal_rate"] == "0.5"
    assert summary["first_attempt_json_rate"] == "0.5"


async def test_eval_refuses_an_empty_case_set_without_touching_the_model(
    eval_env, monkeypatch, capsys
):
    tmp_path, results_dir, _ = eval_env
    fixtures = tmp_path / "cases.json"
    fixtures.write_text("[]")

    def no_provider(_settings):
        raise AssertionError("an empty run must not build a provider")

    monkeypatch.setattr(eval_main, "create_provider", no_provider)

    assert await eval_main.main(["--fixtures", str(fixtures)]) == 2

    assert "no cases" in capsys.readouterr().err

    assert not results_dir.exists()
