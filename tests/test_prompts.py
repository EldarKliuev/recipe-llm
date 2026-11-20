from app.domain.prompts import (
    ADAPT_SYSTEM,
    build_adapt_user,
    build_parse_user,
    build_repair_user,
)
from app.domain.schemas import AdaptRequest, ParseIngredientsRequest
from tests.conftest import ADAPT_PAYLOAD


def test_adapt_prompt_carries_candidates_inventory_and_allowed_ids():
    prompt = build_adapt_user(AdaptRequest.model_validate(ADAPT_PAYLOAD))

    assert "spaghetti" in prompt
    assert "Cacio e Pepe" in prompt
    assert "peanut" in prompt
    assert "[1001, 1002]" in prompt


def test_adapt_system_prompt_forbids_invention_and_verbatim_steps():
    assert "never invention" in ADAPT_SYSTEM
    assert "your own words" in ADAPT_SYSTEM
    assert "missing_ingredients" in ADAPT_SYSTEM


def test_adapt_system_prompt_asks_for_minutes_only_where_timing_matters():
    assert "step_minutes" in ADAPT_SYSTEM
    assert "null" in ADAPT_SYSTEM
    assert "same order" in ADAPT_SYSTEM


def test_parse_prompt_states_the_expected_item_count():
    prompt = build_parse_user(ParseIngredientsRequest(lines=["2 cups flour", "1 tsp salt"]))

    assert "Return exactly 2 items" in prompt
    assert "1. 2 cups flour" in prompt
    assert "2. 1 tsp salt" in prompt


def test_repair_prompt_keeps_the_task_and_adds_the_error():
    repair = build_repair_user("original task", '{"broken":', "base_recipe_id: field required")

    assert "original task" in repair
    assert "base_recipe_id: field required" in repair
    assert '{"broken":' in repair


def test_adapt_prompt_names_the_answer_language():
    english = build_adapt_user(AdaptRequest.model_validate(ADAPT_PAYLOAD))
    russian = build_adapt_user(AdaptRequest.model_validate(ADAPT_PAYLOAD | {"language": "ru"}))

    assert '"language": "English"' in english
    assert '"language": "Russian"' in russian


def test_adapt_system_prompt_keeps_ingredient_names_english_in_any_language():
    # core-api checks names against an English catalog and allergen rules.
    assert "`language`" in ADAPT_SYSTEM
    assert "stay in English" in ADAPT_SYSTEM
    assert "Write in English" not in ADAPT_SYSTEM
