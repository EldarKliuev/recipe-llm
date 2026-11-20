import pytest
from pydantic import ValidationError

from app.domain.schemas import (
    AdaptedRecipe,
    AdaptRequest,
    ParsedIngredients,
    strict_json_schema,
)


def test_strict_schema_closes_objects_and_requires_every_field():
    payload = strict_json_schema(AdaptedRecipe, "adapted_recipe")

    assert payload["name"] == "adapted_recipe"
    assert payload["strict"] is True
    schema = payload["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "base_recipe_id",
        "title",
        "steps",
        "step_minutes",
        "used_ingredients",
        "missing_ingredients",
        "substitutions",
    }
    substitution = schema["$defs"]["Substitution"]
    assert substitution["additionalProperties"] is False
    assert set(substitution["required"]) == {"missing", "replace_with", "note"}


def test_step_minutes_allow_null_but_must_be_sent():
    # Strict mode cannot express "optional"; a step without a timer is null, not absent.
    schema = strict_json_schema(AdaptedRecipe, "adapted_recipe")["schema"]
    items = schema["properties"]["step_minutes"]["items"]
    assert {"type": "integer"} in items["anyOf"]
    assert {"type": "null"} in items["anyOf"]

    base = {
        "base_recipe_id": 1,
        "title": "x",
        "steps": ["a", "b"],
        "used_ingredients": [],
        "missing_ingredients": [],
        "substitutions": [],
    }
    recipe = AdaptedRecipe.model_validate(base | {"step_minutes": [5, None]})
    assert recipe.step_minutes == [5, None]
    with pytest.raises(ValidationError):
        AdaptedRecipe.model_validate(base)


def test_strict_schema_drops_defaults_that_strict_mode_rejects():
    schema = strict_json_schema(ParsedIngredients, "parsed_ingredients")["schema"]
    assert "default" not in schema["properties"]["items"]


def test_adapt_request_rejects_more_than_five_candidates():
    recipe = {"id": 1, "title": "x", "ingredients": ["a"], "steps": ["b"]}
    with pytest.raises(ValidationError):
        AdaptRequest(recipes=[recipe] * 6, inventory=["a"])


def test_adapt_request_needs_inventory():
    recipe = {"id": 1, "title": "x", "ingredients": ["a"], "steps": ["b"]}
    with pytest.raises(ValidationError):
        AdaptRequest(recipes=[recipe], inventory=[])


def test_adapt_request_defaults_prefs():
    recipe = {"id": 1, "title": "x", "ingredients": ["a"], "steps": ["b"]}
    request = AdaptRequest(recipes=[recipe], inventory=["a"])
    assert request.prefs.allergies == []
    assert request.prefs.servings is None
