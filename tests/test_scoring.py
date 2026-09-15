from app.domain.schemas import AdaptedRecipe
from tools.eval.scoring import (
    count_allergen_violations,
    count_out_of_inventory,
    same_food,
)


def recipe(**overrides) -> AdaptedRecipe:
    base = {
        "base_recipe_id": 1,
        "title": "Test",
        "steps": ["Cook it."],
        "used_ingredients": ["spaghetti", "garlic"],
        "missing_ingredients": [],
        "substitutions": [],
    }
    return AdaptedRecipe.model_validate(base | overrides)


def test_same_food_ignores_plurals_and_qualifiers():
    assert same_food("tomatoes", "2 large ripe tomato")
    assert same_food("olive oil", "extra virgin olive oil")
    assert not same_food("olive oil", "sesame oil")
    assert not same_food("chicken thighs", "tofu")


def test_allergen_violation_covers_aliases():
    hits = count_allergen_violations(recipe(used_ingredients=["spaghetti", "butter"]), ["milk"])
    assert hits == ["butter"]


def test_allergen_violation_catches_substitution_suggestions():
    hits = count_allergen_violations(
        recipe(
            substitutions=[
                {"missing": "cream", "replace_with": "cashew cream", "note": "dairy free"}
            ]
        ),
        ["tree nuts"],
    )
    assert hits == ["cashew"]


def test_no_allergies_means_no_violations():
    assert count_allergen_violations(recipe(used_ingredients=["butter"]), []) == []


def test_out_of_inventory_flags_ingredients_not_declared_missing():
    result = count_out_of_inventory(
        recipe(used_ingredients=["spaghetti", "parsley", "garlic"]),
        ["spaghetti", "garlic"],
    )
    assert result == ["parsley"]


def test_declared_missing_ingredients_are_not_counted():
    result = count_out_of_inventory(
        recipe(used_ingredients=["spaghetti", "parsley"], missing_ingredients=["parsley"]),
        ["spaghetti"],
    )
    assert result == []
