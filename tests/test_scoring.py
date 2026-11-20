from app.domain.schemas import AdaptedRecipe, RetrievedRecipe
from tools.eval.scoring import (
    count_allergen_violations,
    count_copied_steps,
    count_off_language_steps,
    count_out_of_inventory,
    same_food,
)


def recipe(**overrides) -> AdaptedRecipe:
    base = {
        "base_recipe_id": 1,
        "title": "Test",
        "steps": ["Cook it."],
        "step_minutes": [None],
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


def test_gluten_free_staples_do_not_count_as_wheat():
    hits = count_allergen_violations(
        recipe(
            used_ingredients=["rice noodles", "shrimp"],
            steps=["Soak the rice noodles.", "Dust with rice flour and gluten-free bread crumbs."],
        ),
        ["wheat"],
    )
    assert hits == []


def test_plain_noodles_next_to_rice_noodles_still_count_as_wheat():
    hits = count_allergen_violations(
        recipe(used_ingredients=["rice noodles"], steps=["Add the egg noodles."]), ["wheat"]
    )
    assert hits == ["noodle"]


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


def source(steps: list[str], recipe_id: int = 1) -> RetrievedRecipe:
    return RetrievedRecipe(id=recipe_id, title="Source", steps=steps)


def test_copied_steps_match_after_case_punctuation_and_spacing():
    result = count_copied_steps(
        recipe(steps=["boil  the PASTA", "Warm garlic gently in the oil.", "Serve."]),
        [source(["Boil the pasta!", "Fry garlic."]), source(["Serve."], recipe_id=2)],
    )
    assert result == ["boil  the PASTA", "Serve."]


def test_rewritten_steps_are_not_counted_as_copies():
    result = count_copied_steps(
        recipe(steps=["Cook the pasta until tender.", "Fry the garlic in oil."]),
        [source(["Boil the pasta.", "Fry garlic."])],
    )
    assert result == []


def test_punctuation_only_steps_never_count_as_copies():
    # Both sides normalise to an empty string; that is not copied dataset text.
    assert count_copied_steps(recipe(steps=["..."]), [source(["-"])]) == []


def test_off_language_steps_are_english_steps_in_a_russian_answer():
    answer = recipe(steps=["Отварите пасту 10 минут.", "Toss with garlic.", "Подавайте."])
    assert count_off_language_steps(answer, "ru") == ["Toss with garlic."]


def test_off_language_steps_are_russian_steps_in_an_english_answer():
    answer = recipe(steps=["Boil the pasta.", "Подавайте."])
    assert count_off_language_steps(answer, "en") == ["Подавайте."]
