"""Scoring helpers for the eval harness.

Deliberately crude string matching: this measures prompt and model regressions between
runs, it is not the validation layer. Real inventory and allergen enforcement lives in
core-api against canonical ingredient ids.
"""

import re

from app.domain.schemas import AdaptedRecipe

# Words that carry no food identity on their own and would create false matches.
GENERIC_WORDS = frozenset(
    {
        "oil",
        "sauce",
        "paste",
        "powder",
        "fresh",
        "dried",
        "ground",
        "large",
        "small",
        "medium",
        "cooked",
        "canned",
        "raw",
        "chopped",
        "sliced",
        "diced",
        "whole",
        "plain",
        "extra",
        "virgin",
        "flake",
        "leaf",
        "clove",
        "piece",
        "fillet",
        "and",
        "of",
        "to",
        "taste",
        "for",
        "the",
        "a",
        "some",
        "salt",
        "pepper",
        "water",
    }
)

ALLERGEN_ALIASES: dict[str, set[str]] = {
    "milk": {
        "milk",
        "butter",
        "cream",
        "cheese",
        "cheddar",
        "parmesan",
        "pecorino",
        "feta",
        "gruyere",
        "yogurt",
        "ghee",
        "mascarpone",
    },
    "egg": {"egg", "mayonnaise", "meringue", "omelette"},
    "wheat": {
        "wheat",
        "flour",
        "bread",
        "breadcrumb",
        "pasta",
        "spaghetti",
        "couscous",
        "barley",
        "ramen",
        "noodle",
        "tortilla",
        "cracker",
    },
    "peanut": {"peanut"},
    "tree nuts": {"almond", "walnut", "cashew", "pecan", "hazelnut", "pistachio", "nut"},
    "soy": {"soy", "soybean", "tofu", "miso", "edamame", "doubanjiang", "gochujang", "tempeh"},
    "sesame": {"sesame", "tahini"},
    "shellfish": {"shrimp", "prawn", "clam", "mussel", "crab", "lobster", "scallop", "oyster"},
    "fish": {"fish", "salmon", "tuna", "anchovy", "cod", "sardine", "mackerel"},
}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def singular(word: str) -> str:
    if len(word) <= 3 or not word.endswith("s") or word.endswith("ss"):
        return word
    if word.endswith("ies"):
        return f"{word[:-3]}y"
    if word.endswith(("oes", "ches", "shes", "xes", "sses")):
        return word[:-2]
    return word[:-1]


def food_tokens(text: str) -> set[str]:
    tokens = {singular(word) for word in normalise(text).split()}
    return tokens - GENERIC_WORDS


def same_food(left: str, right: str) -> bool:
    left_tokens, right_tokens = food_tokens(left), food_tokens(right)
    if not left_tokens or not right_tokens:
        return normalise(left) == normalise(right)
    return bool(left_tokens & right_tokens)


def in_list(needle: str, haystack: list[str]) -> bool:
    return any(same_food(needle, item) for item in haystack)


def allergen_terms(allergies: list[str]) -> set[str]:
    terms: set[str] = set()
    for allergy in allergies:
        key = normalise(allergy)
        terms |= ALLERGEN_ALIASES.get(key, {singular(key)})
    return terms


def count_allergen_violations(recipe: AdaptedRecipe, allergies: list[str]) -> list[str]:
    """Allergen terms the model would actually put in the pot."""
    terms = allergen_terms(allergies)
    if not terms:
        return []
    surfaces = [
        *recipe.used_ingredients,
        *(substitution.replace_with for substitution in recipe.substitutions),
        *recipe.steps,
    ]
    hits = {
        term
        for surface in surfaces
        for term in terms
        if term in {singular(word) for word in normalise(surface).split()}
    }
    return sorted(hits)


def count_out_of_inventory(recipe: AdaptedRecipe, inventory: list[str]) -> list[str]:
    """Ingredients claimed as used that are neither in the fridge nor declared missing."""
    return sorted(
        ingredient
        for ingredient in recipe.used_ingredients
        if not in_list(ingredient, inventory)
        and not in_list(ingredient, recipe.missing_ingredients)
    )
