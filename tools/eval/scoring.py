"""Scoring helpers for the eval harness.

Deliberately crude string matching: this measures prompt and model regressions between
runs, it is not the validation layer. Real inventory and allergen enforcement lives in
core-api against canonical ingredient ids.
"""

import re

from app.domain.schemas import AdaptedRecipe, RetrievedRecipe

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

# Phrases that name a food free of the allergen although they contain one of its aliases.
# They are cut out of the text before matching, so "rice noodles" is not wheat while
# "egg noodles" next to it still is. Written singular, as the matcher sees words.
ALLERGEN_FREE_PHRASES: dict[str, tuple[str, ...]] = {
    "wheat": (
        "rice noodle",
        "rice vermicelli",
        "rice flour",
        "rice paper",
        "glass noodle",
        "corn tortilla",
        "corn flour",
    ),
}
_GLUTEN_FREE = re.compile(r"\bgluten free \w+")


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
    free_phrases = allergen_free_phrases(allergies)
    hits = {term for surface in surfaces for term in terms if term in _words(surface, free_phrases)}
    return sorted(hits)


def allergen_free_phrases(allergies: list[str]) -> tuple[str, ...]:
    return tuple(
        phrase
        for allergy in allergies
        for phrase in ALLERGEN_FREE_PHRASES.get(normalise(allergy), ())
    )


def _words(surface: str, free_phrases: tuple[str, ...]) -> set[str]:
    text = " ".join(singular(word) for word in normalise(surface).split())
    if free_phrases:
        text = _GLUTEN_FREE.sub(" ", text)
        for phrase in free_phrases:
            text = re.sub(rf"\b{phrase}\b", " ", text)
    return set(text.split())


def count_out_of_inventory(recipe: AdaptedRecipe, inventory: list[str]) -> list[str]:
    """Ingredients claimed as used that are neither in the fridge nor declared missing."""
    return sorted(
        ingredient
        for ingredient in recipe.used_ingredients
        if not in_list(ingredient, inventory)
        and not in_list(ingredient, recipe.missing_ingredients)
    )


def count_copied_steps(recipe: AdaptedRecipe, sources: list[RetrievedRecipe]) -> list[str]:
    """Output steps that repeat a supplied step verbatim, ignoring case, punctuation and spacing.

    Steps are public text and must be retold, not copied from the dataset. Every supplied
    recipe counts as a source, whichever one the model based its answer on. A light
    paraphrase slips through: this counts exact copies only.
    """
    originals = {_step_key(step) for source in sources for step in source.steps} - {""}
    return [step for step in recipe.steps if _step_key(step) in originals]


def _step_key(step: str) -> str:
    return " ".join(normalise(step).split())


_CYRILLIC = re.compile("[\u0400-\u04ff]")


def count_off_language_steps(recipe: AdaptedRecipe, language: str) -> list[str]:
    """Steps not written in the requested language: without Cyrillic for Russian, with it
    for English. Mixed text passes as Russian, so this catches whole untranslated steps."""
    if language == "ru":
        return [step for step in recipe.steps if not _CYRILLIC.search(step)]
    return [step for step in recipe.steps if _CYRILLIC.search(step)]
