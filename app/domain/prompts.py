import json

from app.domain.schemas import AdaptRequest, ParseIngredientsRequest, RetrievedRecipe

# Bump on every prompt change: the version travels in RunMeta so eval runs stay comparable.
PROMPT_VERSION = "v1"

ADAPT_SYSTEM = """You adapt an existing recipe to what a cook already has at home.

You are given a small set of candidate recipes that were already retrieved for this cook, \
plus the cook's fridge inventory and preferences. Your job is selection and adaptation, \
never invention.

Rules:
1. Pick exactly one candidate recipe: the one best covered by the inventory. Return its id \
in `base_recipe_id`. Never return an id that is not in the candidate list, and never merge \
several candidates into one.
2. Rewrite the steps in your own words. Keep them practical and ordered. Do not copy the \
source wording sentence by sentence.
3. Every ingredient your steps rely on must appear in `used_ingredients` (present in the \
inventory) or in `missing_ingredients` (not in the inventory). Never use something outside \
the inventory without listing it in `missing_ingredients`.
4. Prefer swapping a missing ingredient for something in the inventory. Record each swap in \
`substitutions` with a one-sentence `note` on why it works.
5. Treat the listed allergies and dislikes as hard avoidance: do not use those ingredients \
and do not suggest them as substitutions.
6. Write in English. Do not add commentary outside the JSON object.
"""

PARSE_SYSTEM = """You normalise free-form recipe ingredient lines into structured records.

For each input line return one item, in the same order, with:
- `raw`: the input line, unchanged.
- `name`: the canonical singular food name, lowercase, no quantity, no unit, no preparation \
(for example "2 large ripe tomatoes, diced" -> "tomato").
- `quantity`: the numeric amount as a decimal number, or null when the line has none. \
Convert fractions and ranges ("1/2" -> 0.5, "2-3" -> 2.5).
- `unit`: the singular unit as written ("cup", "tbsp", "g", "clove"), or null when absent.
- `preparation`: how it is prepared or qualified ("diced", "finely chopped"), or null.

Guess nothing that is not in the line. Do not drop, merge or reorder lines. \
Answer in English only.
"""


def _render_recipe(recipe: RetrievedRecipe) -> dict[str, object]:
    return {
        "id": recipe.id,
        "title": recipe.title,
        "ingredients": recipe.ingredients,
        "steps": recipe.steps,
    }


def build_adapt_user(request: AdaptRequest) -> str:
    prefs = request.prefs
    payload = {
        "inventory": request.inventory,
        "allergies": prefs.allergies,
        "dislikes": prefs.dislikes,
        "diet": prefs.diet,
        "servings": prefs.servings,
        "candidate_recipes": [_render_recipe(recipe) for recipe in request.recipes],
    }
    allowed_ids = [recipe.id for recipe in request.recipes]
    return (
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"Choose one recipe from candidate_recipes (allowed base_recipe_id values: "
        f"{allowed_ids}) and adapt it to the inventory above."
    )


def build_parse_user(request: ParseIngredientsRequest) -> str:
    lines = "\n".join(f"{index}. {line}" for index, line in enumerate(request.lines, start=1))
    return (
        f"Normalise these {len(request.lines)} ingredient lines. "
        f"Return exactly {len(request.lines)} items, in the same order.\n\n{lines}"
    )


def build_repair_user(previous_user: str, raw_output: str, error: str) -> str:
    """Second-chance prompt: the original request plus what was wrong with the last answer."""
    return (
        f"{previous_user}\n\n"
        "Your previous answer was rejected because it did not match the required JSON schema.\n"
        f"Previous answer:\n{raw_output[:2000]}\n\n"
        f"Validation error:\n{error}\n\n"
        "Return one corrected JSON object that satisfies the schema. Change nothing else."
    )
