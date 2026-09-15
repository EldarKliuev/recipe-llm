from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# core-api -> llm-service request payloads
# ---------------------------------------------------------------------------


class RetrievedRecipe(BaseModel):
    """One of the 3-5 recipes core-api already retrieved by SQL set coverage."""

    model_config = ConfigDict(extra="ignore")

    id: int
    title: str
    ingredients: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)


class Prefs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Hints for the model only. core-api owns the hard allergen deny-list.
    allergies: list[str] = Field(default_factory=list)
    diet: str | None = None
    dislikes: list[str] = Field(default_factory=list)
    servings: int | None = Field(default=None, ge=1, le=20)


class AdaptRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    recipes: list[RetrievedRecipe] = Field(min_length=1, max_length=5)
    inventory: list[str] = Field(min_length=1)
    prefs: Prefs = Field(default_factory=Prefs)


class ParseIngredientsRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lines: list[str] = Field(min_length=1, max_length=50)


# ---------------------------------------------------------------------------
# Model output schemas (these drive response_format json_schema)
# ---------------------------------------------------------------------------


class Substitution(BaseModel):
    missing: str
    replace_with: str
    note: str


class AdaptedRecipe(BaseModel):
    """Structured model output. `base_recipe_id` lets core-api prove the model picked
    one of the supplied recipes instead of inventing its own."""

    base_recipe_id: int
    title: str
    steps: list[str]
    used_ingredients: list[str]
    missing_ingredients: list[str]
    substitutions: list[Substitution]


class ParsedIngredient(BaseModel):
    raw: str
    name: str
    quantity: float | None
    unit: str | None
    preparation: str | None


class ParsedIngredients(BaseModel):
    items: list[ParsedIngredient]


# ---------------------------------------------------------------------------
# llm-service -> core-api response payloads
# ---------------------------------------------------------------------------


class RunMeta(BaseModel):
    """Per-run metadata; core-api stores it in generation_runs."""

    provider: str
    model: str
    prompt_version: str
    latency_ms: int
    queue_wait_ms: int
    retries: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class AdaptResponse(BaseModel):
    recipe: AdaptedRecipe
    meta: RunMeta


class ParseIngredientsResponse(BaseModel):
    items: list[ParsedIngredient]
    meta: RunMeta


# ---------------------------------------------------------------------------
# JSON Schema for constrained decoding
# ---------------------------------------------------------------------------


def _tighten(node: Any) -> None:
    """Make every object closed and every property required.

    OpenAI-compatible strict mode rejects optional properties and `default`;
    LM Studio's constrained decoding is happier with closed objects too.
    """
    if isinstance(node, list):
        for item in node:
            _tighten(item)
        return
    if not isinstance(node, dict):
        return

    node.pop("default", None)
    if node.get("type") == "object" or "properties" in node:
        properties = node.get("properties", {})
        node["additionalProperties"] = False
        node["required"] = list(properties)
    for value in node.values():
        _tighten(value)


def strict_json_schema(model: type[BaseModel], name: str) -> dict[str, Any]:
    """Build a `response_format={"type": "json_schema", ...}` payload from a model."""
    schema = model.model_json_schema()
    _tighten(schema)
    return {"name": name, "strict": True, "schema": schema}
