from fastapi import APIRouter

from app.deps import GateDep, ProviderDep, SettingsDep
from app.domain.schemas import ParseIngredientsRequest, ParseIngredientsResponse
from app.usecases.parse import parse_ingredients

router = APIRouter(prefix="/v1", tags=["parse"])


@router.post(
    "/parse-ingredients",
    response_model=ParseIngredientsResponse,
    summary="Normalise free-form ingredient lines into canonical records",
)
async def parse(
    payload: ParseIngredientsRequest,
    provider: ProviderDep,
    gate: GateDep,
    settings: SettingsDep,
) -> ParseIngredientsResponse:
    return await parse_ingredients(
        payload,
        provider=provider,
        gate=gate,
        timeout_s=settings.llm_timeout_sec,
        budget_s=settings.llm_run_budget_sec,
    )
