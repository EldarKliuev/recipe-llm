from fastapi import APIRouter, Request

from app.deps import GateDep, ProviderDep, SettingsDep
from app.domain.schemas import ParseIngredientsRequest, ParseIngredientsResponse
from app.routes.disconnect import cancel_on_disconnect
from app.usecases.parse import parse_ingredients

router = APIRouter(prefix="/v1", tags=["parse"])


@router.post(
    "/parse-ingredients",
    response_model=ParseIngredientsResponse,
    summary="Normalise free-form ingredient lines into canonical records",
)
async def parse(
    payload: ParseIngredientsRequest,
    request: Request,
    provider: ProviderDep,
    gate: GateDep,
    settings: SettingsDep,
) -> ParseIngredientsResponse:
    return await cancel_on_disconnect(
        request,
        parse_ingredients(
            payload,
            provider=provider,
            gate=gate,
            timeout_s=settings.llm_timeout_sec,
            budget_s=settings.llm_run_budget_sec,
        ),
    )
