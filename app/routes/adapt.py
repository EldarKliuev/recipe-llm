from fastapi import APIRouter, Request

from app.deps import GateDep, ProviderDep, SettingsDep
from app.domain.schemas import AdaptRequest, AdaptResponse
from app.routes.disconnect import cancel_on_disconnect
from app.usecases.adapt import adapt_recipe

router = APIRouter(prefix="/v1", tags=["adapt"])


@router.post(
    "/adapt",
    response_model=AdaptResponse,
    summary="Adapt one of the retrieved recipes to the fridge inventory",
)
async def adapt(
    payload: AdaptRequest,
    request: Request,
    provider: ProviderDep,
    gate: GateDep,
    settings: SettingsDep,
) -> AdaptResponse:
    return await cancel_on_disconnect(
        request,
        adapt_recipe(
            payload,
            provider=provider,
            gate=gate,
            timeout_s=settings.llm_timeout_sec,
            budget_s=settings.llm_run_budget_sec,
        ),
    )
