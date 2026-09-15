from fastapi import APIRouter

from app.deps import GateDep, ProviderDep, SettingsDep
from app.domain.schemas import AdaptRequest, AdaptResponse
from app.usecases.adapt import adapt_recipe

router = APIRouter(prefix="/v1", tags=["adapt"])


@router.post(
    "/adapt",
    response_model=AdaptResponse,
    summary="Adapt one of the retrieved recipes to the fridge inventory",
)
async def adapt(
    payload: AdaptRequest,
    provider: ProviderDep,
    gate: GateDep,
    settings: SettingsDep,
) -> AdaptResponse:
    return await adapt_recipe(
        payload,
        provider=provider,
        gate=gate,
        timeout_s=settings.llm_timeout_sec,
    )
