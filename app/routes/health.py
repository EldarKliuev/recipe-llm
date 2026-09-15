import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.deps import MetricsDep, ProviderDep, SettingsDep
from app.domain.prompts import PROMPT_VERSION
from app.errors import ServiceError

log = logging.getLogger(__name__)

router = APIRouter(tags=["ops"])


@router.get("/health", summary="Liveness: the process answers")
async def health(settings: SettingsDep) -> dict[str, object]:
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "prompt_version": PROMPT_VERSION,
    }


@router.get("/ready", summary="Readiness: the configured model is loaded at the provider")
async def ready(provider: ProviderDep, settings: SettingsDep) -> JSONResponse:
    try:
        models = await provider.list_models()
    except ServiceError as exc:
        log.warning("readiness probe failed", extra={"reason": exc.message})
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "reason": exc.code,
                "message": exc.message,
                "model": settings.llm_model,
            },
        )

    if settings.llm_model not in models:
        return JSONResponse(
            status_code=503,
            content={
                "status": "model_not_loaded",
                "message": f"{settings.llm_model} is not among the provider models",
                "model": settings.llm_model,
                "available": models,
            },
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "model": settings.llm_model},
    )


@router.get("/metrics-lite", summary="In-process counters for the queue and model runs")
async def metrics_lite(metrics: MetricsDep, settings: SettingsDep) -> dict[str, object]:
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "prompt_version": PROMPT_VERSION,
        "max_concurrency": settings.llm_max_concurrency,
        **metrics.snapshot(),
    }
