import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from app.config import get_settings
from app.domain.prompts import PROMPT_VERSION
from app.errors import QueueFullError, ServiceError
from app.logging import configure_logging, request_id_var
from app.metrics import Metrics
from app.providers import create_provider
from app.routes import router
from app.usecases.gate import GpuGate

log = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    metrics = Metrics()

    app.state.settings = settings
    app.state.metrics = metrics
    app.state.provider = create_provider(settings)
    app.state.gate = GpuGate(
        max_concurrency=settings.llm_max_concurrency,
        queue_wait_sec=settings.llm_queue_wait_sec,
        metrics=metrics,
    )
    log.info(
        "llm-service starting",
        extra={
            "provider": settings.llm_provider,
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "max_concurrency": settings.llm_max_concurrency,
            "prompt_version": PROMPT_VERSION,
        },
    )
    try:
        yield
    finally:
        await app.state.provider.aclose()
        log.info("llm-service stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Recipe AI llm-service",
        version="0.1.0",
        description=(
            "Adapts already retrieved recipes to a fridge inventory. Called only by core-api."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        request.state.request_id = request_id
        started = perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        duration_ms = int((perf_counter() - started) * 1000)
        response.headers[REQUEST_ID_HEADER] = request_id
        log.info(
            "request handled",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "request_id": request_id,
            },
        )
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
        log.warning(
            "request rejected",
            extra={"code": exc.code, "status": exc.status_code, "reason": exc.message},
        )
        response = JSONResponse(
            status_code=exc.status_code,
            content=_error_body(request, exc.code, exc.message, exc.detail),
        )
        if isinstance(exc, QueueFullError):
            response.headers["Retry-After"] = str(exc.retry_after_sec)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_error_body(
                request,
                "invalid_request",
                "request payload is invalid",
                jsonable_encoder(exc.errors()),
            ),
        )

    return app


def _error_body(
    request: Request, code: str, message: str, detail: object | None
) -> dict[str, object]:
    error: dict[str, object] = {"code": code, "message": message}
    if detail is not None:
        error["detail"] = detail
    return {"error": error, "request_id": getattr(request.state, "request_id", None)}


app = create_app()
