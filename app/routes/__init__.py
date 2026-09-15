from fastapi import APIRouter

from app.routes import adapt, health, parse

router = APIRouter()
router.include_router(health.router)
router.include_router(adapt.router)
router.include_router(parse.router)

__all__ = ["router"]
