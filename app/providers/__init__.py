from app.config import Settings
from app.providers.base import LLMMeta, LLMProvider, MalformedModelJSON
from app.providers.cloud import CloudProvider
from app.providers.lmstudio import LMStudioProvider

__all__ = [
    "CloudProvider",
    "LLMMeta",
    "LLMProvider",
    "LMStudioProvider",
    "MalformedModelJSON",
    "create_provider",
]


def create_provider(settings: Settings) -> LLMProvider:
    provider_cls = LMStudioProvider if settings.llm_provider == "lmstudio" else CloudProvider
    return provider_cls(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
    )
