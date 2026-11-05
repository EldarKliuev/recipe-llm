"""Exit non-zero unless the provider at LLM_BASE_URL answers and lists LLM_MODEL."""

import asyncio
import sys

from app.config import get_settings
from app.errors import ServiceError
from app.providers import create_provider


async def main() -> int:
    settings = get_settings()
    provider = create_provider(settings)
    try:
        models = await provider.list_models()
    except ServiceError as exc:
        print(f"LLM unreachable at {settings.llm_base_url}: {exc.message}", file=sys.stderr)
        return 1
    finally:
        await provider.aclose()

    if settings.llm_model not in models:
        print(f"{settings.llm_model} is not loaded, provider has: {models}", file=sys.stderr)
        return 1
    print(f"LLM ok: {settings.llm_model} at {settings.llm_base_url}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
