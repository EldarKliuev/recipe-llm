from app.providers.base import OpenAICompatibleProvider


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio's local server on the host. Supports constrained decoding via json_schema."""

    name = "lmstudio"
