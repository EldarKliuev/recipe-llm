from app.providers.base import OpenAICompatibleProvider


class CloudProvider(OpenAICompatibleProvider):
    """Any hosted OpenAI-compatible endpoint. Same interface, so the mobile client can be
    served when the home LM Studio is unreachable."""

    name = "cloud"
