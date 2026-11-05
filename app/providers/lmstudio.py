from app.providers.base import OpenAICompatibleProvider


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio's local server on the host. Supports constrained decoding via json_schema.

    Reasoning is off because a reasoning model under json_schema puts the whole answer into
    reasoning_content and leaves content empty, and repair cannot fix that.
    """

    name = "lmstudio"
    reasoning_effort = "none"
