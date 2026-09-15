class ServiceError(Exception):
    """Base error carrying an HTTP status and a machine-readable code for core-api."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, detail: object | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class QueueFullError(ServiceError):
    """Waiting for a GPU slot exceeded LLM_QUEUE_WAIT_SEC."""

    status_code = 503
    code = "queue_full"

    def __init__(self, message: str, retry_after_sec: int) -> None:
        super().__init__(message)
        self.retry_after_sec = retry_after_sec


class ProviderUnavailableError(ServiceError):
    status_code = 503
    code = "provider_unavailable"


class ProviderTimeoutError(ServiceError):
    status_code = 504
    code = "provider_timeout"


class ModelOutputInvalidError(ServiceError):
    """The model failed to produce output matching the schema, repair attempts included."""

    status_code = 422
    code = "invalid_model_output"
