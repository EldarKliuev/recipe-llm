import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    Omit,
    omit,
)
from openai.types.shared import ReasoningEffort

from app.errors import ProviderTimeoutError, ProviderUnavailableError

log = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMMeta:
    provider: str
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class MalformedModelJSON(Exception):
    """The provider answered, but the body is not JSON matching the schema.

    Carries the raw text and the run metadata so the use case can build a repair prompt
    and still report token usage of the wasted attempt.
    """

    def __init__(self, message: str, raw: str, meta: LLMMeta) -> None:
        super().__init__(message)
        self.message = message
        self.raw = raw
        self.meta = meta


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    async def complete_json(
        self, system: str, user: str, schema: dict[str, Any], timeout_s: float
    ) -> tuple[dict[str, Any], LLMMeta]: ...

    async def list_models(self) -> list[str]: ...

    async def aclose(self) -> None: ...


def extract_json_object(text: str) -> str:
    """Tolerate fenced or prefixed output by taking the outermost JSON object."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return text.strip()
    return text[start : end + 1]


class OpenAICompatibleProvider:
    """Shared implementation: LM Studio and cloud differ in base_url, key, model and in
    reasoning_effort, which a backend sets only when it needs it."""

    name = "openai_compatible"
    reasoning_effort: ReasoningEffort | Omit = omit

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=0)

    async def complete_json(
        self, system: str, user: str, schema: dict[str, Any], timeout_s: float
    ) -> tuple[dict[str, Any], LLMMeta]:
        started = perf_counter()
        try:
            completion = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_schema", "json_schema": schema},
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                reasoning_effort=self.reasoning_effort,
                timeout=timeout_s,
            )
        except APITimeoutError as exc:
            raise ProviderTimeoutError(
                f"model did not answer within {timeout_s:.0f}s", detail=str(exc)
            ) from exc
        except APIConnectionError as exc:
            raise ProviderUnavailableError("cannot reach the model", detail=str(exc)) from exc
        except APIStatusError as exc:
            raise ProviderUnavailableError(
                f"model returned HTTP {exc.status_code}", detail=str(exc)
            ) from exc

        latency_ms = int((perf_counter() - started) * 1000)
        usage = completion.usage
        meta = LLMMeta(
            provider=self.name,
            model=completion.model or self.model,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )

        if not completion.choices:
            raise MalformedModelJSON("model returned no choices", raw="", meta=meta)
        content = completion.choices[0].message.content or ""
        try:
            data = json.loads(extract_json_object(content))
        except json.JSONDecodeError as exc:
            raise MalformedModelJSON(
                f"output is not valid JSON: {exc}", raw=content, meta=meta
            ) from exc
        if not isinstance(data, dict):
            raise MalformedModelJSON("output is not a JSON object", raw=content, meta=meta)
        return data, meta

    async def list_models(self) -> list[str]:
        try:
            page = await self._client.models.list()
        except APITimeoutError as exc:
            raise ProviderTimeoutError("provider model list timed out", detail=str(exc)) from exc
        except (APIConnectionError, APIStatusError) as exc:
            raise ProviderUnavailableError("cannot list provider models", detail=str(exc)) from exc
        return [model.id for model in page.data]

    async def aclose(self) -> None:
        await self._client.close()
