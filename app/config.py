from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["lmstudio", "cloud"] = "lmstudio"
    llm_base_url: str = "http://host.docker.internal:1234/v1"
    llm_model: str = "qwen2.5-7b-instruct"
    llm_api_key: str = "lm-studio"
    llm_timeout_sec: float = Field(default=120.0, gt=0)
    # Whole run, queue included. Stays below core-api's LLM_SERVICE_TIMEOUT_SEC (180), otherwise
    # it gives up first and reports "model busy" where our 422 would have degraded honestly.
    llm_run_budget_sec: float = Field(default=170.0, gt=0)
    llm_max_concurrency: int = Field(default=1, ge=1)
    llm_queue_wait_sec: float = Field(default=30.0, gt=0)
    llm_max_tokens: int = Field(default=1536, gt=0)

    http_port: int = 9000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
