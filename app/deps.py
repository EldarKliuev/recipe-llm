from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.metrics import Metrics
from app.providers.base import LLMProvider
from app.usecases.gate import GpuGate


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_provider(request: Request) -> LLMProvider:
    return request.app.state.provider


def get_gate(request: Request) -> GpuGate:
    return request.app.state.gate


def get_metrics(request: Request) -> Metrics:
    return request.app.state.metrics


SettingsDep = Annotated[Settings, Depends(get_settings)]
ProviderDep = Annotated[LLMProvider, Depends(get_provider)]
GateDep = Annotated[GpuGate, Depends(get_gate)]
MetricsDep = Annotated[Metrics, Depends(get_metrics)]
