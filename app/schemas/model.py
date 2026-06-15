"""Схемы реестра моделей."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelRegisterRequest(BaseModel):
    """Регистрация готовой модели из существующего .pt-файла."""

    name: str
    path: str = Field(..., description="Путь к .pt-файлу")
    description: str | None = None
    architecture: str | None = "tcn_multi"
    tickers: list[str] | None = None
    feature_cols: list[str] | None = None
    params: dict | None = None
    metrics: dict | None = None


class ModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    path: str
    framework: str
    architecture: str | None
    run_id: uuid.UUID | None
    params: dict
    metrics: dict | None
    tickers: list[str]
    feature_cols: list[str]
    meta: dict
    created_at: datetime
