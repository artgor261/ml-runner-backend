"""Схемы экспериментов."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.schemas.training import RunMetricPoint, RunRead


class ExperimentCreate(BaseModel):
    name: str
    description: str | None = None


class ExperimentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime
    run_count: int = 0


class ExperimentDetail(ExperimentRead):
    runs: list[RunRead] = []


class RunDetail(RunRead):
    """Полная информация по запуску = «эксперимент» в терминах ТЗ."""

    history: list[RunMetricPoint] = []
