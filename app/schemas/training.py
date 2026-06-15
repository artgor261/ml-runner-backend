"""Схемы обучения и мониторинга."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import Executor, RunStatus


class Hyperparams(BaseModel):
    """Гиперпараметры TCN-модели (все опциональны, есть дефолты в trainer)."""

    input_chunk_length: int | None = Field(None, ge=1)
    output_chunk_length: int | None = Field(None, ge=1)
    kernel_size: int | None = Field(None, ge=1)
    num_filters: int | None = Field(None, ge=1)
    dilation_base: int | None = Field(None, ge=1)
    num_layers: int | None = Field(None, ge=1)
    lr: float | None = Field(None, gt=0)
    batch_size: int | None = Field(None, ge=1)
    n_epochs: int | None = Field(None, ge=1)
    loss: str | None = Field(None, description="mse | l1 | huber | smoothl1")
    device: str | None = Field(None, description="cpu | gpu")


class TrainRequest(BaseModel):
    """Запуск обучения. Гиперпараметры можно передать напрямую (params) либо
    отдельно загрузить из JSON-файла через /training/runs/from-file."""

    experiment_name: str = Field(..., description="Имя эксперимента (создаётся при отсутствии)")
    run_name: str | None = Field(None, description="Имя запуска (по умолчанию генерируется)")
    description: str | None = None

    dataset_id: uuid.UUID | None = Field(None, description="Датасет из БД")
    tickers: list[str] = Field(..., min_length=1)
    feature_cols: list[str] | None = Field(None, description="Признаки (по умолчанию open/high/low/volume)")

    params: Hyperparams = Field(default_factory=Hyperparams)
    executor: Executor = Executor.LOCAL


class RunMetricPoint(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    epoch: int
    train_loss: float | None
    val_loss: float | None


class RunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    experiment_id: uuid.UUID
    name: str
    description: str | None
    status: RunStatus
    executor: Executor
    dataset_id: uuid.UUID | None
    tickers: list[str]
    feature_cols: list[str]
    params: dict
    current_epoch: int | None
    total_epochs: int | None
    metrics: dict | None
    run_dir: str | None
    model_path: str | None
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime


class RunStatusRead(BaseModel):
    """Лёгкий ответ мониторинга статуса/прогресса."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    status: RunStatus
    current_epoch: int | None
    total_epochs: int | None
    metrics: dict | None
    error: str | None


class TrainingStatusDetail(RunStatusRead):
    """Статус + последние точки истории лоссов."""

    history: list[RunMetricPoint] = []
