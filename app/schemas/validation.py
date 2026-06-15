"""Схемы валидации моделей."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class ValidationRequest(BaseModel):
    """Запуск валидации. Тестовые данные берутся из датасета (dataset_id) либо
    из ранее загруженного каталога parquet (parquet_dir)."""

    model_id: uuid.UUID | None = Field(None, description="ID зарегистрированной модели")
    model_path: str | None = Field(None, description="Прямой путь к .pt (если без регистрации)")

    dataset_id: uuid.UUID | None = Field(None, description="Датасет с тестовыми данными")
    parquet_dir: str | None = Field(None, description="Каталог с parquet (альтернатива dataset_id)")

    tickers: list[str] = Field(..., min_length=1)
    feature_cols: list[str] | None = None

    include_predictions: bool = True
    include_backtest: bool = False
    backtest_threshold: float = Field(0.01 / 25, description="Порог принятия торгового решения")


class TickerPrediction(BaseModel):
    index: list[int]
    predicted: list[float]
    actual: list[float]


class ValidationResponse(BaseModel):
    output_chunk_length: int
    tickers: list[str]
    metrics: dict
    date_ranges: dict
    predictions: dict[str, TickerPrediction] | None = None
    backtest: dict | None = None
