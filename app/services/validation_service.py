"""Сервис валидации: запускает app.ml.validator на выбранной модели и данных."""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.ml.preprocessing import prepare_multi
from app.ml.validator import validate as ml_validate
from app.models import Dataset, RegisteredModel
from app.schemas.validation import ValidationRequest


class ValidationError(Exception):
    """Доменная ошибка валидации."""


async def _resolve_model_path(session: AsyncSession, req: ValidationRequest) -> str:
    if req.model_id is not None:
        model = await session.get(RegisteredModel, req.model_id)
        if model is None:
            raise ValidationError(f"Модель {req.model_id} не найдена")
        return model.path
    if req.model_path:
        if not Path(req.model_path).expanduser().exists():
            raise ValidationError(f"Файл модели не найден: {req.model_path}")
        return req.model_path
    raise ValidationError("Нужно указать model_id или model_path")


async def _resolve_parquet_dir(session: AsyncSession, req: ValidationRequest) -> str:
    if req.dataset_id is not None:
        dataset = await session.get(Dataset, req.dataset_id)
        if dataset is None:
            raise ValidationError(f"Датасет {req.dataset_id} не найден")
        return dataset.path
    if req.parquet_dir:
        if not Path(req.parquet_dir).expanduser().is_dir():
            raise ValidationError(f"Каталог не найден: {req.parquet_dir}")
        return req.parquet_dir
    return str(settings.parquets_dir)


def _run(*, model_path: str, parquet_dir: str, req: ValidationRequest) -> dict:
    prepared = prepare_multi(
        tickers=req.tickers,
        parquet_dir=parquet_dir,
        feature_cols=req.feature_cols or None,
    )
    return ml_validate(
        model_path=model_path,
        prepared=prepared,
        include_predictions=req.include_predictions,
        include_backtest=req.include_backtest,
        backtest_threshold=req.backtest_threshold,
    )


async def run_validation(session: AsyncSession, req: ValidationRequest) -> dict:
    model_path = await _resolve_model_path(session, req)
    parquet_dir = await _resolve_parquet_dir(session, req)
    # тяжёлый CPU-bound расчёт — выносим из event loop
    return await asyncio.to_thread(_run, model_path=model_path, parquet_dir=parquet_dir, req=req)
