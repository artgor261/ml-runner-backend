"""Запуск обучения и мониторинг."""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.models import RunStatus
from app.schemas.training import RunMetricPoint, RunRead, TrainRequest, TrainingStatusDetail
from app.services import experiment_service, training_service
from app.services.training_service import TrainingError

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/runs", response_model=RunRead, status_code=status.HTTP_201_CREATED)
async def start_training(req: TrainRequest, session: AsyncSession = Depends(get_session)):
    """Запустить обучение с гиперпараметрами, переданными напрямую."""
    try:
        return await training_service.create_and_submit(session, req)
    except TrainingError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


@router.post("/runs/from-file", response_model=RunRead, status_code=status.HTTP_201_CREATED)
async def start_training_from_file(
    config: UploadFile = File(..., description="JSON-файл с телом TrainRequest"),
    session: AsyncSession = Depends(get_session),
):
    """Запустить обучение, загрузив конфигурацию гиперпараметров из JSON-файла."""
    raw = await config.read()
    try:
        payload = json.loads(raw)
        req = TrainRequest.model_validate(payload)
    except (json.JSONDecodeError, PydanticValidationError) as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Некорректный JSON-конфиг: {e}")
    try:
        return await training_service.create_and_submit(session, req)
    except TrainingError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


@router.get("/runs", response_model=list[RunRead])
async def list_runs(
    active: bool | None = Query(None, description="true — активные, false — завершённые"),
    run_status: RunStatus | None = Query(None, alias="status"),
    session: AsyncSession = Depends(get_session),
):
    """Список запусков обучения (активные/завершённые)."""
    return await training_service.list_runs(session, status=run_status, active=active)


@router.get("/runs/{run_id}", response_model=RunRead)
async def get_run(run_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    run = await experiment_service.get_run(session, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Запуск не найден")
    return run


@router.get("/runs/{run_id}/status", response_model=TrainingStatusDetail)
async def get_run_status(run_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    """Текущий статус: эпоха, loss и история метрик."""
    run = await experiment_service.get_run(session, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Запуск не найден")
    history = await experiment_service.get_run_history(session, run_id)
    points = [RunMetricPoint.model_validate(h, from_attributes=True) for h in history]
    return TrainingStatusDetail(
        id=run.id,
        name=run.name,
        status=run.status,
        current_epoch=run.current_epoch,
        total_epochs=run.total_epochs,
        metrics=run.metrics,
        error=run.error,
        history=points,
    )


@router.post("/runs/{run_id}/cancel", response_model=RunRead)
async def cancel_run(run_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    run = await training_service.cancel_run(session, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Запуск не найден")
    return run
