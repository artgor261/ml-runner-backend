"""Эксперименты (MLflow-подобное хранилище запусков)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.schemas.experiment import (
    ExperimentCreate,
    ExperimentDetail,
    ExperimentRead,
    RunDetail,
)
from app.schemas.training import RunMetricPoint, RunRead
from app.services import experiment_service

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentRead, status_code=status.HTTP_201_CREATED)
async def create_experiment(req: ExperimentCreate, session: AsyncSession = Depends(get_session)):
    exp = await experiment_service.create_experiment(
        session, name=req.name, description=req.description
    )
    return ExperimentRead.model_validate(exp, from_attributes=True).model_copy(update={"run_count": 0})


@router.get("", response_model=list[ExperimentRead])
async def list_experiments(session: AsyncSession = Depends(get_session)):
    """Список экспериментов с количеством запусков."""
    pairs = await experiment_service.list_experiments(session)
    return [
        ExperimentRead.model_validate(exp, from_attributes=True).model_copy(update={"run_count": count})
        for exp, count in pairs
    ]


@router.get("/runs", response_model=list[RunRead])
async def list_all_runs(session: AsyncSession = Depends(get_session)):
    """Плоский список всех запусков (для общей таблицы экспериментов)."""
    return await experiment_service.list_runs(session)


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run_detail(run_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    """Полная информация по запуску: гиперпараметры, история, метрики, пути."""
    run = await experiment_service.get_run(session, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Запуск не найден")
    history = await experiment_service.get_run_history(session, run_id)
    points = [RunMetricPoint.model_validate(h, from_attributes=True) for h in history]
    detail = RunDetail.model_validate(run, from_attributes=True)
    return detail.model_copy(update={"history": points})


@router.get("/{experiment_id}", response_model=ExperimentDetail)
async def get_experiment(experiment_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    exp = await experiment_service.get_experiment(session, experiment_id)
    if exp is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Эксперимент не найден")
    detail = ExperimentDetail.model_validate(exp, from_attributes=True)
    return detail.model_copy(update={"run_count": len(exp.runs)})


@router.get("/{experiment_id}/runs", response_model=list[RunRead])
async def list_experiment_runs(
    experiment_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    return await experiment_service.list_runs(session, experiment_id=experiment_id)
