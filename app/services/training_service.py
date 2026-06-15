"""Сервис обучения: создание запусков, постановка в исполнитель, мониторинг."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.jobs import get_runner
from app.models import Dataset, Executor, Run, RunStatus
from app.schemas.training import TrainRequest
from app.services import experiment_service

ARCHITECTURE = "tcn_multi"
ACTIVE_STATUSES = (RunStatus.PENDING, RunStatus.RUNNING)


class TrainingError(Exception):
    """Доменная ошибка обучения."""


def _params_dict(req: TrainRequest) -> dict:
    params = req.params.model_dump(exclude_none=True)
    params.setdefault("device", "cpu")
    return params


async def create_and_submit(session: AsyncSession, req: TrainRequest) -> Run:
    # валидация датасета
    if req.dataset_id is not None:
        dataset = await session.get(Dataset, req.dataset_id)
        if dataset is None:
            raise TrainingError(f"Датасет {req.dataset_id} не найден")

    exp = await experiment_service.get_or_create_experiment(
        session, name=req.experiment_name
    )

    run_name = req.run_name or f"{ARCHITECTURE}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = settings.runs_dir / exp.name / run_name

    run = Run(
        experiment_id=exp.id,
        name=run_name,
        description=req.description,
        status=RunStatus.PENDING,
        executor=req.executor,
        dataset_id=req.dataset_id,
        tickers=req.tickers,
        feature_cols=req.feature_cols or [],
        params=_params_dict(req),
        total_epochs=req.params.n_epochs,
        run_dir=str(run_dir),
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    # ставим задачу в исполнитель (local -> subprocess; datasphere -> заглушка)
    runner = get_runner(req.executor)
    try:
        runner.submit(run.id)
    except NotImplementedError as e:
        run.status = RunStatus.FAILED
        run.error = str(e)
        await session.commit()
        await session.refresh(run)
        raise TrainingError(str(e)) from e

    await session.refresh(run)
    return run


async def list_runs(
    session: AsyncSession, *, status: RunStatus | None = None, active: bool | None = None
) -> list[Run]:
    stmt = select(Run).order_by(Run.created_at.desc())
    if status is not None:
        stmt = stmt.where(Run.status == status)
    elif active is True:
        stmt = stmt.where(Run.status.in_(ACTIVE_STATUSES))
    elif active is False:
        stmt = stmt.where(Run.status.notin_(ACTIVE_STATUSES))
    rows = await session.scalars(stmt)
    return list(rows)


async def cancel_run(session: AsyncSession, run_id: uuid.UUID) -> Run | None:
    run = await session.get(Run, run_id)
    if run is None:
        return None
    if run.status not in ACTIVE_STATUSES:
        return run
    runner = get_runner(run.executor)
    runner.cancel(run.id)
    await session.refresh(run)
    return run
