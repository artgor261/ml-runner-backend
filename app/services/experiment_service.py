"""Сервис экспериментов и запусков (аналог MLflow поверх PostgreSQL)."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Experiment, Run, RunMetric


async def get_or_create_experiment(
    session: AsyncSession, *, name: str, description: str | None = None
) -> Experiment:
    exp = await session.scalar(select(Experiment).where(Experiment.name == name))
    if exp is None:
        exp = Experiment(name=name, description=description)
        session.add(exp)
        await session.flush()
    return exp


async def create_experiment(
    session: AsyncSession, *, name: str, description: str | None = None
) -> Experiment:
    exp = Experiment(name=name, description=description)
    session.add(exp)
    await session.commit()
    await session.refresh(exp)
    return exp


async def list_experiments(session: AsyncSession) -> list[tuple[Experiment, int]]:
    stmt = (
        select(Experiment, func.count(Run.id))
        .outerjoin(Run, Run.experiment_id == Experiment.id)
        .group_by(Experiment.id)
        .order_by(Experiment.created_at.desc())
    )
    rows = await session.execute(stmt)
    return [(exp, count) for exp, count in rows.all()]


async def get_experiment(session: AsyncSession, experiment_id: uuid.UUID) -> Experiment | None:
    stmt = (
        select(Experiment)
        .where(Experiment.id == experiment_id)
        .options(selectinload(Experiment.runs))
    )
    return await session.scalar(stmt)


async def list_runs(
    session: AsyncSession, *, experiment_id: uuid.UUID | None = None
) -> list[Run]:
    stmt = select(Run).order_by(Run.created_at.desc())
    if experiment_id is not None:
        stmt = stmt.where(Run.experiment_id == experiment_id)
    rows = await session.scalars(stmt)
    return list(rows)


async def get_run(session: AsyncSession, run_id: uuid.UUID) -> Run | None:
    return await session.get(Run, run_id)


async def get_run_history(session: AsyncSession, run_id: uuid.UUID) -> list[RunMetric]:
    rows = await session.scalars(
        select(RunMetric).where(RunMetric.run_id == run_id).order_by(RunMetric.epoch)
    )
    return list(rows)
