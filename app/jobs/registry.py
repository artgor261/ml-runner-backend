"""Фабрика исполнителей обучения по значению Executor."""

from __future__ import annotations

from app.jobs.base import JobRunner
from app.jobs.datasphere_runner import DataSphereRunner
from app.jobs.local_runner import LocalProcessRunner
from app.models import Executor

_RUNNERS: dict[Executor, JobRunner] = {
    Executor.LOCAL: LocalProcessRunner(),
    Executor.DATASPHERE: DataSphereRunner(),
}


def get_runner(executor: Executor) -> JobRunner:
    runner = _RUNNERS.get(executor)
    if runner is None:
        raise ValueError(f"Неизвестный исполнитель: {executor}")
    return runner
