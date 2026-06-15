"""Интерфейс запуска обучения.

JobRunner абстрагирует «где исполняется обучение»: локальный процесс, Yandex
DataSphere, в будущем — Celery. Статус и прогрепсс хранятся в БД, поэтому раннеру
достаточно уметь запустить и (опционально) остановить задачу.
"""

from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable


@runtime_checkable
class JobRunner(Protocol):
    """Контракт исполнителя обучения."""

    name: str

    def submit(self, run_id: uuid.UUID) -> None:
        """Запустить обучение для запуска run_id (неблокирующе)."""
        ...

    def cancel(self, run_id: uuid.UUID) -> bool:
        """Остановить выполняющийся запуск. Возвращает True, если остановлен."""
        ...
