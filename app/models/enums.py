"""Перечисления, общие для ORM-моделей и схем."""

from __future__ import annotations

import enum


class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class Executor(str, enum.Enum):
    LOCAL = "local"
    DATASPHERE = "datasphere"


class DatasetSource(str, enum.Enum):
    MOEX = "moex"
    GDRIVE = "gdrive"
    UPLOAD = "upload"
