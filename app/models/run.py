"""Запуск обучения (Run) и его история лоссов (RunMetric).

Каждый запуск = отдельный «эксперимент» в терминах ТЗ: хранит гиперпараметры,
историю обучения, метрики, пути к чекпоинтам/логам/модели.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enums import Executor, RunStatus
from app.models.mixins import Timestamps, UUIDPrimaryKey

if TYPE_CHECKING:
    from app.models.experiment import Experiment


class Run(UUIDPrimaryKey, Timestamps, Base):
    __tablename__ = "runs"

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus), default=RunStatus.PENDING, index=True
    )
    executor: Mapped[Executor] = mapped_column(Enum(Executor), default=Executor.LOCAL)

    # конфигурация обучения
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True
    )
    tickers: Mapped[list[str]] = mapped_column(JSON, default=list)
    feature_cols: Mapped[list[str]] = mapped_column(JSON, default=list)
    params: Mapped[dict] = mapped_column(JSON, default=dict)

    # прогресс
    current_epoch: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # результаты
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # артефакты (filesystem)
    run_dir: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    checkpoints_dir: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    log_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # исполнение
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    experiment: Mapped["Experiment"] = relationship(back_populates="runs")
    metric_history: Mapped[list["RunMetric"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="RunMetric.epoch",
    )


class RunMetric(UUIDPrimaryKey, Base):
    """Одна точка истории обучения (эпоха): train/val loss + доп. метрики."""

    __tablename__ = "run_metrics"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), index=True
    )
    epoch: Mapped[int] = mapped_column(Integer, index=True)
    train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    extra: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now()
    )

    run: Mapped["Run"] = relationship(back_populates="metric_history")
