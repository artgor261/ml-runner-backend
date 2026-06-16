"""Эксперимент — логическая группировка запусков обучения (аналог MLflow Experiment)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.mixins import Timestamps, UUIDPrimaryKey

if TYPE_CHECKING:
    from app.models.run import Run


class Experiment(UUIDPrimaryKey, Timestamps, Base):
    __tablename__ = "experiments"

    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    runs: Mapped[list["Run"]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        order_by="Run.created_at.desc()",
    )
