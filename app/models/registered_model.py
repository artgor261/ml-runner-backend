"""Зарегистрированная модель — запись о .pt-файле (обученном здесь или загруженном)."""

from __future__ import annotations

import uuid

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.mixins import Timestamps, UUIDPrimaryKey


class RegisteredModel(UUIDPrimaryKey, Timestamps, Base):
    __tablename__ = "registered_models"

    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # путь к .pt
    path: Mapped[str] = mapped_column(String(1024))
    framework: Mapped[str] = mapped_column(String(64), default="darts")
    architecture: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # связь с породившим запуском (если модель обучена в системе)
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )

    params: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tickers: Mapped[list[str]] = mapped_column(JSON, default=list)
    feature_cols: Mapped[list[str]] = mapped_column(JSON, default=list)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
