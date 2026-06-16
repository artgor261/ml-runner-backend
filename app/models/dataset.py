"""Датасет — каталог с parquet-файлами тикеров (из MOEX, локального пути, GDrive)."""

from __future__ import annotations

from sqlalchemy import JSON, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.enums import DatasetSource
from app.models.mixins import Timestamps, UUIDPrimaryKey


class Dataset(UUIDPrimaryKey, Timestamps, Base):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[DatasetSource] = mapped_column(Enum(DatasetSource), index=True)

    # каталог с <TICKER>.parquet
    path: Mapped[str] = mapped_column(String(1024))
    tickers: Mapped[list[str]] = mapped_column(JSON, default=list)

    interval: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start: Mapped[str | None] = mapped_column(String(64), nullable=True)
    end: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rows: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # произвольные метаданные (board, исходный путь/URL, статистика по тикерам и т.п.)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
