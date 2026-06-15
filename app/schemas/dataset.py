"""Схемы управления данными/датасетами."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.enums import DatasetSource


class MoexLoadRequest(BaseModel):
    """Загрузка исторических данных с MOEX."""

    name: str = Field(..., description="Имя создаваемого датасета")
    tickers: list[str] = Field(..., min_length=1, description="Список тикеров")
    start: str = Field(..., description="Начало диапазона, YYYY-MM-DD")
    end: str | None = Field(None, description="Конец диапазона, YYYY-MM-DD")
    board: str | None = Field(None, description="Режим торгов (по умолчанию из настроек)")
    interval: int | None = Field(None, description="Интервал свечей (по умолчанию из настроек)")
    concurrency: int | None = Field(None, ge=1, le=20, description="Параллельных загрузок")
    description: str | None = None


class LocalImportRequest(BaseModel):
    """Регистрация датасета из локального каталога с parquet-файлами."""

    name: str
    path: str = Field(..., description="Путь к каталогу с <TICKER>.parquet")
    tickers: list[str] | None = Field(None, description="Если не указано — берутся все .parquet из каталога")
    description: str | None = None


class GDriveImportRequest(BaseModel):
    """Импорт датасета из Google Drive (по ссылке/ID папки или файла)."""

    name: str
    gdrive_url: str = Field(..., description="Ссылка или ID папки/файла Google Drive")
    tickers: list[str] | None = None
    description: str | None = None


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    source: DatasetSource
    path: str
    tickers: list[str]
    interval: int | None
    start: str | None
    end: str | None
    rows: int | None
    meta: dict
    created_at: datetime
