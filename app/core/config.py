"""Конфигурация приложения (pydantic-settings).

Все пути к артефактам выводятся от PROJECT_ROOT, чтобы переиспользовать уже
существующие каталоги ноутбуков (parquets/, runs/, models/, checkpoints/, ...).
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> корень репозитория backend
_BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Приложение ---
    app_name: str = "ML-Trading API"
    api_prefix: str = "/api/v1"
    debug: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"])

    # --- База данных ---
    database_url: str = "postgresql+asyncpg://mltrading:mltrading@localhost:5432/mltrading"
    db_auto_create: bool = True  # create_all при старте (удобно для dev; в prod — alembic)

    # --- Пути ---
    # По умолчанию артефакты лежат внутри репозитория (self-contained).
    # Чтобы переиспользовать данные из ноутбучного проекта (parquets/, runs/, ...),
    # задайте PROJECT_ROOT в .env, например PROJECT_ROOT=/home/Artem/ml-trading
    project_root: Path = _BACKEND_DIR

    # --- MOEX ---
    moex_board: str = "TQBR"
    moex_interval: int = 1
    moex_concurrency: int = 5

    # --- Обучение / воркер ---
    training_python: str | None = None  # интерпретатор для subprocess-воркера

    # ---------- Производные значения ----------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def database_url_sync(self) -> str:
        """Sync-URL (psycopg2) для воркера обучения, выведенный из async-URL."""
        return self.database_url.replace("+asyncpg", "+psycopg2")

    @property
    def worker_python(self) -> str:
        return self.training_python or sys.executable

    # Каталоги артефактов
    @property
    def parquets_dir(self) -> Path:
        return self.project_root / "parquets"

    @property
    def datasets_dir(self) -> Path:
        return self.project_root / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.project_root / "runs"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def checkpoints_dir(self) -> Path:
        return self.project_root / "checkpoints"

    @property
    def loss_history_dir(self) -> Path:
        return self.project_root / "loss_history"

    @property
    def uploads_dir(self) -> Path:
        return self.project_root / "uploads"

    def ensure_dirs(self) -> None:
        for d in (
            self.parquets_dir,
            self.datasets_dir,
            self.runs_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.loss_history_dir,
            self.uploads_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
