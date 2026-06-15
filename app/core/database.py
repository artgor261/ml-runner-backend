"""Подключение к PostgreSQL.

Два движка:
- async (asyncpg) — для FastAPI-эндпоинтов;
- sync (psycopg2) — для воркера обучения, выполняющегося в отдельном процессе.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей."""


# ---------- Async (API) ----------
async_engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(async_engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI-зависимость: async-сессия БД."""
    async with AsyncSessionLocal() as session:
        yield session


# ---------- Sync (воркер обучения) ----------
sync_engine = create_engine(settings.database_url_sync, echo=False, pool_pre_ping=True)
SyncSessionLocal = sessionmaker(sync_engine, expire_on_commit=False, class_=Session)


@contextmanager
def sync_session():
    """Контекстный менеджер sync-сессии (используется в процессе-воркере)."""
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def init_models() -> None:
    """Создать таблицы (dev-режим). В проде использовать alembic."""
    # импорт моделей регистрирует их в metadata
    from app import models  # noqa: F401

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
