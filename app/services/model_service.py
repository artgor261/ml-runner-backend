"""Сервис реестра моделей: регистрация .pt, список, получение, удаление."""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import RegisteredModel
from app.schemas.model import ModelRegisterRequest


class ModelError(Exception):
    """Доменная ошибка реестра моделей."""


async def register(session: AsyncSession, req: ModelRegisterRequest) -> RegisteredModel:
    existing = await session.scalar(
        select(RegisteredModel).where(RegisteredModel.name == req.name)
    )
    if existing is not None:
        raise ModelError(f"Модель с именем '{req.name}' уже зарегистрирована")

    path = Path(req.path).expanduser()
    if not path.exists():
        raise ModelError(f"Файл модели не найден: {path}")

    model = RegisteredModel(
        name=req.name,
        description=req.description,
        path=str(path),
        framework="darts",
        architecture=req.architecture,
        params=req.params or {},
        metrics=req.metrics,
        tickers=req.tickers or [],
        feature_cols=req.feature_cols or [],
    )
    session.add(model)
    await session.commit()
    await session.refresh(model)
    return model


async def register_from_file(
    session: AsyncSession, *, name: str, saved_path: Path, description: str | None
) -> RegisteredModel:
    model = RegisteredModel(
        name=name,
        description=description,
        path=str(saved_path),
        framework="darts",
        architecture="tcn_multi",
        meta={"uploaded": True},
    )
    session.add(model)
    await session.commit()
    await session.refresh(model)
    return model


async def list_models(session: AsyncSession) -> list[RegisteredModel]:
    rows = await session.scalars(
        select(RegisteredModel).order_by(RegisteredModel.created_at.desc())
    )
    return list(rows)


async def get_model(session: AsyncSession, model_id: uuid.UUID) -> RegisteredModel | None:
    return await session.get(RegisteredModel, model_id)


async def delete_model(session: AsyncSession, model_id: uuid.UUID) -> bool:
    model = await session.get(RegisteredModel, model_id)
    if model is None:
        return False
    await session.delete(model)
    await session.commit()
    return True
