"""Реестр моделей."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.core.config import settings
from app.schemas.common import Message
from app.schemas.model import ModelRead, ModelRegisterRequest
from app.services import model_service
from app.services.model_service import ModelError

router = APIRouter(prefix="/models", tags=["models"])


@router.post("/register", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
async def register_model(req: ModelRegisterRequest, session: AsyncSession = Depends(get_session)):
    """Зарегистрировать модель из существующего .pt-файла по пути."""
    try:
        return await model_service.register(session, req)
    except ModelError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


@router.post("/upload", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
async def upload_model(
    name: str = Form(...),
    description: str | None = Form(None),
    file: UploadFile = File(..., description=".pt-файл модели"),
    session: AsyncSession = Depends(get_session),
):
    """Загрузить .pt-файл модели и зарегистрировать его."""
    if not file.filename.endswith(".pt"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Ожидается файл .pt")

    dest_dir = settings.models_dir / "uploaded"
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved_path = dest_dir / f"{name}.pt"
    saved_path.write_bytes(await file.read())

    try:
        return await model_service.register_from_file(
            session, name=name, saved_path=saved_path, description=description
        )
    except ModelError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


@router.get("", response_model=list[ModelRead])
async def list_models(session: AsyncSession = Depends(get_session)):
    """Список зарегистрированных моделей."""
    return await model_service.list_models(session)


@router.get("/{model_id}", response_model=ModelRead)
async def get_model(model_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    model = await model_service.get_model(session, model_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Модель не найдена")
    return model


@router.delete("/{model_id}", response_model=Message)
async def delete_model(model_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    ok = await model_service.delete_model(session, model_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Модель не найдена")
    return Message(detail="Модель удалена из реестра")
