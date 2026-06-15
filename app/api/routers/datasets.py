"""Управление данными/датасетами."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.schemas.common import Message
from app.schemas.dataset import (
    DatasetRead,
    GDriveImportRequest,
    LocalImportRequest,
    MoexLoadRequest,
)
from app.services import dataset_service
from app.services.dataset_service import DatasetError

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/moex", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def load_from_moex(req: MoexLoadRequest, session: AsyncSession = Depends(get_session)):
    """Загрузить исторические данные с MOEX (несколько тикеров параллельно)."""
    try:
        return await dataset_service.create_from_moex(session, req)
    except DatasetError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


@router.post("/local", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def import_from_local(req: LocalImportRequest, session: AsyncSession = Depends(get_session)):
    """Зарегистрировать датасет из локального каталога с parquet-файлами."""
    try:
        return await dataset_service.create_from_local(session, req)
    except DatasetError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


@router.post("/gdrive", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
async def import_from_gdrive(req: GDriveImportRequest, session: AsyncSession = Depends(get_session)):
    """Импортировать датасет из Google Drive по ссылке/ID."""
    try:
        return await dataset_service.create_from_gdrive(session, req)
    except (DatasetError, RuntimeError) as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


@router.get("", response_model=list[DatasetRead])
async def list_datasets(session: AsyncSession = Depends(get_session)):
    """Список доступных датасетов."""
    return await dataset_service.list_datasets(session)


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(dataset_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    dataset = await dataset_service.get_dataset(session, dataset_id)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Датасет не найден")
    return dataset


@router.delete("/{dataset_id}", response_model=Message)
async def delete_dataset(dataset_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    ok = await dataset_service.delete_dataset(session, dataset_id)
    if not ok:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Датасет не найден")
    return Message(detail="Датасет удалён")
