"""Валидация моделей."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.schemas.validation import ValidationRequest, ValidationResponse
from app.services import validation_service
from app.services.validation_service import ValidationError

router = APIRouter(prefix="/validation", tags=["validation"])


@router.post("/run", response_model=ValidationResponse)
async def run_validation(req: ValidationRequest, session: AsyncSession = Depends(get_session)):
    """Запустить валидацию: метрики, предсказания и данные для графиков (без изображений)."""
    try:
        result = await validation_service.run_validation(session, req)
    except ValidationError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except FileNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    return result
