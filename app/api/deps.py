"""Зависимости FastAPI."""

from __future__ import annotations

from app.core.database import get_session

__all__ = ["get_session"]
