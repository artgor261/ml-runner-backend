"""Общие схемы."""

from __future__ import annotations

from pydantic import BaseModel


class Message(BaseModel):
    detail: str


class IDResponse(BaseModel):
    id: str
