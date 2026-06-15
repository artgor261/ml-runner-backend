"""Точка входа FastAPI.

Запуск (из каталога backend/):
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import datasets, experiments, health, models, training, validation
from app.core.config import settings
from app.core.database import init_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    if settings.db_auto_create:
        await init_models()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="REST API ML-пайплайна торговых ботов (данные, обучение, эксперименты, модели, валидация).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Роутеры
app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(datasets.router, prefix=settings.api_prefix)
app.include_router(training.router, prefix=settings.api_prefix)
app.include_router(experiments.router, prefix=settings.api_prefix)
app.include_router(models.router, prefix=settings.api_prefix)
app.include_router(validation.router, prefix=settings.api_prefix)


@app.get("/")
async def root() -> dict:
    return {"app": settings.app_name, "docs": "/docs", "api": settings.api_prefix}
