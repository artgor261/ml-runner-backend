"""Слой запуска задач обучения (JobRunner): local сейчас, DataSphere/Celery позже."""

from app.jobs.base import JobRunner
from app.jobs.registry import get_runner

__all__ = ["JobRunner", "get_runner"]
