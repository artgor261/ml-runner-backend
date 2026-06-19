"""Процесс-воркер обучения: `python -m app.jobs.worker <run_id>`.

Исполняется LocalProcessRunner'ом в отдельном процессе. Сам обновляет статус,
прогресс и результаты запуска в БД (sync-сессия), что позволяет эндпоинтам
мониторинга видеть обучение в реальном времени.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.core.database import sync_session
from app.ml.callbacks import DBProgressCallback
from app.ml.preprocessing import prepare_multi
from app.ml.trainer import DEFAULT_PARAMS, run_training
from app.models import Dataset, RegisteredModel, Run, RunStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("worker")


def _json_safe(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def execute(run_id: uuid.UUID) -> None:
    # 1. помечаем запуск выполняющимся
    with sync_session() as session:
        run = session.get(Run, run_id)
        if run is None:
            logger.error("Run %s не найден", run_id)
            return
        run.status = RunStatus.RUNNING
        run.started_at = datetime.now()
        run.pid = os.getpid()

        exp_name = run.experiment.name
        run_name = run.name
        params = dict(run.params or {})
        tickers = list(run.tickers or [])
        feature_cols = list(run.feature_cols or [])
        device = str(params.get("device", "cpu"))
        dataset = session.get(Dataset, run.dataset_id) if run.dataset_id else None
        parquet_dir = dataset.path if dataset else str(settings.parquets_dir)
        run_dir = Path(run.run_dir) if run.run_dir else settings.runs_dir / exp_name / run_name

    try:
        # 2. подготовка данных (общая логика ноутбуков)
        prepared = prepare_multi(
            tickers=tickers,
            parquet_dir=parquet_dir,
            feature_cols=[col.lower() for col in feature_cols] or None,
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        full_params = {**DEFAULT_PARAMS, **params}
        with open(run_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "params": _json_safe(full_params),
                    "tickers": tickers,
                    "feature_cols": feature_cols,
                    "date_ranges": prepared.date_ranges,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        total_epochs = int(full_params["n_epochs"])
        progress_cb = DBProgressCallback(run_id=run_id, total_epochs=total_epochs)

        # 3. обучение (переиспользуем app.ml.trainer)
        result = run_training(
            prepared=prepared,
            params=params,
            exp_name=exp_name,
            run_name=run_name,
            run_dir=run_dir,
            checkpoints_dir=settings.checkpoints_dir / exp_name / run_name,
            models_dir=settings.models_dir / exp_name,
            loss_history_dir=settings.loss_history_dir / exp_name,
            device=device,
            extra_callbacks=[progress_cb],
        )

        # 4. фиксируем успех + авторегистрация модели
        with sync_session() as session:
            run = session.get(Run, run_id)
            run.status = RunStatus.COMPLETED
            run.metrics = result["metrics"]
            run.model_path = result["model_path"]
            run.checkpoints_dir = str(settings.checkpoints_dir / exp_name / run_name)
            run.run_dir = str(run_dir)
            run.finished_at = datetime.now()
            run.current_epoch = total_epochs

            session.add(
                RegisteredModel(
                    name=f"{exp_name}_{run_name}",
                    description=f"Автоматически зарегистрирована из запуска {run_name}",
                    path=result["model_path"],
                    framework="darts",
                    architecture="tcn_multi",
                    run_id=run_id,
                    params=_json_safe(full_params),
                    metrics=result["metrics"],
                    tickers=tickers,
                    feature_cols=feature_cols,
                )
            )
        logger.info("Запуск %s завершён успешно", run_name)

    except Exception as exc:  # noqa: BLE001
        logger.error("Запуск %s упал: %s", run_name, exc)
        tb = traceback.format_exc()
        with sync_session() as session:
            run = session.get(Run, run_id)
            if run is not None:
                run.status = RunStatus.FAILED
                run.error = tb[-8000:]
                run.finished_at = datetime.now()
        raise


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m app.jobs.worker <run_id>", file=sys.stderr)
        sys.exit(2)
    execute(uuid.UUID(sys.argv[1]))


if __name__ == "__main__":
    main()
