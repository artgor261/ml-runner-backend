"""Обучение TCN-модели — перенос из train_tcn_multi.ipynb / train_overnight.py.

Многотикерное обучение: несколько временных рядов как target и как covariates.
Функция чистая (без input/print-driven логики), принимает подготовленные серии
и параметры, возвращает метрики и пути к артефактам.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn
from darts.models import TCNModel
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from app.ml.callbacks import LossHistoryCallback
from app.ml.preprocessing import PreparedData, create_chunks, get_chunks_sum

logger = logging.getLogger(__name__)

INPUT_CHUNK = 180

LOSS_FUNCTIONS = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "mae": nn.L1Loss,
    "huber": nn.HuberLoss,
    "smoothl1": nn.SmoothL1Loss,
}

DEFAULT_PARAMS = {
    "input_chunk_length": 180,
    "output_chunk_length": 5,
    "kernel_size": 3,
    "num_filters": 16,
    "dilation_base": 2,
    "num_layers": 4,
    "lr": 3e-4,
    "batch_size": 256,
    "n_epochs": 10,
    "loss": "mse",
}


def build_loss_fn(name: str) -> nn.Module:
    cls = LOSS_FUNCTIONS.get(str(name).lower())
    if cls is None:
        raise ValueError(f"Неизвестная функция потерь: {name}. Доступно: {list(LOSS_FUNCTIONS)}")
    return cls()


def _eval_per_ticker(*, model: TCNModel, prepared: PreparedData, output_chunk: int) -> list[dict]:
    """Метрики по чанк-суммам для каждого тикера (как в ноутбуке)."""
    all_metrics = []
    for item in prepared.items:
        pred = model.historical_forecasts(
            series=item.test_series,
            past_covariates=item.test_cov,
            forecast_horizon=output_chunk,
            stride=1,
            last_points_only=False,
            retrain=False,
            predict_kwargs={"dataloader_kwargs": {"num_workers": 4}},
        )
        actual_flat = item.test_series[INPUT_CHUNK : -(output_chunk - 1)].values().flatten()
        actual_chunks = create_chunks(actual_flat=actual_flat, output_chunk=output_chunk)

        pred_sum = get_chunks_sum(series=pred)
        actual_sum = get_chunks_sum(series=actual_chunks)

        all_metrics.append(
            {
                "ticker": item.ticker,
                "MAE": mean_absolute_error(actual_sum, pred_sum),
                "RMSE": root_mean_squared_error(actual_sum, pred_sum),
                "R2": r2_score(actual_sum, pred_sum),
                "sign_accuracy": float(np.mean(np.sign(pred_sum) == np.sign(actual_sum))),
            }
        )
    return all_metrics


def _aggregate_metrics(per_ticker: list[dict]) -> dict:
    keys = ["MAE", "RMSE", "R2", "sign_accuracy"]
    agg = {k: float(np.mean([m[k] for m in per_ticker])) for k in keys}
    agg["per_ticker"] = {m["ticker"]: {k: m[k] for k in keys} for m in per_ticker}
    return agg


def run_training(
    *,
    prepared: PreparedData,
    params: dict,
    exp_name: str,
    run_name: str,
    run_dir: Path,
    checkpoints_dir: Path,
    models_dir: Path,
    loss_history_dir: Path,
    device: str = "cpu",
    extra_callbacks: list[Callback] | None = None,
) -> dict:
    """Полный цикл обучения: fit -> история лоссов -> метрики -> сохранение модели.

    Возвращает dict с метриками и путями к артефактам.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    loss_history_dir.mkdir(parents=True, exist_ok=True)

    p = {**DEFAULT_PARAMS, **params}
    output_chunk = p["output_chunk_length"]

    loss_cb = LossHistoryCallback()
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename="epoch={epoch}",
        save_top_k=-1,
        every_n_epochs=1,
    )
    callbacks: list[Callback] = [loss_cb, checkpoint_cb, *(extra_callbacks or [])]

    model = TCNModel(
        input_chunk_length=p["input_chunk_length"],
        output_chunk_length=output_chunk,
        kernel_size=p["kernel_size"],
        num_filters=p["num_filters"],
        dilation_base=p["dilation_base"],
        num_layers=p["num_layers"],
        optimizer_kwargs={"lr": p["lr"]},
        batch_size=p["batch_size"],
        n_epochs=p["n_epochs"],
        pl_trainer_kwargs={
            "accelerator": device,
            "callbacks": callbacks,
            "logger": False,
            "enable_checkpointing": True,
        },
        loss_fn=build_loss_fn(p["loss"]),
        random_state=40,
    )

    logger.info("Старт обучения %s (%d тикеров, %d эпох)", run_name, len(prepared.tickers), p["n_epochs"])
    model.fit(
        series=prepared.train_series_list,
        past_covariates=prepared.train_cov_list,
        val_series=prepared.test_series_list,
        val_past_covariates=prepared.test_cov_list,
        dataloader_kwargs={"num_workers": 4},
    )

    # история лоссов
    history = loss_cb.history
    base = loss_history_dir / run_name
    pd.DataFrame(history).to_csv(f"{base}.csv", index=False)
    with open(f"{base}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # метрики
    per_ticker = _eval_per_ticker(model=model, prepared=prepared, output_chunk=output_chunk)
    metrics = _aggregate_metrics(per_ticker)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # модель
    model_path = models_dir / f"{exp_name}_{run_name}.pt"
    model.save(str(model_path))

    logger.info("Обучение завершено %s: MAE=%.6f R2=%.6f", run_name, metrics["MAE"], metrics["R2"])

    return {
        "metrics": metrics,
        "history": history,
        "model_path": str(model_path),
        "loss_history_csv": f"{base}.csv",
    }
