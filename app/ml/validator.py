"""Валидация модели — перенос из validate_tcn_multi.ipynb.

Загрузка .pt, historical_forecasts, расчёт метрик, формирование данных для
графиков на стороне фронтенда (без генерации изображений) и бэктест.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torchmetrics
from darts.models import TCNModel
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from app.ml.preprocessing import PreparedData, create_chunks, get_chunks_sum

logger = logging.getLogger(__name__)

INPUT_CHUNK = 180


def _register_safe_globals() -> None:
    """Разрешённые при torch.load классы (как в ноутбуке validate)."""
    torch.serialization.add_safe_globals(
        [
            torch.nn.modules.loss.MSELoss,
            torch.nn.modules.loss.L1Loss,
            torchmetrics.collections.MetricCollection,
            torch.optim.Adam,
        ]
    )


def load_model(*, model_path: str | Path) -> TCNModel:
    """Загрузка TCNModel из .pt с отключением логгера/колбэков (как в ноутбуке)."""
    _register_safe_globals()
    model = TCNModel.load(str(model_path))
    if hasattr(model, "trainer_params"):
        model.trainer_params["logger"] = False
        model.trainer_params["callbacks"] = []
    return model


def _backtest_ticker(*, item, pred, output_chunk: int, threshold: float) -> dict:
    """Бэктест по одному тикеру (перенос блока «Бэктест» из ноутбука)."""
    test_real = item.test_real
    test_target = item.test_series

    close_prev = test_real[INPUT_CHUNK : -(output_chunk - 1)]["close"].reset_index(drop=True)
    close_actual_scaled = test_target[INPUT_CHUNK : -(output_chunk - 1)]

    pl = [0.0]
    accum = [0.0]
    decisions = [0]

    n = min(len(pred), len(close_prev))
    for i in range(1, n):
        cumsum = pred[i].cumsum().last_value()
        if abs(cumsum) >= threshold:
            decisions.append(1 if cumsum > 0 else -1)
        else:
            decisions.append(0)

        close_actual_i = (close_actual_scaled[i].last_value() + 1) * close_prev[i].item()
        close_diff = close_actual_i - close_prev[i].item()

        pl.append(decisions[i - 1] * close_diff)
        accum.append(accum[i - 1] + pl[i])

    return {
        "close_prev": [float(x) for x in close_prev[:n].tolist()],
        "decision": decisions,
        "pl": pl,
        "accum": accum,
        "final_pl": accum[-1] if accum else 0.0,
    }


def validate(
    *,
    model_path: str | Path,
    prepared: PreparedData,
    include_predictions: bool = True,
    include_backtest: bool = False,
    backtest_threshold: float = 0.01 / 25,
) -> dict:
    """Полная валидация: метрики + (опц.) предсказания и бэктест для графиков.

    Возвращает JSON-сериализуемый dict, готовый для фронтенда (без изображений).
    """
    model = load_model(model_path=model_path)
    output_chunk = model.output_chunk_length

    all_preds = model.historical_forecasts(
        series=prepared.test_series_list,
        past_covariates=prepared.test_cov_list,
        retrain=False,
        forecast_horizon=output_chunk,
        last_points_only=False,
        predict_kwargs={"dataloader_kwargs": {"num_workers": 4}},
    )

    per_ticker_metrics = []
    predictions = {}
    backtests = {}

    for item, pred in zip(prepared.items, all_preds):
        actual_flat = item.test_series[INPUT_CHUNK : -(output_chunk - 1)].values().flatten()
        actual_chunks = create_chunks(actual_flat=actual_flat, output_chunk=output_chunk)

        pred_sum = get_chunks_sum(series=pred)
        actual_sum = get_chunks_sum(series=actual_chunks)

        per_ticker_metrics.append(
            {
                "ticker": item.ticker,
                "MAE": mean_absolute_error(actual_sum, pred_sum),
                "RMSE": root_mean_squared_error(actual_sum, pred_sum),
                "R2": r2_score(actual_sum, pred_sum),
                "sign_accuracy": float(np.mean(np.sign(pred_sum) == np.sign(actual_sum))),
            }
        )

        if include_predictions:
            predictions[item.ticker] = {
                "index": list(range(len(pred_sum))),
                "predicted": pred_sum,
                "actual": actual_sum,
            }

        if include_backtest:
            backtests[item.ticker] = _backtest_ticker(
                item=item, pred=pred, output_chunk=output_chunk, threshold=backtest_threshold
            )

    keys = ["MAE", "RMSE", "R2", "sign_accuracy"]
    aggregated = {k: float(np.mean([m[k] for m in per_ticker_metrics])) for k in keys}
    aggregated["per_ticker"] = {
        m["ticker"]: {k: m[k] for k in keys} for m in per_ticker_metrics
    }

    result = {
        "output_chunk_length": output_chunk,
        "tickers": prepared.tickers,
        "metrics": aggregated,
        "date_ranges": prepared.date_ranges,
    }
    if include_predictions:
        result["predictions"] = predictions
    if include_backtest:
        result["backtest"] = backtests
    return result
