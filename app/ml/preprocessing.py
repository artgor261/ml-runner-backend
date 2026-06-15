"""Подготовка данных — перенос из ноутбуков (transform_stock_data, chunks, сборка серий).

Логика идентична ноутбукам train_tcn_multi / validate_tcn_multi, чтобы результаты
обучения и валидации совпадали.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from darts import TimeSeries

# Дефолты из ноутбуков
DEFAULT_FEATURE_COLS = ["open", "high", "low", "volume"]
TARGET_COL = "close"
INPUT_CHUNK_DEFAULT = 180
TRADING_HOURS = list(range(10, 24))  # отбрасываем часы 00-09


def load_from_parquet(*, ticker: str, parquet_dir: str | Path) -> pd.DataFrame:
    """Загрузка данных тикера из parquet (как в ноутбуках)."""
    path = Path(parquet_dir) / f"{ticker.upper()}.parquet"
    return pd.read_parquet(path, engine="fastparquet")


def transform_stock_data(*, df: pd.DataFrame) -> pd.DataFrame:
    """Нормировка относительно prev_close + log-diff объёма (дословно из ноутбуков)."""
    data = df.copy()

    prev_close = data["close"].shift(1)

    data["open"] = (data["open"] / prev_close) - 1
    data["high"] = (data["high"] / prev_close) - 1
    data["low"] = (data["low"] / prev_close) - 1
    data["close"] = (data["close"] / prev_close) - 1
    data["volume"] = np.log1p(data["volume"]).diff()

    data = data.dropna().reset_index(drop=True)
    return data


def create_chunks(*, actual_flat, output_chunk):
    """Формирует чанки длины output_chunk из одномерного массива (дословно из ноутбуков)."""
    return [
        TimeSeries.from_series(pd.Series(actual_flat[i : output_chunk + i]))
        for i in range(len(actual_flat))
    ]


def get_chunks_sum(*, series, output_chunk=None):
    """Сумма каждого чанка (дословно из ноутбуков)."""
    return [float(np.sum(s.values().flatten())) for s in series]


@dataclass
class TickerSeries:
    ticker: str
    train_series: TimeSeries
    train_cov: TimeSeries
    test_series: TimeSeries
    test_cov: TimeSeries
    test_real: pd.DataFrame  # ненормированные данные для бэктеста
    date_range: dict


@dataclass
class PreparedData:
    """Подготовленные многотикерные серии darts для обучения/валидации."""

    tickers: list[str]
    feature_cols: list[str]
    items: list[TickerSeries] = field(default_factory=list)

    @property
    def train_series_list(self) -> list[TimeSeries]:
        return [i.train_series for i in self.items]

    @property
    def train_cov_list(self) -> list[TimeSeries]:
        return [i.train_cov for i in self.items]

    @property
    def test_series_list(self) -> list[TimeSeries]:
        return [i.test_series for i in self.items]

    @property
    def test_cov_list(self) -> list[TimeSeries]:
        return [i.test_cov for i in self.items]

    @property
    def date_ranges(self) -> dict:
        return {i.ticker: i.date_range for i in self.items}


def prepare_multi(
    *,
    tickers: list[str],
    parquet_dir: str | Path,
    feature_cols: list[str] | None = None,
    train_ratio: float = 0.8,
    hours: list[int] | None = None,
) -> PreparedData:
    """Готовит многотикерные train/test серии — общая логика train/validate ноутбуков."""
    feature_cols = feature_cols or list(DEFAULT_FEATURE_COLS)
    hours = hours or list(TRADING_HOURS)

    prepared = PreparedData(tickers=list(tickers), feature_cols=feature_cols)

    for ticker in tickers:
        raw = load_from_parquet(ticker=ticker, parquet_dir=parquet_dir)
        df = transform_stock_data(df=raw)

        mask = pd.to_datetime(df["begin"]).dt.hour.isin(hours)
        df = df[mask].reset_index(drop=True)
        raw_h = raw[pd.to_datetime(raw["begin"]).dt.hour.isin(hours)].reset_index(drop=True)

        train_size = round(train_ratio * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:].reset_index(drop=True)
        test_real = raw_h.iloc[train_size:-1].reset_index(drop=True)

        prepared.items.append(
            TickerSeries(
                ticker=ticker,
                train_series=TimeSeries.from_dataframe(train_df, value_cols=[TARGET_COL]),
                train_cov=TimeSeries.from_dataframe(train_df, value_cols=feature_cols),
                test_series=TimeSeries.from_dataframe(test_df, value_cols=[TARGET_COL]),
                test_cov=TimeSeries.from_dataframe(test_df, value_cols=feature_cols),
                test_real=test_real,
                date_range={
                    "train_start": str(train_df["begin"].min()),
                    "train_end": str(train_df["begin"].max()),
                    "test_start": str(test_df["begin"].min()),
                    "test_end": str(test_df["begin"].max()),
                },
            )
        )

    return prepared
