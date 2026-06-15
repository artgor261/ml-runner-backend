"""Загрузка исторических данных с MOEX — перенос из data_load.ipynb.

Отличия от ноутбука: убраны input()/print-driven вызовы, добавлены параметры
start/end и возврат структурированного результата вместо побочной печати.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiohttp
import aiomoex
import pandas as pd

logger = logging.getLogger(__name__)


async def fetch_candles(
    *,
    session: aiohttp.ClientSession,
    ticker: str,
    start: str,
    end: str | None = None,
    board: str = "TQBR",
    interval: int = 1,
) -> pd.DataFrame:
    """Загрузка свечей по одному тикеру (как fetch_candles в ноутбуке)."""
    data = await aiomoex.get_board_candles(
        session=session,
        security=ticker,
        interval=interval,
        start=start,
        end=end,
        board=board,
    )
    return pd.DataFrame(data)


async def _fetch_with_semaphore(
    *,
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    ticker: str,
    start: str,
    end: str | None,
    board: str,
    interval: int,
    save_dir: Path,
    retries: int = 5,
    retry_delay: float = 2.0,
) -> pd.DataFrame:
    """Обёртка с семафором и ретраями (как fetch_with_semaphore в ноутбуке)."""
    async with semaphore:
        await asyncio.sleep(0.3)
        for attempt in range(1, retries + 1):
            try:
                df = await fetch_candles(
                    session=session, ticker=ticker, start=start, end=end,
                    board=board, interval=interval,
                )
                df.to_parquet(save_dir / f"{ticker}.parquet", engine="fastparquet", index=False)
                logger.info("Сохранено: %s.parquet (%d строк)", ticker, len(df))
                return df
            except (aiohttp.ClientConnectionError, aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
                if attempt == retries:
                    logger.error("Ошибка %s после %d попыток: %s", ticker, retries, e)
                    raise
                wait = retry_delay * (2 ** (attempt - 1))
                logger.warning("%s: попытка %d/%d не удалась (%s), повтор через %.1fс",
                               ticker, attempt, retries, e, wait)
                await asyncio.sleep(wait)


async def load_all(
    *,
    tickers: list[str],
    start: str,
    end: str | None = None,
    save_dir: str | Path,
    board: str = "TQBR",
    interval: int = 1,
    concurrency: int = 5,
) -> dict[str, pd.DataFrame]:
    """Параллельная загрузка списка тикеров (как load_all в ноутбуке).

    Возвращает {ticker: DataFrame} только для успешно загруженных тикеров.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        coros = [
            _fetch_with_semaphore(
                semaphore=semaphore, session=session, ticker=t, start=start, end=end,
                board=board, interval=interval, save_dir=save_dir,
            )
            for t in tickers
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

    out: dict[str, pd.DataFrame] = {}
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            logger.warning("Пропущен %s: %s", ticker, result)
        else:
            out[ticker] = result
    return out
