"""Сервис управления датасетами: MOEX / локальный путь / Google Drive + БД."""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.integrations.gdrive import download_from_gdrive
from app.ml.data_loader import load_all
from app.models import Dataset, DatasetSource
from app.schemas.dataset import GDriveImportRequest, LocalImportRequest, MoexLoadRequest

logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """Доменная ошибка работы с датасетами."""


def _scan_stats(*, path: Path, tickers: list[str]) -> dict:
    """Считает строки и диапазон дат по parquet-файлам (best-effort)."""
    total_rows = 0
    per_ticker: dict[str, dict] = {}
    starts, ends = [], []
    for t in tickers:
        fp = path / f"{t.upper()}.parquet"
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp, engine="fastparquet")
        except Exception as e:  # noqa: BLE001
            logger.warning("Не удалось прочитать %s: %s", fp, e)
            continue
        total_rows += len(df)
        info = {"rows": len(df)}
        if "begin" in df.columns and len(df):
            begin = pd.to_datetime(df["begin"])
            info["start"] = str(begin.min())
            info["end"] = str(begin.max())
            starts.append(begin.min())
            ends.append(begin.max())
        per_ticker[t.upper()] = info
    return {
        "rows": total_rows,
        "per_ticker": per_ticker,
        "start": str(min(starts)) if starts else None,
        "end": str(max(ends)) if ends else None,
    }


async def _ensure_unique_name(session: AsyncSession, name: str) -> None:
    existing = await session.scalar(select(Dataset).where(Dataset.name == name))
    if existing is not None:
        raise DatasetError(f"Датасет с именем '{name}' уже существует")


async def create_from_moex(session: AsyncSession, req: MoexLoadRequest) -> Dataset:
    await _ensure_unique_name(session, req.name)

    save_dir = settings.datasets_dir / req.name
    board = req.board or settings.moex_board
    interval = req.interval or settings.moex_interval
    concurrency = req.concurrency or settings.moex_concurrency

    results = await load_all(
        tickers=req.tickers,
        start=req.start,
        end=req.end,
        save_dir=save_dir,
        board=board,
        interval=interval,
        concurrency=concurrency,
    )
    loaded = list(results.keys())
    if not loaded:
        raise DatasetError("Не удалось загрузить ни одного тикера с MOEX")

    stats = await asyncio.to_thread(_scan_stats, path=save_dir, tickers=loaded)
    dataset = Dataset(
        name=req.name,
        description=req.description,
        source=DatasetSource.MOEX,
        path=str(save_dir),
        tickers=loaded,
        interval=interval,
        start=req.start,
        end=req.end,
        rows=stats["rows"],
        meta={"board": board, "requested": req.tickers, "skipped": sorted(set(req.tickers) - set(loaded)),
              "per_ticker": stats["per_ticker"]},
    )
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    return dataset


async def create_from_local(session: AsyncSession, req: LocalImportRequest) -> Dataset:
    await _ensure_unique_name(session, req.name)

    path = Path(req.path).expanduser()
    if not path.is_dir():
        raise DatasetError(f"Каталог не найден: {path}")

    tickers = req.tickers or [p.stem for p in sorted(path.glob("*.parquet"))]
    if not tickers:
        raise DatasetError(f"В каталоге нет parquet-файлов: {path}")

    stats = await asyncio.to_thread(_scan_stats, path=path, tickers=tickers)
    dataset = Dataset(
        name=req.name,
        description=req.description,
        source=DatasetSource.LOCAL,
        path=str(path),
        tickers=[t.upper() for t in tickers],
        start=stats["start"],
        end=stats["end"],
        rows=stats["rows"],
        meta={"per_ticker": stats["per_ticker"]},
    )
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    return dataset


async def create_from_gdrive(session: AsyncSession, req: GDriveImportRequest) -> Dataset:
    await _ensure_unique_name(session, req.name)

    dest = settings.datasets_dir / req.name
    downloaded = await asyncio.to_thread(
        download_from_gdrive, url_or_id=req.gdrive_url, dest_dir=dest
    )
    tickers = [t.upper() for t in (req.tickers or downloaded)]

    stats = await asyncio.to_thread(_scan_stats, path=dest, tickers=tickers)
    dataset = Dataset(
        name=req.name,
        description=req.description,
        source=DatasetSource.GDRIVE,
        path=str(dest),
        tickers=tickers,
        start=stats["start"],
        end=stats["end"],
        rows=stats["rows"],
        meta={"gdrive_url": req.gdrive_url, "per_ticker": stats["per_ticker"]},
    )
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    return dataset


async def list_datasets(session: AsyncSession) -> list[Dataset]:
    rows = await session.scalars(select(Dataset).order_by(Dataset.created_at.desc()))
    return list(rows)


async def get_dataset(session: AsyncSession, dataset_id: uuid.UUID) -> Dataset | None:
    return await session.get(Dataset, dataset_id)


async def delete_dataset(session: AsyncSession, dataset_id: uuid.UUID) -> bool:
    dataset = await session.get(Dataset, dataset_id)
    if dataset is None:
        return False
    await session.delete(dataset)
    await session.commit()
    return True
