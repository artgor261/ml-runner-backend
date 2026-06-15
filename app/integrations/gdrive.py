"""Загрузка датасетов из Google Drive через gdown (публичные ссылки/папки)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_from_gdrive(*, url_or_id: str, dest_dir: str | Path) -> list[str]:
    """Скачивает файл или папку Google Drive в dest_dir.

    Возвращает список скачанных parquet-файлов (имена без расширения = тикеры).
    Требует публичный доступ по ссылке. Для приватных файлов нужен OAuth (TODO).
    """
    try:
        import gdown
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Пакет gdown не установлен. Добавьте gdown в зависимости.") from e

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    is_folder = "folders" in url_or_id or "drive/folders" in url_or_id
    if is_folder:
        gdown.download_folder(url=url_or_id, output=str(dest), quiet=False, use_cookies=False)
    else:
        # одиночный файл
        out = str(dest / "download.parquet")
        gdown.download(url=url_or_id, output=out, quiet=False, fuzzy=True)

    parquets = sorted(dest.glob("*.parquet"))
    if not parquets:
        raise RuntimeError(f"В {dest} не найдено parquet-файлов после загрузки из Google Drive")
    return [p.stem for p in parquets]
