"""LocalProcessRunner — обучение в отдельном процессе на той же машине.

Запускает `python -m app.jobs.worker <run_id>`. Прогресс и статус воркер пишет
в БД сам, поэтому раннер только стартует процесс и умеет его остановить по PID.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from app.core.config import settings
from app.core.database import sync_session
from app.models import Run, RunStatus

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parents[2]


class LocalProcessRunner:
    name = "local"

    def submit(self, run_id: uuid.UUID) -> None:
        with sync_session() as session:
            run = session.get(Run, run_id)
            base = Path(run.run_dir) if run and run.run_dir else settings.runs_dir / str(run_id)
        log_path = base / "worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logfile = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            [settings.worker_python, "-m", "app.jobs.worker", str(run_id)],
            cwd=str(_BACKEND_DIR),
            stdout=logfile,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(_BACKEND_DIR)},
            start_new_session=True,
        )
        logger.info("Запущен воркер обучения run=%s pid=%s", run_id, proc.pid)

        with sync_session() as session:
            run = session.get(Run, run_id)
            if run is not None:
                run.pid = proc.pid
                run.log_path = str(log_path)

    def cancel(self, run_id: uuid.UUID) -> bool:
        with sync_session() as session:
            run = session.get(Run, run_id)
            if run is None or run.pid is None:
                return False
            pid = run.pid
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if run.status in (RunStatus.RUNNING, RunStatus.PENDING):
                run.status = RunStatus.STOPPED
            return True
