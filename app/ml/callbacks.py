"""Колбэки обучения.

LossHistoryCallback — перенос из ноутбуков (копит историю лоссов).
DBProgressCallback — пишет прогресс (эпоха, train/val loss) в PostgreSQL,
чтобы эндпоинты мониторинга видели обучение в реальном времени.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pytorch_lightning.callbacks import Callback


class LossHistoryCallback(Callback):
    """Копит историю train/val loss по эпохам (дословно из ноутбуков)."""

    def __init__(self):
        self.train_losses: dict[int, float] = {}
        self.val_losses: dict[int, float] = {}

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.train_losses[trainer.current_epoch] = loss.item()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        loss = trainer.callback_metrics.get("val_loss")
        if loss is not None:
            self.val_losses[trainer.current_epoch] = loss.item()

    @property
    def history(self) -> list[dict]:
        epochs = sorted(set(self.train_losses) | set(self.val_losses))
        return [
            {
                "epoch": epoch,
                "train_loss": self.train_losses.get(epoch),
                "val_loss": self.val_losses.get(epoch),
            }
            for epoch in epochs
        ]


class DBProgressCallback(Callback):
    """Пишет прогресс обучения в БД после каждой эпохи (для мониторинга)."""

    def __init__(self, *, run_id: uuid.UUID, total_epochs: int):
        self.run_id = run_id
        self.total_epochs = total_epochs

    def _record(self, epoch: int, train_loss, val_loss):
        # импорт внутри, т.к. колбэк живёт в процессе-воркере
        from app.core.database import sync_session
        from app.models import Run, RunMetric

        with sync_session() as session:
            session.add(
                RunMetric(
                    run_id=self.run_id,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    recorded_at=datetime.now(),
                )
            )
            run = session.get(Run, self.run_id)
            if run is not None:
                run.current_epoch = epoch + 1
                run.total_epochs = self.total_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        self._record(
            epoch,
            train_loss.item() if train_loss is not None else None,
            val_loss.item() if val_loss is not None else None,
        )
