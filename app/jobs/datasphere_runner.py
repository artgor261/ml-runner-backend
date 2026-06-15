"""DataSphereRunner — архитектурная заготовка интеграции с Yandex DataSphere.

На текущем этапе достаточно интерфейса. Реальная реализация будет:
1. упаковывать код и конфиг запуска;
2. отправлять job через DataSphere Jobs API / CLI (datasphere);
3. периодически синхронизировать статус и метрики обратно в БД.

См. https://yandex.cloud/ru/docs/datasphere/concepts/jobs/
"""

from __future__ import annotations

import logging
import uuid

logger = logging.getLogger(__name__)


class DataSphereRunner:
    name = "datasphere"

    def __init__(self, *, project_id: str | None = None, oauth_token: str | None = None):
        self.project_id = project_id
        self.oauth_token = oauth_token

    def submit(self, run_id: uuid.UUID) -> None:
        # TODO: сформировать config.yaml для `datasphere project job execute`,
        #       загрузить датасет/код, запустить job, сохранить job_id в Run.meta.
        raise NotImplementedError(
            "DataSphere-исполнитель пока не реализован. "
            "Заготовка интерфейса готова — используйте executor='local'."
        )

    def cancel(self, run_id: uuid.UUID) -> bool:
        # TODO: `datasphere project job cancel --id <job_id>`
        raise NotImplementedError("Отмена DataSphere-задач пока не реализована.")
