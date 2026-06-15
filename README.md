# ML-Trading Backend (FastAPI + PostgreSQL)

REST API над существующим ML-кодом из ноутбуков (`data_load`, `train_tcn_multi`,
`validate_tcn_multi`). Бизнес-логика переиспользуется без переписывания: функции
ноутбуков перенесены в `app/ml/*` и обёрнуты сервисами и эндпоинтами.

## Архитектура

```
app/
  core/        конфигурация (pydantic-settings), подключение к БД (async+sync)
  models/      ORM-модели (datasets, experiments, runs, run_metrics, registered_models)
  schemas/     Pydantic-схемы запросов/ответов
  ml/          переиспользуемый код ноутбуков:
                 preprocessing  — transform_stock_data, сборка multi-серий darts
                 data_loader    — загрузка MOEX (aiomoex), как в data_load.ipynb
                 trainer        — обучение TCNModel (train_tcn_multi / train_overnight)
                 validator      — historical_forecasts, метрики, данные для графиков
                 callbacks      — LossHistoryCallback + DBProgressCallback (прогресс в БД)
  jobs/        JobRunner: LocalProcessRunner (subprocess) сейчас,
               DataSphereRunner (заготовка), worker.py — процесс обучения
  services/    бизнес-логика поверх ORM и ml/
  api/routers/ HTTP-эндпоинты
  main.py      сборка приложения
```

**Поток обучения:** `POST /training/runs` → создаётся `Run` (PENDING) →
`LocalProcessRunner` стартует `python -m app.jobs.worker <run_id>` в отдельном
процессе → воркер пишет статус/эпохи/loss в PostgreSQL → эндпоинты мониторинга
читают прогресс в реальном времени. Заменить исполнитель на DataSphere/Celery —
реализовать `JobRunner` и зарегистрировать в `app/jobs/registry.py`.

**Хранилище экспериментов (MLflow-подобное):** метаданные (эксперименты, запуски,
история, метрики, гиперпараметры) — в PostgreSQL; артефакты (чекпоинты, `.pt`,
логи, `params.json`, `metrics.json`) — на диске в `runs/`, `checkpoints/`,
`models/`, `loss_history/` (переиспользуются каталоги ноутбуков).

## Установка

```bash
# окружение (Python 3.11+ локально; код совместим с 3.10 — см. ниже про DataSphere)
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# БД (нужен sudo один раз)
sudo bash scripts/setup_db.sh

cp .env.example .env   # при необходимости отредактировать (например, PROJECT_ROOT)
```

### Python-версии и Yandex DataSphere

- **Локально** backend работает на Python 3.11+.
- **На Yandex DataSphere** воркер обучения исполняется на **Python 3.10**.
- Чтобы код обучения (`app/ml/*`) гарантированно запускался и там, и там:
  - вся кодовая база написана совместимо с 3.10 (`requires-python = ">=3.10"`,
    `ruff`/`black` нацелены на `py310` — см. `pyproject.toml`);
  - зависимости удалённой задачи вынесены в отдельный **`requirements-datasphere.txt`**
    (только ML-пакеты под 3.10, без web/DB-слоя), который будет использоваться при
    сборке окружения DataSphere-задачи.
- CLI-клиент `datasphere` ставится локально (совместим с 3.8+) и нужен лишь для
  постановки задач. Сама интеграция (`DataSphereRunner`) пока не реализована —
  готова только архитектурная заготовка.

## Запуск

```bash
.venv/bin/uvicorn app.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

Таблицы создаются автоматически при старте (`db_auto_create=true`). Для прод-режима
используйте Alembic:

```bash
.venv/bin/alembic revision --autogenerate -m "init"
.venv/bin/alembic upgrade head
```

## Основные эндпоинты (`/api/v1`)

| Сценарий            | Метод и путь |
|---------------------|--------------|
| Данные: MOEX        | `POST /datasets/moex` |
| Данные: локальный путь | `POST /datasets/local` |
| Данные: Google Drive   | `POST /datasets/gdrive` |
| Список датасетов    | `GET /datasets` |
| Обучение            | `POST /training/runs` |
| Обучение из JSON-файла | `POST /training/runs/from-file` |
| Статус/эпоха/loss   | `GET /training/runs/{id}/status` |
| Активные/завершённые | `GET /training/runs?active=true|false` |
| Эксперименты        | `GET /experiments`, `GET /experiments/{id}` |
| Запуск (детально)   | `GET /experiments/runs/{run_id}` |
| Регистрация модели  | `POST /models/register`, `POST /models/upload` |
| Список моделей      | `GET /models` |
| Валидация           | `POST /validation/run` |

## Пример: обучение

```jsonc
// POST /api/v1/training/runs
{
  "experiment_name": "TCN_MULTI",
  "tickers": ["LKOH", "ROSN", "GAZP"],
  "feature_cols": ["open", "high", "low", "volume"],
  "dataset_id": null,            // null -> используется общий каталог parquets/
  "executor": "local",
  "params": { "n_epochs": 3, "batch_size": 128, "device": "cpu" }
}
```

## Пример: валидация

```jsonc
// POST /api/v1/validation/run
{
  "model_id": "…",              // или "model_path": "models/.../*.pt"
  "tickers": ["LKOH"],
  "include_predictions": true,  // данные для графиков (без картинок)
  "include_backtest": true
}
```
