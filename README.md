# recipe-llm — llm-service

Внутренний сервис Recipe AI, единственный процесс, который общается с языковой моделью. `core-api` присылает ему 1–5 рецептов, уже подобранных SQL-поиском по холодильнику, а сервис просит модель выбрать один из них и адаптировать под имеющиеся продукты и предпочтения. Ответ — строго структурированный JSON.

```
web  →  core-api  →  llm-service  →  модель (LM Studio на хосте или облачный OpenAI-совместимый endpoint)
```

Сервис опционален: подбор рецептов по холодильнику работает и без него. Если модель недоступна или ответила мусором, `core-api` честно показывает неадаптированный рецепт или «модель занята».

## Быстрый старт

Нужны Python 3.12+, [uv](https://docs.astral.sh/uv/) и [LM Studio](https://lmstudio.ai/) с загруженной моделью и запущенным сервером на `:1234`.

```bash
make install
```

```bash
make env
```

Впиши в `.env` в `LLM_MODEL` идентификатор модели из LM Studio (по умолчанию `qwen/qwen3.5-9b`), затем:

```bash
make llm-check
```

```bash
make dev
```

Сервис поднимется на `http://127.0.0.1:9000`, OpenAPI-документация будет на `/docs`. Проверка:

```bash
make ready
```

```bash
make adapt
```

`make adapt` отправляет в `/v1/adapt` первую eval-фикстуру.

### В Docker

Контейнер подключается к внешней сети `recipe-net`, которую создаёт корневой Makefile проекта Recipe AI. LM Studio остаётся на хосте и в Compose не входит.

```bash
make up
```

`.env` настроен на запуски с хоста (`127.0.0.1`), поэтому Compose подменяет `LLM_BASE_URL` значением `LLM_DOCKER_BASE_URL`, по умолчанию `http://host.docker.internal:1234/v1`. Чтобы направить контейнер на облачного провайдера, задай в `.env` оба адреса.

Образ `prod` (`make build`) запускает gunicorn ровно с одним воркером. Так нужно: семафор, ограничивающий доступ к GPU, живёт внутри процесса.

## API

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/v1/adapt` | Адаптировать один из присланных рецептов под инвентарь |
| POST | `/v1/parse-ingredients` | Разобрать строки ингредиентов (до 50) в структуру; используется ingest-скриптом `core-api` |
| GET | `/health` | Liveness: процесс отвечает |
| GET | `/ready` | Readiness: провайдер доступен и `LLM_MODEL` есть в списке моделей |
| GET | `/metrics-lite` | Очередь, счётчики запусков, доля repair, p95 задержки |

Форма запросов и ответов задана Pydantic-моделями в [`app/domain/schemas.py`](app/domain/schemas.py). Это единственный источник правды: отдельного OpenAPI-файла нет.

Пример запроса к `/v1/adapt`:

```json
{
  "recipes": [
    {"id": 42, "title": "Tomato omelette", "ingredients": ["2 eggs", "1 tomato"], "steps": ["..."]}
  ],
  "inventory": ["eggs", "tomato", "butter"],
  "prefs": {"allergies": ["peanuts"], "diet": null, "dislikes": [], "servings": 2}
}
```

Ответ:

```json
{
  "recipe": {
    "base_recipe_id": 42,
    "title": "...",
    "steps": ["..."],
    "used_ingredients": ["eggs", "tomato"],
    "missing_ingredients": [],
    "substitutions": [{"missing": "...", "replace_with": "...", "note": "..."}]
  },
  "meta": {
    "provider": "lmstudio", "model": "qwen/qwen3.5-9b", "prompt_version": "v1",
    "latency_ms": 8123, "queue_wait_ms": 0, "retries": 0,
    "prompt_tokens": 950, "completion_tokens": 410
  }
}
```

### Статусы

`core-api` решает, что показать пользователю, по статусу ответа, поэтому смысл статусов — часть контракта.

| Статус | `error.code` | Когда | Что делает `core-api` |
|---|---|---|---|
| 400 | `invalid_request` | Некорректный запрос (вместо стандартного 422 FastAPI) | Отказ в самом запросе |
| 422 | `invalid_model_output` | Модель не дала валидный ответ даже после repair-попыток; причина в `error.detail.reason` | Показывает неадаптированный рецепт |
| 503 | `queue_full`, `provider_unavailable` | Очередь к GPU полна (с заголовком `Retry-After`) или провайдер недоступен | «Модель занята» |
| 504 | `provider_timeout` | Модель не ответила ни разу за отведённое время | «Модель занята» |
| 499 | `client_disconnected` | Клиент разорвал соединение; виден только в логах сервиса | — |

Тело ошибки: `{"error": {"code", "message", "detail?"}, "request_id": "..."}`. Заголовок `X-Request-ID` принимается от вызывающей стороны или генерируется, возвращается в ответе и попадает во все JSON-логи.

## Как это устроено

```
routes/  →  usecases/{adapt,parse}.py  →  usecases/runner.py  →  usecases/gate.py  →  providers/
```

- **`run_structured`** ([`runner.py`](app/usecases/runner.py)): одна попытка и до двух repair-попыток, в каждой модели показывают текст ошибки валидации. Ограничение на общее время (`LLM_RUN_BUDGET_SEC`) отсчитывается с момента входа в очередь.
- **`GpuGate`** ([`gate.py`](app/usecases/gate.py)): семафор на `LLM_MAX_CONCURRENCY` слотов. Кто ждёт слот дольше `LLM_QUEUE_WAIT_SEC`, получает 503. Если клиент отключился, работа отменяется и слот освобождается.
- **Схема вывода** порождает `response_format` через `strict_json_schema`, поэтому модель декодирует ответ строго по ней.
- **Провайдеры** ([`providers/`](app/providers/)): `lmstudio` и `cloud`, переключаются через `LLM_PROVIDER` без правок кода. LM Studio отправляется `reasoning_effort="none"`, иначе модель с рассуждениями оставляет `content` пустым.

## Настройки

Читаются из окружения или `.env` ([`app/config.py`](app/config.py)). Шаблон — [`.env.example`](.env.example).

| Переменная | По умолчанию в `.env.example` | Назначение |
|---|---|---|
| `LLM_PROVIDER` | `lmstudio` | `lmstudio` или `cloud` |
| `LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | Адрес провайдера для запусков с хоста |
| `LLM_DOCKER_BASE_URL` | `http://host.docker.internal:1234/v1` | Адрес провайдера для контейнера |
| `LLM_MODEL` | `qwen/qwen3.5-9b` | Идентификатор модели у провайдера |
| `LLM_API_KEY` | `lm-studio` | Ключ провайдера; в логи не пишется |
| `LLM_TIMEOUT_SEC` | `120` | Таймаут одного вызова модели |
| `LLM_RUN_BUDGET_SEC` | `170` | Общий лимит на запрос вместе с очередью; должен быть меньше `LLM_SERVICE_TIMEOUT_SEC=180` в `core-api` |
| `LLM_MAX_CONCURRENCY` | `1` | Одновременных вызовов модели |
| `LLM_QUEUE_WAIT_SEC` | `30` | Сколько ждать слот до 503 |
| `LLM_MAX_TOKENS` | `1536` | Лимит токенов ответа |
| `HTTP_PORT` | `9000` | Пока ни на что не влияет: порт зашит в Dockerfile, Compose и Makefile |
| `LOG_LEVEL` | `INFO` | Уровень логов |

## Разработка

```bash
make help
```

Основные цели:

| Команда | Что делает |
|---|---|
| `make test` | pytest без live-теста; модель не нужна, используется `FakeProvider` |
| `make lint` / `make fmt` | ruff: проверка / автоформат |
| `make test-live` | smoke-тест против настоящего LM Studio |
| `make check` | lint + test + llm-check; без запущенной модели упадёт |
| `make health` / `make ready` / `make metrics` | Пробы запущенного сервиса |

Один тест:

```bash
uv run pytest tests/test_gate.py::test_queue_wait_timeout_raises_queue_full
```

CI в репозитории нет, поэтому перед коммитом обязательно запусти `make lint` и `make test`.

## Eval

Прогон 30 фиксированных кейсов из [`tools/eval/fixtures/cases.json`](tools/eval/fixtures/cases.json) через настроенную модель. Результат — CSV в `tools/eval/results/` (в git не попадает) с метриками: валидный JSON с первой попытки, нарушения аллергенов, ингредиенты вне инвентаря, не указанные в `missing`, доля отказов, медиана и p95 задержки. Postgres не нужен.

```bash
make eval-local
```

```bash
uv run python -m tools.eval --limit 3
```

`make eval` делает то же самое в контейнере. Eval обязателен при смене `LLM_MODEL` или промпта. Любая правка промпта поднимает `PROMPT_VERSION` в [`app/domain/prompts.py`](app/domain/prompts.py), иначе прогоны нельзя сравнить.

## Чего здесь нет

Сервис не ищет и не ранжирует рецепты, не ходит в Postgres или Redis и не вызывает `core-api`. Аллергены он передаёт модели как подсказку, а жёсткий запрет применяет `core-api`.

## Документы

- [`CLAUDE.md`](CLAUDE.md) — инварианты, правила работы и Definition of Done
- [`TASKS.md`](TASKS.md) — открытые задачи
