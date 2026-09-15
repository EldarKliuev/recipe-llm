FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./


FROM base AS dev
RUN uv sync --frozen --no-install-project
COPY . .
EXPOSE 9000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]


FROM base AS prod
RUN uv sync --frozen --no-install-project --no-dev
COPY app ./app
COPY tools ./tools
EXPOSE 9000
# One worker on purpose: the GPU semaphore lives in-process, and the service is IO-bound.
CMD ["gunicorn", "app.main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "1", \
     "-b", "0.0.0.0:9000", \
     "--timeout", "180", \
     "--graceful-timeout", "30"]
