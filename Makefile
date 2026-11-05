# llm-service shortcuts. Local targets run on the host via uv; compose targets run
# in the container. The shared recipe-net network is created by the root Makefile.

SHELL := /bin/bash
COMPOSE ?= docker compose
SERVICE ?= llm-service
PORT ?= 9000
BASE ?= http://127.0.0.1:$(PORT)
CASES ?= tools/eval/fixtures/cases.json

.DEFAULT_GOAL := help

.PHONY: help install env dev test test-live lint fmt check llm-check up down restart logs sh build eval eval-local health ready metrics adapt clean

help: ## List the available targets
	@grep -hE '^[a-z%-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# --- local development -------------------------------------------------------

install: ## Create the venv and install dependencies from uv.lock
	uv sync

env: ## Create .env from .env.example if it does not exist yet
	@test -f .env && echo ".env already exists, leaving it alone" \
		|| { cp .env.example .env; echo "created .env, set LLM_MODEL to the model loaded in LM Studio"; }

dev: ## Run the service on the host with reload (LM Studio at LLM_BASE_URL)
	uv run uvicorn app.main:app --host 127.0.0.1 --port $(PORT) --reload

test: ## Run the test suite (the live smoke test is excluded)
	uv run pytest

test-live: ## Run the smoke test against a real LM Studio server
	uv run pytest -m live -v

lint: ## Check formatting and lint rules
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Reformat the code
	uv run ruff format .
	uv run ruff check --fix .

check: lint test llm-check ## Lint, tests and the LM Studio connection

llm-check: ## Is LM Studio at LLM_BASE_URL reachable and LLM_MODEL listed
	uv run python -m tools.check_llm

# --- compose -----------------------------------------------------------------

up: ## Start the service in the background (needs .env and recipe-net)
	@docker network inspect recipe-net > /dev/null 2>&1 \
		|| { echo "recipe-net is missing: create it from the root Makefile first"; exit 1; }
	$(COMPOSE) up -d --build

down: ## Stop the service
	$(COMPOSE) down

restart: ## Recreate the container
	$(COMPOSE) up -d --force-recreate

logs: ## Follow the service logs
	$(COMPOSE) logs -f $(SERVICE)

sh: ## Open a shell in the running container
	$(COMPOSE) exec $(SERVICE) bash

build: ## Build the prod image
	docker build --target prod -t recipe-llm-service:prod .

# --- evals -------------------------------------------------------------------

eval: ## Run the 30 fixtures in the container, write CSV to tools/eval/results
	$(COMPOSE) run --rm $(SERVICE) python -m tools.eval

eval-local: ## Same eval run from the host
	uv run python -m tools.eval

# --- probes ------------------------------------------------------------------

# Status code is printed rather than turned into a failure: a 503 from /ready carries the
# reason in its body, and that is the whole point of asking.

health: ## Liveness probe
	@curl -sS -w '\nHTTP %{http_code}\n' $(BASE)/health

ready: ## Readiness probe: is LLM_MODEL loaded at the provider
	@curl -sS -w '\nHTTP %{http_code}\n' $(BASE)/ready

metrics: ## Queue depth, run counters, repair rate, latency percentiles
	@curl -sS -w '\nHTTP %{http_code}\n' $(BASE)/metrics-lite

adapt: ## Send the first eval fixture to /v1/adapt
	@uv run python -c "import json,pathlib,sys; json.dump(json.loads(pathlib.Path('$(CASES)').read_text())[0], sys.stdout)" \
		| curl -sS -w '\nHTTP %{http_code}\n' -X POST $(BASE)/v1/adapt \
			-H 'Content-Type: application/json' -d @-

clean: ## Remove caches, the venv and eval results
	rm -rf .venv .pytest_cache .ruff_cache tools/eval/results
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
