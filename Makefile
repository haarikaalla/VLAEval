.PHONY: help install install-dev lint format type-check test test-cov \
        run-api run-worker docker-build docker-up docker-down \
        download-data train evaluate migrate clean frontend-install frontend-dev frontend-build

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

lint: ## Run ruff lint checks
	ruff check src tests

format: ## Auto-format code with black + ruff
	black src tests
	ruff check --fix src tests

type-check: ## Run mypy static type checking
	mypy src

test: ## Run test suite
	pytest

test-cov: ## Run tests with coverage report
	pytest --cov=vla_eval --cov-report=html --cov-report=term-missing

security: ## Run bandit security scan
	bandit -c pyproject.toml -r src

run-api: ## Run the FastAPI backend locally (reload enabled)
	uvicorn vla_eval.api.main:app --reload --host 0.0.0.0 --port 8000

run-worker: ## Run background job worker locally
	python -m vla_eval.api.services.background_worker

docker-build: ## Build all Docker images via docker compose
	docker compose build

docker-up: ## Start the full stack (API, worker, MLflow, DB, frontend)
	docker compose up -d

docker-down: ## Stop the full stack
	docker compose down

migrate: ## Run database migrations
	alembic upgrade head

download-data: ## Download a dataset via CLI (usage: make download-data DATASET=lerobot/pusht)
	vla-eval data download $(DATASET)

train: ## Launch training via Hydra config (usage: make train CONFIG=configs/config.yaml)
	vla-eval train --config-name config

evaluate: ## Run evaluation/benchmark suite
	vla-eval evaluate --config-name config

frontend-install: ## Install frontend dependencies
	cd frontend && npm install

frontend-dev: ## Run frontend dev server
	cd frontend && npm run dev

frontend-build: ## Build frontend for production
	cd frontend && npm run build

clean: ## Remove caches and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage build dist *.egg-info
