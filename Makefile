.DEFAULT_GOAL := help
VENV := backend/.venv
PY := $(VENV)/bin/python

.PHONY: help install test lint format migrate seed run web up down logs build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the backend venv and install both tiers
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r backend/requirements-dev.txt
	cd frontend && npm install

test: ## Run the backend test-suite
	cd backend && ../$(VENV)/bin/python -m pytest

lint: ## Lint both tiers
	cd backend && ../$(VENV)/bin/ruff check .
	cd frontend && npm run lint

format: ## Auto-format the backend
	cd backend && ../$(VENV)/bin/ruff format . && ../$(VENV)/bin/ruff check --fix .

migrate: ## Apply database migrations
	cd backend && ../$(VENV)/bin/alembic upgrade head

seed: ## Load demo data
	cd backend && ../$(VENV)/bin/python -m scripts.seed

run: ## Run the API with reload
	cd backend && ../$(VENV)/bin/uvicorn app.main:app --reload --port 8000

web: ## Run the Vite dev server
	cd frontend && npm run dev

up: ## Start the full stack in Docker
	docker compose up --build -d

down: ## Stop the stack
	docker compose down

logs: ## Tail stack logs
	docker compose logs -f

build: ## Build the production frontend bundle
	cd frontend && npm run build

clean: ## Remove build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf backend/.pytest_cache backend/.ruff_cache frontend/dist
