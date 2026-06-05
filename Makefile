.PHONY: help build up down logs ps health env restart pull reset
help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

env:  ## Copy .env.example -> .env if missing
	@test -f .env || cp .env.example .env

build: env  ## Build the app image
	docker compose build

up: env  ## Start all services in detached mode
	docker compose up -d

down:  ## Stop all services
	docker compose down

logs:  ## Tail logs of all services
	docker compose logs -f --tail=100

ps:  ## List service status
	docker compose ps

health:  ## Curl /health on the app
	@curl -fsS http://localhost:$${APP_PORT:-8000}/health | python3 -m json.tool || echo "app not healthy yet"

restart:  ## Restart everything
	docker compose down && docker compose up -d

pull:  ## Pull latest images
	docker compose pull

reset:  ## DESTRUCTIVE: stop + remove volumes (data loss)
	docker compose down -v
