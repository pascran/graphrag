# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates libpq-dev \
        zlib1g-dev libssl-dev libffi-dev \
        libreoffice-common \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (layer cache)
COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e .

# Copy app
COPY app ./app
COPY migrations ./migrations
COPY scripts ./scripts
COPY alembic.ini ./alembic.ini

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
