# syntax=docker/dockerfile:1.7
# Background worker image: same runtime dependencies as the API, but runs the
# job-polling worker loop instead of the HTTP server.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

FROM base AS builder
WORKDIR /build
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY README.md ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --no-deps -e .

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_ENV=production

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 vlaeval \
    && useradd --uid 1000 --gid vlaeval --shell /bin/bash --create-home vlaeval

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=vlaeval:vlaeval src ./src
COPY --chown=vlaeval:vlaeval configs ./configs
COPY --chown=vlaeval:vlaeval pyproject.toml README.md ./

RUN mkdir -p /app/data /app/models /app/reports /app/mlartifacts && \
    chown -R vlaeval:vlaeval /app

USER vlaeval

ENTRYPOINT ["python", "-m", "vla_eval.api.services.background_worker"]
