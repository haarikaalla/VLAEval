# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Base stage: shared OS deps + Python virtualenv
# ---------------------------------------------------------------------------
FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VENV_PATH=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# ---------------------------------------------------------------------------
# Builder stage: install Python dependencies (cached separately from source)
# ---------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY README.md ./

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --no-deps -e .

# ---------------------------------------------------------------------------
# Runtime stage: minimal image with only what's needed to run the API
# ---------------------------------------------------------------------------
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    APP_ENV=production

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 vlaeval \
    && useradd --uid 1000 --gid vlaeval --shell /bin/bash --create-home vlaeval

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=vlaeval:vlaeval src ./src
COPY --chown=vlaeval:vlaeval configs ./configs
COPY --chown=vlaeval:vlaeval scripts ./scripts
COPY --chown=vlaeval:vlaeval pyproject.toml README.md ./

RUN mkdir -p /app/data /app/models /app/reports /app/mlartifacts && \
    chown -R vlaeval:vlaeval /app

USER vlaeval

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

ENTRYPOINT ["uvicorn", "vla_eval.api.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
