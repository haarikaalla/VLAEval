# syntax=docker/dockerfile:1.7
# MLflow tracking server with PostgreSQL backend store + filesystem/S3 artifact root.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "mlflow>=2.13.0" psycopg2-binary boto3

RUN groupadd --gid 1000 mlflow && useradd --uid 1000 --gid mlflow --create-home mlflow
RUN mkdir -p /mlartifacts && chown -R mlflow:mlflow /mlartifacts
USER mlflow

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

ENTRYPOINT ["mlflow", "server"]
CMD ["--host", "0.0.0.0", "--port", "5000"]
