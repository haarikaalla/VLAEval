# API Reference

The FastAPI backend serves all endpoints under the `/api/v1` prefix. Interactive documentation is auto-generated and available at:

- Swagger UI: `GET /api/docs`
- ReDoc: `GET /api/redoc`
- Raw OpenAPI schema: `GET /api/openapi.json`

## Authentication

Two mechanisms are supported:

1. **API key** (primary) — send `X-API-Key: <key>` on every request. The key is compared using `hmac.compare_digest` to avoid timing attacks. Configure via `API_DEFAULT_API_KEY` (see `.env.example`).
2. **JWT** (optional, for delegated/short-lived access) — exchange your API key for a bearer token:

   ```bash
   curl -X POST -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/auth/token
   # => {"access_token": "...", "token_type": "bearer", "expires_in_minutes": 60}
   ```

   Then send `Authorization: Bearer <token>` on subsequent requests.

Unauthenticated endpoints: `GET /healthz`, `GET /readyz`, `GET /metrics` (Prometheus).

## Endpoints

### Health

| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | Liveness probe — always returns 200 if the process is up. |
| GET | `/readyz` | Readiness probe — checks database connectivity. |

### Datasets (`/api/v1/datasets`)

| Method | Path | Description |
|---|---|---|
| GET | `/` | List all registered public datasets. |
| GET | `/{name}` | Get metadata for one dataset (404 with `error_code=DatasetNotFoundError` if unknown). |
| POST | `/download` | Trigger an async background download of a dataset (202 Accepted). |

### Models (`/api/v1/models`)

| Method | Path | Description |
|---|---|---|
| GET | `/` | List all registered model implementations (`baseline-cnn`, `openvla`, `lerobot-act`, `lerobot-diffusion`). |

### Training (`/api/v1/training`)

| Method | Path | Description |
|---|---|---|
| POST | `/jobs` | Submit a training job (dataset, model, hyperparameters). Returns a `Job` with `status=pending`. |
| GET | `/jobs/{job_id}` | Poll job status/result. |

### Evaluation (`/api/v1/evaluation`)

| Method | Path | Description |
|---|---|---|
| POST | `/jobs` | Submit an evaluation/benchmark job. |
| GET | `/jobs/{job_id}` | Poll job status/result (includes composite score, metrics, report path once complete). |

### Leaderboard (`/api/v1/leaderboard`)

| Method | Path | Description |
|---|---|---|
| GET | `/` | Ranked leaderboard entries, optionally filtered by `dataset` query param. |

### Auth (`/api/v1/auth`)

| Method | Path | Description |
|---|---|---|
| POST | `/token` | Exchange a valid API key for a short-lived JWT. |

## Error format

Domain errors (subclasses of `VLAEvalError`) are mapped to HTTP responses with a consistent shape:

```json
{
  "detail": "Dataset 'foo' is not registered.",
  "error_code": "DatasetNotFoundError"
}
```

Status code mapping: `DatasetNotFoundError` / `ModelNotFoundError` / `JobNotFoundError` → 404, `AuthenticationError` → 401, `AuthorizationError` → 403, `DatasetDownloadError` / `ModelLoadError` → 502, other `VLAEvalError` subclasses → 400/500 as appropriate.

## Observability

- **Structured logs**: every request gets a `request_id` and timing recorded via middleware (`vla_eval.api.main`), rendered as JSON in staging/production and console-formatted in development.
- **Metrics**: Prometheus metrics exposed at `/metrics` via `prometheus-fastapi-instrumentator` (request counts, latency histograms). Scraped by the bundled Prometheus service in Docker Compose / Kubernetes and visualized in the pre-provisioned Grafana dashboard.
