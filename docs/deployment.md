# Deployment Guide

## Docker Compose

The full stack — API, background worker, MLflow, PostgreSQL, Redis, frontend, Prometheus, Grafana — is defined in [`docker-compose.yml`](../docker-compose.yml).

```bash
cp .env.example .env     # fill in secrets (APP_SECRET_KEY, API_DEFAULT_API_KEY, etc.)
docker compose up -d --build
docker compose logs -f api worker   # tail logs
```

Services and default ports:

| Service | Port | Purpose |
|---|---|---|
| `frontend` | 8080 | React dashboard (nginx), proxies `/api/` to `api:8000` |
| `api` | 8000 | FastAPI backend |
| `worker` | — | Background job worker (training/evaluation execution) |
| `mlflow` | 5000 | MLflow tracking server (Postgres-backed) |
| `postgres` | 5432 | Application DB + MLflow backend store |
| `redis` | 6379 | Reserved for future caching/queueing |
| `prometheus` | 9090 | Metrics scraping |
| `grafana` | 3000 | Dashboards (Prometheus datasource pre-provisioned) |

Run database migrations (applied automatically on API container startup via lifespan `init_db()`; run manually if needed):

```bash
docker compose exec api alembic upgrade head
```

For local development with hot-reload, `docker-compose.override.yml` is applied automatically by `docker compose` and mounts source directories + adds `--reload` to the API command.

Tear down:

```bash
docker compose down            # stop containers, keep volumes
docker compose down -v         # stop and remove volumes (destructive)
```

## Kubernetes (Kustomize)

Manifests live under [`deploy/k8s/`](../deploy/k8s/):

```
base/                     # namespace, configmap, deployments, statefulset, PVCs, ingress
overlays/staging/         # staging patches (replicas, image tag, secrets)
overlays/production/      # production patches (replicas, image tag, secrets)
```

Deploy to staging:

```bash
kubectl apply -k deploy/k8s/overlays/staging
```

Deploy to production:

```bash
kubectl apply -k deploy/k8s/overlays/production
```

Before applying to a real cluster:

1. **Secrets**: replace the placeholder values in `overlays/{staging,production}/secret.yaml` with real secrets — ideally via a secret manager (Sealed Secrets, External Secrets Operator, SOPS) rather than committing plaintext.
2. **Images**: the overlays reference GHCR images tagged `staging`/`latest`, built and pushed by the `cd.yml` GitHub Actions workflow. Point `kustomization.yaml` `images:` entries at your own registry if not using GHCR.
3. **Storage**: `base/pvc.yaml` requests `ReadWriteMany` PVCs (dataset cache, model checkpoints, MLflow artifacts) — ensure your cluster's default StorageClass supports RWX (e.g. NFS, EFS, Azure Files), or switch to `ReadWriteOnce` + per-pod storage if not needed across replicas.
4. **GPU workers**: `base/deployment-worker.yaml` includes a commented `nvidia.com/gpu: "1"` resource limit for GPU-backed training; uncomment it only if your node pool has GPU nodes and the NVIDIA device plugin installed. Remove it entirely for CPU-only clusters.
5. **Ingress/TLS**: `base/ingress.yaml` assumes an `nginx` IngressClass and `cert-manager` for TLS — adjust annotations/`ClusterIssuer` name to match your cluster.

Scaling: `deployment-api` and `deployment-worker` include `HorizontalPodAutoscaler` resources (API: 2–10 replicas at 70% CPU; worker: 1–5 replicas) — tune thresholds based on observed load.

## CI/CD

GitHub Actions workflows (`.github/workflows/`):

- **`ci.yml`** — lint + type-check, pytest matrix (3.10/3.11/3.12), security scan (bandit + pip-audit), frontend lint/test, Docker build validation. Runs on every push/PR.
- **`cd.yml`** — builds and pushes `api`/`worker`/`frontend`/`mlflow` images to GHCR, scans images with Trivy, then deploys to the target environment (`staging` on `main`, `production` on tags/manual dispatch) via `kubectl` + Kustomize.
- **`security.yml`** — CodeQL (Python + JS/TS), Dependabot dependency review on PRs, secret scanning via gitleaks.
- **`nightly-benchmark.yml`** — scheduled run of the offline benchmark suite against the baseline model, uploading the generated report as a workflow artifact.

## Production checklist

- [ ] Set a strong, unique `APP_SECRET_KEY` and `API_DEFAULT_API_KEY` (the app refuses to start in `production`/`staging` with the insecure default — see `core/config.py`).
- [ ] Use a managed PostgreSQL instance (not the bundled container) for durability.
- [ ] Configure `MLFLOW_BACKEND_STORE_URI` and artifact storage (S3-compatible) for MLflow persistence.
- [ ] Restrict CORS origins (`API_CORS_ORIGINS`) to your actual frontend domain(s).
- [ ] Enable HTTPS/TLS at the ingress layer; never expose the API or MLflow directly without TLS.
- [ ] Review resource requests/limits and HPA thresholds against real workload profiling.
