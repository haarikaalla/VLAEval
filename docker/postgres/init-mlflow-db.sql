-- Initializes separate databases for the application and MLflow backend store
-- inside the single shared PostgreSQL instance used in docker-compose.
CREATE DATABASE mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO vla_eval;
