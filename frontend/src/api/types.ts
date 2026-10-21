export interface DatasetInfo {
  name: string;
  hub_repo_id: string;
  description: string;
  task_type: string;
  action_dim: number;
  modalities: string[];
  license: string;
  tags: string[];
}

export interface ModelInfo {
  name: string;
}

export type JobStatus = "pending" | "running" | "succeeded" | "failed" | "cancelled";
export type JobType = "training" | "evaluation" | "dataset_download";

export interface JobResponse {
  id: string;
  job_type: JobType;
  status: JobStatus;
  result: Record<string, unknown> | null;
  error_message: string | null;
  mlflow_run_id: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface TrainingRequest {
  model_name: string;
  dataset_name: string;
  num_epochs: number;
  batch_size: number;
  learning_rate: number;
  device: "cpu" | "cuda" | "mps";
  run_name?: string;
}

export interface EvaluationRequest {
  model_name: string;
  dataset_name: string;
  checkpoint_path?: string | null;
  device: "cpu" | "cuda" | "mps";
  max_samples?: number | null;
  evaluation_mode: "open_loop" | "closed_loop";
  simulator: "synthetic-reach";
  num_episodes: number;
  max_steps: number;
  action_dim: number;
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  expires_in_minutes: number;
  role: string;
}

export interface AuthUser {
  subject: string;
  role: string;
  auth_type: string;
}

export interface LeaderboardEntry {
  rank: number;
  model_name: string;
  dataset_name: string;
  composite_score: number;
  action_mse: number;
  latency_p95_ms: number;
  success_rate: number | null;
  num_samples: number;
  created_at: string;
}
