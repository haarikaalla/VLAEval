import { apiClient } from "@/api/client";
import type {
  DatasetInfo,
  EvaluationRequest,
  JobResponse,
  LeaderboardEntry,
  ModelInfo,
  AuthUser,
  TokenResponse,
  TrainingRequest,
} from "@/api/types";

export async function exchangeApiKey(apiKey: string): Promise<TokenResponse> {
  const { data } = await apiClient.post<TokenResponse>(
    "/auth/token",
    undefined,
    { headers: { "X-API-Key": apiKey } },
  );
  return data;
}

export async function fetchCurrentUser(): Promise<AuthUser> {
  const { data } = await apiClient.get<AuthUser>("/auth/me");
  return data;
}

export async function fetchDatasets(): Promise<DatasetInfo[]> {
  const { data } = await apiClient.get<DatasetInfo[]>("/datasets");
  return data;
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const { data } = await apiClient.get<ModelInfo[]>("/models");
  return data;
}

export async function fetchLeaderboard(datasetName?: string): Promise<LeaderboardEntry[]> {
  const { data } = await apiClient.get<LeaderboardEntry[]>("/leaderboard", {
    params: datasetName ? { dataset_name: datasetName } : undefined,
  });
  return data;
}

export async function submitTrainingJob(request: TrainingRequest): Promise<JobResponse> {
  const { data } = await apiClient.post<JobResponse>("/training/jobs", request);
  return data;
}

export async function fetchTrainingJob(jobId: string): Promise<JobResponse> {
  const { data } = await apiClient.get<JobResponse>(`/training/jobs/${jobId}`);
  return data;
}

export async function fetchTrainingJobs(): Promise<JobResponse[]> {
  const { data } = await apiClient.get<JobResponse[]>("/training/jobs");
  return data;
}

export async function submitEvaluationJob(request: EvaluationRequest): Promise<JobResponse> {
  const { data } = await apiClient.post<JobResponse>("/evaluation/jobs", request);
  return data;
}

export async function fetchEvaluationJobs(): Promise<JobResponse[]> {
  const { data } = await apiClient.get<JobResponse[]>("/evaluation/jobs");
  return data;
}

export async function downloadEvaluationReport(jobId: string): Promise<void> {
  const { data } = await apiClient.get<Blob>(`/evaluation/jobs/${jobId}/report`, {
    responseType: "blob",
  });
  const url = URL.createObjectURL(data);
  const link = document.createElement("a");
  link.href = url;
  link.download = `vla-eval-${jobId.slice(0, 8)}-report.md`;
  link.click();
  URL.revokeObjectURL(url);
}

export async function triggerDatasetDownload(name: string): Promise<{ message: string }> {
  const { data } = await apiClient.post<{ message: string }>("/datasets/download", { name });
  return data;
}
