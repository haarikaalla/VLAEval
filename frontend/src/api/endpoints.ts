import { apiClient } from "@/api/client";
import type {
  DatasetInfo,
  EvaluationRequest,
  JobResponse,
  LeaderboardEntry,
  ModelInfo,
  TrainingRequest,
} from "@/api/types";

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

export async function triggerDatasetDownload(name: string): Promise<{ message: string }> {
  const { data } = await apiClient.post<{ message: string }>("/datasets/download", { name });
  return data;
}
