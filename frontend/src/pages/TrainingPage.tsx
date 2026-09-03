import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchModels, fetchTrainingJobs, submitTrainingJob } from "@/api/endpoints";
import type { TrainingRequest } from "@/api/types";
import { ErrorBanner, LoadingSpinner } from "@/components/Feedback";
import { JobStatusBadge } from "@/components/JobStatusBadge";

const DEFAULT_FORM: TrainingRequest = {
  model_name: "baseline-cnn",
  dataset_name: "synthetic",
  num_epochs: 10,
  batch_size: 32,
  learning_rate: 1e-4,
  device: "cpu",
};

export function TrainingPage() {
  const [form, setForm] = useState<TrainingRequest>(DEFAULT_FORM);
  const queryClient = useQueryClient();

  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: fetchModels });
  const jobsQuery = useQuery({
    queryKey: ["training-jobs"],
    queryFn: fetchTrainingJobs,
    refetchInterval: 5000,
  });

  const submitMutation = useMutation({
    mutationFn: submitTrainingJob,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["training-jobs"] }),
  });

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Training Jobs</h2>

      <form
        className="bg-white rounded-lg shadow p-4 grid grid-cols-2 md:grid-cols-3 gap-4"
        onSubmit={(e) => {
          e.preventDefault();
          submitMutation.mutate(form);
        }}
      >
        <label className="text-sm">
          Model
          <select
            className="mt-1 w-full border border-slate-300 rounded-md px-2 py-1.5"
            value={form.model_name}
            onChange={(e) => setForm((f) => ({ ...f, model_name: e.target.value }))}
          >
            {modelsQuery.data?.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
        </label>
        <label className="text-sm">
          Dataset
          <input
            className="mt-1 w-full border border-slate-300 rounded-md px-2 py-1.5"
            value={form.dataset_name}
            onChange={(e) => setForm((f) => ({ ...f, dataset_name: e.target.value }))}
          />
        </label>
        <label className="text-sm">
          Epochs
          <input
            type="number"
            className="mt-1 w-full border border-slate-300 rounded-md px-2 py-1.5"
            value={form.num_epochs}
            onChange={(e) => setForm((f) => ({ ...f, num_epochs: Number(e.target.value) }))}
          />
        </label>
        <label className="text-sm">
          Batch size
          <input
            type="number"
            className="mt-1 w-full border border-slate-300 rounded-md px-2 py-1.5"
            value={form.batch_size}
            onChange={(e) => setForm((f) => ({ ...f, batch_size: Number(e.target.value) }))}
          />
        </label>
        <label className="text-sm">
          Learning rate
          <input
            type="number"
            step="0.00001"
            className="mt-1 w-full border border-slate-300 rounded-md px-2 py-1.5"
            value={form.learning_rate}
            onChange={(e) => setForm((f) => ({ ...f, learning_rate: Number(e.target.value) }))}
          />
        </label>
        <label className="text-sm">
          Device
          <select
            className="mt-1 w-full border border-slate-300 rounded-md px-2 py-1.5"
            value={form.device}
            onChange={(e) => setForm((f) => ({ ...f, device: e.target.value as TrainingRequest["device"] }))}
          >
            <option value="cpu">cpu</option>
            <option value="cuda">cuda</option>
            <option value="mps">mps</option>
          </select>
        </label>
        <div className="col-span-full flex items-center gap-3">
          <button
            type="submit"
            disabled={submitMutation.isPending}
            className="bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white text-sm font-medium px-4 py-2 rounded-md"
          >
            {submitMutation.isPending ? "Submitting..." : "Submit training job"}
          </button>
          {submitMutation.isError && <ErrorBanner message="Failed to submit job. Check your API key." />}
        </div>
      </form>

      {jobsQuery.isLoading && <LoadingSpinner label="Loading jobs..." />}
      {jobsQuery.data && (
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-100 text-slate-600">
              <tr>
                <th className="px-4 py-2 text-left">Job ID</th>
                <th className="px-4 py-2 text-left">Status</th>
                <th className="px-4 py-2 text-left">Created</th>
              </tr>
            </thead>
            <tbody>
              {jobsQuery.data.map((job) => (
                <tr key={job.id} className="border-t">
                  <td className="px-4 py-2 font-mono text-xs">{job.id}</td>
                  <td className="px-4 py-2">
                    <JobStatusBadge status={job.status} />
                  </td>
                  <td className="px-4 py-2 text-slate-500">{new Date(job.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
