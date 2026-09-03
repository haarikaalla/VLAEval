import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchEvaluationJobs, fetchModels, submitEvaluationJob } from "@/api/endpoints";
import type { EvaluationRequest } from "@/api/types";
import { ErrorBanner, LoadingSpinner } from "@/components/Feedback";
import { JobStatusBadge } from "@/components/JobStatusBadge";

const DEFAULT_FORM: EvaluationRequest = {
  model_name: "baseline-cnn",
  dataset_name: "synthetic",
  device: "cpu",
  max_samples: 100,
};

export function EvaluationPage() {
  const [form, setForm] = useState<EvaluationRequest>(DEFAULT_FORM);
  const queryClient = useQueryClient();

  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: fetchModels });
  const jobsQuery = useQuery({
    queryKey: ["evaluation-jobs"],
    queryFn: fetchEvaluationJobs,
    refetchInterval: 5000,
  });

  const submitMutation = useMutation({
    mutationFn: submitEvaluationJob,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["evaluation-jobs"] }),
  });

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Evaluation Jobs</h2>

      <form
        className="bg-white rounded-lg shadow p-4 grid grid-cols-2 md:grid-cols-4 gap-4"
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
          Max samples
          <input
            type="number"
            className="mt-1 w-full border border-slate-300 rounded-md px-2 py-1.5"
            value={form.max_samples ?? ""}
            onChange={(e) => setForm((f) => ({ ...f, max_samples: Number(e.target.value) }))}
          />
        </label>
        <label className="text-sm">
          Device
          <select
            className="mt-1 w-full border border-slate-300 rounded-md px-2 py-1.5"
            value={form.device}
            onChange={(e) => setForm((f) => ({ ...f, device: e.target.value as EvaluationRequest["device"] }))}
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
            {submitMutation.isPending ? "Submitting..." : "Submit evaluation job"}
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
                <th className="px-4 py-2 text-left">Score</th>
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
                  <td className="px-4 py-2">
                    {job.result && typeof job.result.composite_score === "number"
                      ? job.result.composite_score.toFixed(2)
                      : "-"}
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
