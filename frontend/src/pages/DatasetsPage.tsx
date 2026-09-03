import { useQuery } from "@tanstack/react-query";
import { fetchDatasets } from "@/api/endpoints";
import { ErrorBanner, LoadingSpinner } from "@/components/Feedback";

export function DatasetsPage() {
  const { data, isLoading, isError } = useQuery({ queryKey: ["datasets"], queryFn: fetchDatasets });

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">Registered Datasets</h2>

      {isLoading && <LoadingSpinner label="Loading datasets..." />}
      {isError && <ErrorBanner message="Failed to load datasets." />}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {data?.map((d) => (
          <div key={d.name} className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-brand-700">{d.name}</h3>
              <span className="text-xs text-slate-400">{d.license}</span>
            </div>
            <p className="text-sm text-slate-600 mt-1">{d.description}</p>
            <div className="flex flex-wrap gap-1 mt-3">
              {d.tags.map((tag) => (
                <span key={tag} className="text-xs bg-brand-50 text-brand-700 px-2 py-0.5 rounded-full">
                  {tag}
                </span>
              ))}
            </div>
            <div className="mt-3 text-xs text-slate-500 space-y-0.5">
              <p>Hub repo: {d.hub_repo_id}</p>
              <p>Task type: {d.task_type}</p>
              <p>Action dim: {d.action_dim}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
