import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchDatasets, fetchLeaderboard } from "@/api/endpoints";
import { ErrorBanner, LoadingSpinner } from "@/components/Feedback";

export function LeaderboardPage() {
  const [datasetFilter, setDatasetFilter] = useState<string>("");

  const datasetsQuery = useQuery({ queryKey: ["datasets"], queryFn: fetchDatasets });
  const leaderboardQuery = useQuery({
    queryKey: ["leaderboard", datasetFilter],
    queryFn: () => fetchLeaderboard(datasetFilter || undefined),
    refetchInterval: 15000,
  });

  const chartData = useMemo(
    () =>
      (leaderboardQuery.data ?? []).map((entry) => ({
        name: entry.model_name,
        score: entry.composite_score,
      })),
    [leaderboardQuery.data],
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold">Leaderboard</h2>
        <select
          className="border border-slate-300 rounded-md px-3 py-1.5 text-sm bg-white"
          value={datasetFilter}
          onChange={(e) => setDatasetFilter(e.target.value)}
        >
          <option value="">All datasets</option>
          {datasetsQuery.data?.map((d) => (
            <option key={d.name} value={d.name}>
              {d.name}
            </option>
          ))}
        </select>
      </div>

      {leaderboardQuery.isLoading && <LoadingSpinner label="Loading leaderboard..." />}
      {leaderboardQuery.isError && (
        <ErrorBanner message="Failed to load leaderboard. Is the API running?" />
      )}

      {leaderboardQuery.data && leaderboardQuery.data.length > 0 && (
        <>
          <div className="bg-white rounded-lg shadow p-4 h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Bar dataKey="score" fill="#2563eb" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-white rounded-lg shadow overflow-hidden">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-100 text-slate-600">
                <tr>
                  <th className="px-4 py-2 text-left">Rank</th>
                  <th className="px-4 py-2 text-left">Model</th>
                  <th className="px-4 py-2 text-left">Dataset</th>
                  <th className="px-4 py-2 text-right">Score</th>
                  <th className="px-4 py-2 text-right">Action MSE</th>
                  <th className="px-4 py-2 text-right">p95 Latency (ms)</th>
                  <th className="px-4 py-2 text-right">Success Rate</th>
                </tr>
              </thead>
              <tbody>
                {leaderboardQuery.data.map((entry) => (
                  <tr key={`${entry.model_name}-${entry.dataset_name}-${entry.rank}`} className="border-t">
                    <td className="px-4 py-2 font-medium">{entry.rank}</td>
                    <td className="px-4 py-2">{entry.model_name}</td>
                    <td className="px-4 py-2 text-slate-500">{entry.dataset_name}</td>
                    <td className="px-4 py-2 text-right font-semibold">{entry.composite_score.toFixed(2)}</td>
                    <td className="px-4 py-2 text-right">{entry.action_mse.toFixed(4)}</td>
                    <td className="px-4 py-2 text-right">{entry.latency_p95_ms.toFixed(2)}</td>
                    <td className="px-4 py-2 text-right">
                      {entry.success_rate !== null ? `${(entry.success_rate * 100).toFixed(1)}%` : "N/A"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {leaderboardQuery.data && leaderboardQuery.data.length === 0 && (
        <p className="text-slate-500 text-sm">
          No benchmark results yet. Submit a training + evaluation job to populate the leaderboard.
        </p>
      )}
    </div>
  );
}
