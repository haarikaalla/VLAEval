import clsx from "clsx";
import type { JobStatus } from "@/api/types";

const STATUS_STYLES: Record<JobStatus, string> = {
  pending: "bg-amber-100 text-amber-800",
  running: "bg-blue-100 text-blue-800",
  succeeded: "bg-emerald-100 text-emerald-800",
  failed: "bg-red-100 text-red-800",
  cancelled: "bg-slate-100 text-slate-800",
};

export function JobStatusBadge({ status }: { status: JobStatus }) {
  return (
    <span
      className={clsx(
        "inline-block px-2 py-0.5 rounded-full text-xs font-medium capitalize",
        STATUS_STYLES[status],
      )}
    >
      {status}
    </span>
  );
}
