"use client";

import { cn } from "@/lib/utils";
import type { JobStatus } from "@/lib/types";

const styles: Record<JobStatus, { dot: string; text: string; bg: string; label: string }> = {
  pending: { dot: "bg-muted",     text: "text-muted",     bg: "bg-line/[0.06]",     label: "Queued"  },
  running: { dot: "bg-warn",      text: "text-warn",      bg: "bg-warn/10",         label: "Running" },
  done:    { dot: "bg-success",   text: "text-success",   bg: "bg-success/10",      label: "Done"    },
  error:   { dot: "bg-danger",    text: "text-danger",    bg: "bg-danger/10",       label: "Error"   },
};

export function StatusBadge({ status }: { status: JobStatus }) {
  const s = styles[status];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full",
        "text-[10px] font-semibold tracking-label uppercase",
        s.bg, s.text,
      )}
    >
      <span
        className={cn(
          "w-1.5 h-1.5 rounded-full",
          status === "running" && "animate-pulse-dot",
          s.dot,
        )}
      />
      {s.label}
    </span>
  );
}
