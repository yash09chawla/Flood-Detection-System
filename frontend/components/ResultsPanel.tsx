"use client";

import { Download, Sparkles } from "lucide-react";
import { Card, CardHeader, CardTitle } from "./ui/Card";
import { StatusBadge } from "./StatusBadge";
import { EmptyState } from "./EmptyState";
import { FormatIcon, formatFromFilename } from "./FormatIcon";
import { cn, formatKm2, formatPercent } from "@/lib/utils";
import { fileUrl } from "@/lib/api";
import type { JobState } from "@/lib/types";

interface Props {
  job: JobState | null;
  landmaskOpacity: number;
  onLandmaskOpacityChange: (v: number) => void;
}

const DOWNLOADS: Array<{ key: keyof NonNullable<JobState["files"]>; label: string }> = [
  { key: "overlay_tif",        label: "Overlay map (with satellite)" },
  { key: "overlay_color_tif",  label: "Overlay color (transparent)" },
  { key: "class_tif",          label: "Class map (0/1/2)" },
  { key: "flood_tif",          label: "Flood mask" },
  { key: "perm_tif",           label: "Permanent water" },
  { key: "flood_shp_zip",      label: "Flood polygons" },
  { key: "perm_shp_zip",       label: "Permanent polygons" },
  { key: "png",                label: "Report visualization" },
  { key: "overlay_water_png",  label: "Water overlay (red/blue)" },
];

export function ResultsPanel({
  job,
  landmaskOpacity,
  onLandmaskOpacityChange,
}: Props) {
  // Empty state ─── no prediction yet
  if (!job) {
    return (
      <aside
        className="w-[360px] shrink-0 h-full overflow-y-auto scrollbar-thin
                    glass relative glass-sheen rounded-2xl
                    animate-slide-in-r flex items-center justify-center"
      >
        <EmptyState
          title="Awaiting prediction"
          description="Pick a region on the left and run the model to see the affected area, downloads, and a satellite-overlay map."
        />
      </aside>
    );
  }

  const { status, progress, message, stats, files, error } = job;

  return (
    <aside
      className="w-[360px] shrink-0 h-full overflow-y-auto scrollbar-thin
                  glass relative glass-sheen rounded-2xl animate-slide-in-r"
    >
      <div className="px-5 pt-5 pb-5 space-y-4">
        {/* HEADER */}
        <header className="flex items-center justify-between">
          <div>
            <p className="font-serif text-[15px] text-ink leading-none">Prediction</p>
            <p className="text-[10px] tabular text-subtle mt-1">
              job · {job.id.slice(0, 8)}
            </p>
          </div>
          <StatusBadge status={status} />
        </header>

        {/* RUNNING / PENDING */}
        {(status === "pending" || status === "running") && (
          <Card>
            <p className="text-[13px] text-ink leading-snug">{message || "Working…"}</p>
            <div className="mt-3 h-1 rounded-full bg-line/[0.08] overflow-hidden">
              <div
                className="h-full shimmer-bar transition-all duration-500 ease-out-expo"
                style={{ width: `${Math.max(2, progress * 100)}%` }}
              />
            </div>
            <p className="mt-2 text-[10px] tabular text-muted text-right">
              {Math.round(progress * 100)}%
            </p>
          </Card>
        )}

        {/* ERROR */}
        {status === "error" && (
          <Card className="border-danger/30 bg-danger/5">
            <CardTitle className="text-danger mb-1">Error</CardTitle>
            <p className="text-xs text-text whitespace-pre-wrap break-words">
              {error || message || "Prediction failed."}
            </p>
          </Card>
        )}

        {/* DONE */}
        {status === "done" && stats && (
          <>
            {/* KPI card */}
            <Card>
              <CardHeader>
                <CardTitle>Affected area</CardTitle>
                <Sparkles size={12} className="text-accent" />
              </CardHeader>
              <div className="grid grid-cols-2 gap-3">
                <KPI
                  label="Flood"
                  value={formatKm2(stats.flood_km2)}
                  swatchClass="bg-flood"
                  valueClass="text-flood"
                />
                <KPI
                  label="Permanent"
                  value={formatKm2(stats.permanent_km2)}
                  swatchClass="bg-permanent"
                  valueClass="text-permanent"
                />
              </div>
              <div className="h-px bg-line/[0.08] my-3" />
              <dl className="space-y-1.5 text-[12px]">
                <Row label="Total scene"  value={formatKm2(stats.total_km2)} />
                <Row label="Flood %"      value={formatPercent(stats.flood_pct)} />
                <Row label="Permanent %"  value={formatPercent(stats.permanent_pct)} />
                {stats.date && <Row label="Date" value={stats.date} mono />}
                {stats.n_tiles && stats.n_tiles > 1 && (
                  <Row label="Tiles" value={String(stats.n_tiles)} mono />
                )}
              </dl>
            </Card>

            {/* Opacity slider */}
            <Card>
              <CardHeader>
                <CardTitle>Land mask</CardTitle>
                <span className="text-[11px] tabular font-medium text-accent">
                  {Math.round(landmaskOpacity * 100)}%
                </span>
              </CardHeader>
              <input
                type="range"
                min={0}
                max={100}
                step={5}
                value={Math.round(landmaskOpacity * 100)}
                onChange={(e) => onLandmaskOpacityChange(Number(e.target.value) / 100)}
                className="slider-input"
                aria-label="Land mask opacity"
              />
              <div className="flex justify-between text-[10px] text-subtle mt-1.5">
                <span>Pure satellite</span>
                <span>Solid mask</span>
              </div>
              <p className="text-[11px] text-muted mt-3 leading-snug">
                Slider fades the gray non-water mask only — red flood &amp; blue
                permanent water always stay fully visible.
              </p>
            </Card>

            {/* Downloads */}
            <Card>
              <CardHeader>
                <CardTitle>Downloads</CardTitle>
                <span className="text-[10px] text-subtle">{Object.keys(files || {}).length} files</span>
              </CardHeader>
              <div className="flex flex-col gap-1 -mx-1">
                {DOWNLOADS.map(({ key, label }) => {
                  const filename = files?.[key];
                  const enabled = !!filename;
                  return (
                    <a
                      key={key}
                      href={enabled ? fileUrl(job.id, filename!) : undefined}
                      target={enabled ? "_blank" : undefined}
                      rel="noopener noreferrer"
                      download={enabled ? filename : undefined}
                      className={cn(
                        "flex items-center gap-2.5 px-2 py-2 rounded-md",
                        "text-[12px] transition-all duration-150",
                        enabled
                          ? "text-text hover:bg-line/[0.06] hover:text-ink cursor-pointer"
                          : "text-subtle cursor-not-allowed opacity-50",
                      )}
                    >
                      {enabled ? (
                        <FormatIcon format={formatFromFilename(filename!)} />
                      ) : (
                        <span className="inline-flex h-6 min-w-[36px] items-center justify-center
                                          rounded px-1.5 text-[10px] tracking-wider bg-line/[0.06] text-subtle">
                          —
                        </span>
                      )}
                      <span className="flex-1 truncate">{label}</span>
                      {enabled && <Download size={13} className="text-subtle shrink-0" />}
                    </a>
                  );
                })}
              </div>
            </Card>
          </>
        )}
      </div>
    </aside>
  );
}

/* ---------- helpers ---------- */
function KPI({
  label,
  value,
  swatchClass,
  valueClass,
}: {
  label: string;
  value: string;
  swatchClass: string;
  valueClass: string;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 text-[10px] tracking-label uppercase font-medium text-muted mb-1">
        <span className={cn("w-1.5 h-1.5 rounded-sm", swatchClass)} />
        {label}
      </div>
      <p className={cn("font-semibold tabular text-[22px] leading-none -tracking-[0.01em]", valueClass)}>
        {value}
      </p>
    </div>
  );
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <dt className="text-muted">{label}</dt>
      <dd className={cn("text-text tabular", mono && "font-mono text-[11px]")}>{value}</dd>
    </div>
  );
}
