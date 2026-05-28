"use client";

import { useState } from "react";
import { ArrowRight, MousePointerClick } from "lucide-react";
import { Input } from "./ui/Input";
import { Button } from "./ui/Button";
import { todayISO } from "@/lib/utils";
import type { Bbox, CoordinatesRequest } from "@/lib/types";

interface Props {
  drawnBbox: Bbox | null;
  onSubmit: (req: CoordinatesRequest) => void;
  busy: boolean;
}

export function TileTab({ drawnBbox, onSubmit, busy }: Props) {
  const [date, setDate] = useState(todayISO());

  const handle = () => {
    if (!drawnBbox) return;
    const [lon_min, lat_min, lon_max, lat_max] = drawnBbox;
    onSubmit({ lon_min, lat_min, lon_max, lat_max, date });
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="rounded-lg bg-accent-soft/60 border border-accent/15 px-3.5 py-3
                       flex gap-2.5 items-start">
        <MousePointerClick size={14} className="text-accent shrink-0 mt-0.5" strokeWidth={1.75} />
        <p className="text-[12px] leading-relaxed text-text">
          Click two corners on the map to draw a rectangle. Click again to redraw.
        </p>
      </div>

      <div className="rounded-lg border border-border bg-canvas/50 px-3.5 py-3">
        <p className="text-[10px] font-medium tracking-label uppercase text-muted mb-1.5">
          Selected bbox
        </p>
        {drawnBbox ? (
          <div className="font-mono text-[11px] tabular leading-relaxed text-ink space-y-0.5">
            <div className="flex gap-2"><span className="text-muted w-7">lon</span>{drawnBbox[0].toFixed(4)}<span className="text-subtle">→</span>{drawnBbox[2].toFixed(4)}</div>
            <div className="flex gap-2"><span className="text-muted w-7">lat</span>{drawnBbox[1].toFixed(4)}<span className="text-subtle">→</span>{drawnBbox[3].toFixed(4)}</div>
          </div>
        ) : (
          <p className="text-xs text-subtle">No selection yet.</p>
        )}
      </div>

      <Input
        label="Flood date"
        type="date"
        value={date}
        onChange={(e) => setDate(e.target.value)}
      />

      <Button onClick={handle} disabled={busy || !drawnBbox} fullWidth size="lg">
        {busy ? "Predicting…" : <>Predict on selection <ArrowRight size={14} /></>}
      </Button>
    </div>
  );
}
