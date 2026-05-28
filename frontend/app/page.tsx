"use client";

import { useState, useCallback, useMemo } from "react";
import { MapView } from "@/components/MapView";
import { InputPanel, type TabValue } from "@/components/InputPanel";
import { ResultsPanel } from "@/components/ResultsPanel";
import {
  predictCoordinates,
  predictShapefile,
  pollJob,
  fileUrl,
} from "@/lib/api";
import type {
  Bbox,
  CoordinatesRequest,
  JobState,
} from "@/lib/types";

export default function HomePage() {
  const [job, setJob]               = useState<JobState | null>(null);
  const [drawnBbox, setDrawnBbox]   = useState<Bbox | null>(null);
  const [activeTab, setActiveTab]   = useState<TabValue>("coords");
  const [landmaskOpacity, setLandmaskOpacity] = useState(0.6);

  const busy = job?.status === "pending" || job?.status === "running";

  const startCoordinates = useCallback(async (req: CoordinatesRequest) => {
    try {
      const { job_id } = await predictCoordinates(req);
      const initial: JobState = {
        id: job_id,
        status: "pending",
        progress: 0,
        message: "Queued…",
        stats: null,
        files: null,
        error: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      setJob(initial);
      await pollJob(job_id, (state) => setJob(state));
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setJob((prev) => prev ? {
        ...prev,
        status: "error",
        error: msg,
        message: msg,
      } : null);
    }
  }, []);

  const startShapefile = useCallback(async (zip: File, date: string) => {
    try {
      const { job_id } = await predictShapefile(zip, date);
      const initial: JobState = {
        id: job_id,
        status: "pending",
        progress: 0,
        message: "Queued…",
        stats: null,
        files: null,
        error: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      setJob(initial);
      await pollJob(job_id, (state) => setJob(state));
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setJob((prev) => prev ? {
        ...prev,
        status: "error",
        error: msg,
        message: msg,
      } : null);
    }
  }, []);

  // Two-layer overlay: water (red/blue, constant) + landmask (gray, slider-controlled).
  // Falls back to the combined overlay_png if backend doesn't ship the split layers.
  const overlay = useMemo(() => {
    if (job?.status !== "done") return null;
    if (!job.stats?.bbox) return null;
    const f = job.files;
    if (!f) return null;
    const waterFile    = f.overlay_water_png || f.overlay_png || f.png;
    const landmaskFile = f.overlay_landmask_png || f.overlay_water_png || f.png;
    if (!waterFile || !landmaskFile) return null;
    return {
      waterUrl:    fileUrl(job.id, waterFile),
      landmaskUrl: fileUrl(job.id, landmaskFile),
      bbox:        job.stats.bbox,
    };
  }, [job?.status, job?.id, job?.files, job?.stats?.bbox]);

  return (
    <main className="relative h-screen w-screen overflow-hidden bg-canvas">
      {/* Map fills the viewport */}
      <div className="absolute inset-0">
        <MapView
          drawingEnabled={activeTab === "tile"}
          onBboxDrawn={setDrawnBbox}
          predictionOverlay={overlay}
          landmaskOpacity={landmaskOpacity}
        />
      </div>

      {/* Floating left panel */}
      <div className="absolute top-4 left-4 bottom-4 z-10 flex">
        <InputPanel
          drawnBbox={drawnBbox}
          onPredictCoordinates={startCoordinates}
          onPredictShapefile={startShapefile}
          onActiveTabChange={setActiveTab}
          busy={!!busy}
        />
      </div>

      {/* Floating right panel */}
      <div className="absolute top-4 right-4 bottom-4 z-10 flex">
        <ResultsPanel
          job={job}
          landmaskOpacity={landmaskOpacity}
          onLandmaskOpacityChange={setLandmaskOpacity}
        />
      </div>
    </main>
  );
}
