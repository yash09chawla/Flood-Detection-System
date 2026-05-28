"use client";

import { useState } from "react";
import { Crosshair, Square, FileUp } from "lucide-react";
import { Tabs } from "./ui/Tabs";
import { CoordinatesTab } from "./CoordinatesTab";
import { TileTab } from "./TileTab";
import { ShapefileTab } from "./ShapefileTab";
import { BrandMark } from "./BrandMark";
import { ThemeToggle } from "./ThemeToggle";
import type { Bbox, CoordinatesRequest } from "@/lib/types";

type TabValue = "coords" | "tile" | "shapefile";

interface Props {
  drawnBbox: Bbox | null;
  onPredictCoordinates: (req: CoordinatesRequest) => void;
  onPredictShapefile:   (zip: File, date: string) => void;
  onActiveTabChange?:   (tab: TabValue) => void;
  busy: boolean;
}

export function InputPanel({
  drawnBbox,
  onPredictCoordinates,
  onPredictShapefile,
  onActiveTabChange,
  busy,
}: Props) {
  const [tab, setTab] = useState<TabValue>("coords");

  const handleTabChange = (next: TabValue) => {
    setTab(next);
    onActiveTabChange?.(next);
  };

  return (
    <aside
      className="w-[360px] shrink-0 h-full overflow-y-auto scrollbar-thin
                 glass relative glass-sheen rounded-2xl
                 animate-slide-in-l"
    >
      {/* HEADER ─── brand + theme toggle */}
      <header className="flex items-center justify-between gap-2 px-5 pt-5 pb-4">
        <BrandMark />
        <ThemeToggle />
      </header>

      <div className="px-5 pb-5 space-y-4">
        <p className="text-xs leading-relaxed text-muted">
          Predict <span className="text-flood font-medium">flood</span> vs{" "}
          <span className="text-permanent font-medium">permanent water</span>{" "}
          on Sentinel-1 imagery, anywhere in the world.
        </p>

        <div className="h-px bg-line/[0.08]" />

        {/* Mode picker */}
        <div>
          <p className="text-[10px] font-medium tracking-label uppercase text-muted mb-2">
            Region input
          </p>
          <Tabs<TabValue>
            value={tab}
            onChange={handleTabChange}
            options={[
              { value: "coords",    label: "Coords",  icon: <Crosshair size={12} strokeWidth={1.75} /> },
              { value: "tile",      label: "Tile",    icon: <Square    size={12} strokeWidth={1.75} /> },
              { value: "shapefile", label: "Shape",   icon: <FileUp    size={12} strokeWidth={1.75} /> },
            ]}
            className="w-full justify-between"
          />
        </div>

        {/* Active tab body */}
        <div className="animate-fade-in" key={tab}>
          {tab === "coords" && (
            <CoordinatesTab onSubmit={onPredictCoordinates} busy={busy} />
          )}
          {tab === "tile" && (
            <TileTab drawnBbox={drawnBbox} onSubmit={onPredictCoordinates} busy={busy} />
          )}
          {tab === "shapefile" && (
            <ShapefileTab onSubmit={onPredictShapefile} busy={busy} />
          )}
        </div>

        <div className="h-px bg-line/[0.08]" />

        {/* FOOTER ─── tech credit */}
        <footer className="text-[11px] text-subtle leading-relaxed">
          ResNet50 + UNet++ trained on Sen1Floods11.<br />
          Inference on local FastAPI · Earth Engine fetch.
        </footer>
      </div>
    </aside>
  );
}

export type { TabValue };
