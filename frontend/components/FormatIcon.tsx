"use client";

import { cn } from "@/lib/utils";

type Format = "tif" | "shp" | "png" | "zip" | "json";

interface Props {
  format: Format;
  className?: string;
}

// Tiny rectangular tag with the file format. Color hints at what it is.
const STYLES: Record<Format, { label: string; color: string }> = {
  tif:  { label: "TIF",  color: "bg-permanent/10  text-permanent  ring-permanent/30" },
  shp:  { label: "SHP",  color: "bg-flood/10      text-flood      ring-flood/30"     },
  png:  { label: "PNG",  color: "bg-success/10    text-success    ring-success/30"   },
  zip:  { label: "ZIP",  color: "bg-warn/10       text-warn       ring-warn/30"      },
  json: { label: "JSON", color: "bg-accent/10     text-accent     ring-accent/30"    },
};

export function FormatIcon({ format, className }: Props) {
  const s = STYLES[format];
  return (
    <span
      className={cn(
        "inline-flex h-6 min-w-[36px] items-center justify-center rounded px-1.5",
        "text-[10px] font-semibold tracking-wider tabular ring-1",
        s.color,
        className,
      )}
    >
      {s.label}
    </span>
  );
}

export function formatFromFilename(name: string): Format {
  const lower = name.toLowerCase();
  if (lower.endsWith(".tif") || lower.endsWith(".tiff")) return "tif";
  if (lower.endsWith(".shp")) return "shp";
  if (lower.endsWith(".png")) return "png";
  if (lower.endsWith(".zip")) return "zip";
  if (lower.endsWith(".json") || lower.endsWith(".geojson")) return "json";
  return "zip";
}
