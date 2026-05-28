"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { ArrowRight, Upload, FileArchive, X } from "lucide-react";
import { Input } from "./ui/Input";
import { Button } from "./ui/Button";
import { todayISO, cn } from "@/lib/utils";

interface Props {
  onSubmit: (zip: File, date: string) => void;
  busy: boolean;
}

export function ShapefileTab({ onSubmit, busy }: Props) {
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [date,    setDate]    = useState(todayISO());
  const [err,     setErr]     = useState<string | null>(null);

  const onDrop = useCallback((accepted: File[]) => {
    setErr(null);
    const file = accepted[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".zip")) {
      setErr("Please upload a .zip containing the .shp/.shx/.dbf/.prj files.");
      return;
    }
    setZipFile(file);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: { "application/zip": [".zip"] },
  });

  const handle = () => {
    if (!zipFile) return;
    onSubmit(zipFile, date);
  };

  return (
    <div className="flex flex-col gap-3">
      {!zipFile ? (
        <div
          {...getRootProps()}
          className={cn(
            "rounded-lg border-2 border-dashed px-6 py-8 text-center cursor-pointer",
            "transition-all duration-200 ease-out-expo",
            isDragActive
              ? "border-accent bg-accent-soft/70 scale-[1.01]"
              : "border-border bg-canvas/50 hover:border-accent/40 hover:bg-accent-soft/30",
          )}
        >
          <input {...getInputProps()} />
          <div className="mx-auto mb-3 inline-flex items-center justify-center
                          w-10 h-10 rounded-lg bg-accent/10 text-accent">
            <Upload size={18} strokeWidth={1.75} />
          </div>
          <p className="text-[13px] text-ink font-medium leading-tight">
            {isDragActive ? "Drop your .zip here" : "Drag a zipped shapefile"}
          </p>
          <p className="text-[11px] text-muted mt-1">
            or click to browse — must contain <span className="font-mono">.shp .shx .dbf .prj</span>
          </p>
        </div>
      ) : (
        <div className="rounded-lg border border-border bg-canvas/60 px-3.5 py-3
                         flex items-center gap-3 group">
          <div className="w-9 h-9 rounded-md bg-warn/10 text-warn grid place-items-center shrink-0">
            <FileArchive size={16} strokeWidth={1.75} />
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-[13px] font-medium text-ink truncate">{zipFile.name}</p>
            <p className="text-[11px] text-muted tabular">
              {(zipFile.size / 1024).toFixed(1)} KB · ready to upload
            </p>
          </div>
          <button
            type="button"
            onClick={() => setZipFile(null)}
            className="w-7 h-7 rounded-md text-subtle hover:text-danger hover:bg-danger/10
                       transition-colors duration-150 grid place-items-center cursor-pointer"
            aria-label="Remove file"
          >
            <X size={14} />
          </button>
        </div>
      )}

      {err && (
        <p className="text-xs text-danger leading-snug bg-danger/5 border border-danger/20
                       rounded-md px-3 py-2">
          {err}
        </p>
      )}

      <Input
        label="Flood date"
        type="date"
        value={date}
        onChange={(e) => setDate(e.target.value)}
      />

      <Button onClick={handle} disabled={busy || !zipFile} fullWidth size="lg">
        {busy ? "Predicting…" : <>Predict on shapefile <ArrowRight size={14} /></>}
      </Button>
    </div>
  );
}
