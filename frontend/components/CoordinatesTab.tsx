"use client";

import { useState } from "react";
import { ArrowRight } from "lucide-react";
import { Input } from "./ui/Input";
import { Button } from "./ui/Button";
import { todayISO } from "@/lib/utils";
import type { CoordinatesRequest } from "@/lib/types";

interface Props {
  onSubmit: (req: CoordinatesRequest) => void;
  busy: boolean;
}

export function CoordinatesTab({ onSubmit, busy }: Props) {
  const [lonMin, setLonMin] = useState("-66.0");
  const [latMin, setLatMin] = useState("-13.7");
  const [lonMax, setLonMax] = useState("-65.95");
  const [latMax, setLatMax] = useState("-13.65");
  const [date,   setDate]   = useState(todayISO());
  const [err,    setErr]    = useState<string | null>(null);

  const handle = () => {
    setErr(null);
    const nums = [lonMin, latMin, lonMax, latMax].map((s) => parseFloat(s));
    if (nums.some((n) => Number.isNaN(n))) {
      setErr("All coordinate fields must be numbers.");
      return;
    }
    if (nums[0] >= nums[2] || nums[1] >= nums[3]) {
      setErr("min must be less than max for both longitude and latitude.");
      return;
    }
    onSubmit({
      lon_min: nums[0], lat_min: nums[1],
      lon_max: nums[2], lat_max: nums[3],
      date,
    });
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 gap-2">
        <Input mono label="Lon min" value={lonMin} onChange={(e) => setLonMin(e.target.value)} />
        <Input mono label="Lon max" value={lonMax} onChange={(e) => setLonMax(e.target.value)} />
        <Input mono label="Lat min" value={latMin} onChange={(e) => setLatMin(e.target.value)} />
        <Input mono label="Lat max" value={latMax} onChange={(e) => setLatMax(e.target.value)} />
      </div>
      <Input
        label="Flood date"
        type="date"
        value={date}
        onChange={(e) => setDate(e.target.value)}
        hint="Sentinel-1 looked up within ±6 days of this date."
      />
      {err && (
        <p className="text-xs text-danger leading-snug bg-danger/5 border border-danger/20
                       rounded-md px-3 py-2">
          {err}
        </p>
      )}
      <Button onClick={handle} disabled={busy} fullWidth size="lg">
        {busy ? "Predicting…" : <>Predict floods <ArrowRight size={14} /></>}
      </Button>
    </div>
  );
}
