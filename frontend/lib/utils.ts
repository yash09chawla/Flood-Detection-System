import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatKm2(km2: number | undefined | null): string {
  if (km2 == null || isNaN(km2)) return "—";
  if (km2 < 0.01) return "<0.01 km²";
  if (km2 < 100) return `${km2.toFixed(2)} km²`;
  return `${km2.toFixed(0)} km²`;
}

export function formatPercent(pct: number | undefined | null): string {
  if (pct == null || isNaN(pct)) return "—";
  return `${pct.toFixed(2)}%`;
}

export function todayISO(): string {
  return new Date().toISOString().slice(0, 10);
}
