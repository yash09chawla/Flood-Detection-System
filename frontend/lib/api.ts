// HTTP client for the FastAPI backend.
// Base URL comes from NEXT_PUBLIC_API_BASE — set in .env.local to your ngrok URL.

import type {
  CoordinatesRequest,
  HealthResponse,
  JobState,
} from "./types";

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000").replace(/\/$/, "");

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || body.error || JSON.stringify(body);
    } catch {
      /* ignore — keep statusText */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(): Promise<HealthResponse> {
  return jsonFetch<HealthResponse>("/api/health");
}

export async function predictCoordinates(req: CoordinatesRequest): Promise<{ job_id: string }> {
  return jsonFetch<{ job_id: string }>("/api/predict/coordinates", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function predictShapefile(zipFile: File, date: string): Promise<{ job_id: string }> {
  const fd = new FormData();
  fd.append("shp_zip", zipFile, zipFile.name);
  fd.append("date", date);

  const res = await fetch(`${API_BASE}/api/predict/shapefile`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json();
}

export async function getJob(jobId: string): Promise<JobState> {
  return jsonFetch<JobState>(`/api/jobs/${jobId}`);
}

/**
 * Polls a job every `intervalMs` until it reaches a terminal status (done/error)
 * or `timeoutMs` elapses. Calls `onUpdate` after each poll for live UI updates.
 */
export async function pollJob(
  jobId: string,
  onUpdate: (state: JobState) => void,
  intervalMs = 2000,
  timeoutMs = 15 * 60 * 1000, // 15 min
): Promise<JobState> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const state = await getJob(jobId);
    onUpdate(state);
    if (state.status === "done" || state.status === "error") {
      return state;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(`Polling timed out after ${timeoutMs / 1000}s`);
}

export function fileUrl(jobId: string, filename: string): string {
  return `${API_BASE}/api/files/${jobId}/${encodeURIComponent(filename)}`;
}

export { API_BASE, ApiError };
