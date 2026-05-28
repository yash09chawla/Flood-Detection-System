// Mirrors the FastAPI response schemas in flood-detection-src/api.py +
// JobState in flood-detection-src/job_runner.py

export type JobStatus = "pending" | "running" | "done" | "error";

export interface PredictionStats {
  flood_km2: number;
  permanent_km2: number;
  total_km2: number;
  flood_pct: number;
  permanent_pct: number;
  bbox?: [number, number, number, number]; // [lon_min, lat_min, lon_max, lat_max]
  date?: string;
  bbox_size_km?: [number, number];
  source_crs?: string;
  n_tiles?: number;
  has_satellite_overlay?: boolean;
}

export interface JobFiles {
  class_tif?: string;             // single-band uint8 0/1/2
  flood_tif?: string;             // binary flood mask
  perm_tif?: string;              // binary permanent water mask
  overlay_tif?: string;           // RGB GeoTIFF: S2 base + red/blue burned in
  overlay_color_tif?: string;     // RGBA TIF: transparent non-water, only red/blue
  overlay_png?: string;           // legacy combined overlay
  overlay_water_png?: string;     // RGBA: opaque red/blue ONLY (constant opacity)
  overlay_landmask_png?: string;  // RGBA: gray non-water ONLY (slider-controlled)
  png?: string;                   // "report style" RGB PNG (red/blue/gray)
  flood_shp_zip?: string;
  perm_shp_zip?: string;
  // Filenames are basenames; the URL is f"{API_BASE}/api/files/{job_id}/{filename}"
}

export interface JobState {
  id: string;
  status: JobStatus;
  progress: number; // 0..1
  message: string;
  stats: PredictionStats | null;
  files: JobFiles | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface CoordinatesRequest {
  lon_min: number;
  lat_min: number;
  lon_max: number;
  lat_max: number;
  date: string; // YYYY-MM-DD
}

export interface HealthResponse {
  ok: boolean;
  model_loaded: boolean;
  gee_initialized: boolean;
  checkpoint: string;
  init_error: string | null;
}

export type Bbox = [number, number, number, number]; // [west, south, east, north]
