"""
Asynchronous job orchestration for the flood-prediction web API.

A "job" wraps one prediction run end-to-end:
    GEE fetch → 3-class prediction → area calc → vector export → file paths

Status is held in process memory (no Redis) — fine for a single-user demo.
The FastAPI endpoint creates a JobState, kicks off a BackgroundTask, and the
frontend polls /api/jobs/{id} until status == "done" (or "error").
"""

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(__file__))

from inference         import (FloodPredictor, sliding_window_predict, postprocess,
                                visualize_flood_map, save_geotiff, load_and_normalize,
                                make_overlay_png, make_overlay_water_png,
                                make_overlay_landmask_png,
                                make_overlay_tif, make_overlay_color_tif)
from area_calculator   import compute_area_km2, compute_total_extent_km2
from raster_to_vector  import class_map_to_shapefiles
from shapefile_handler import (read_shapefile_zip, clip_raster_to_polygon,
                                bbox_size_km)


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

@dataclass
class JobState:
    id:          str
    status:      str  = "pending"   # pending | running | done | error
    progress:    float = 0.0        # 0..1
    message:     str  = ""
    stats:       dict | None = None
    files:       dict | None = None
    error:       str  | None = None
    created_at:  str  = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at:  str  = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":         self.id,
            "status":     self.status,
            "progress":   self.progress,
            "message":    self.message,
            "stats":      self.stats,
            "files":      self.files,
            "error":      self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# Process-wide registry. Wrapped as a class so api.py can import it cleanly.
class JobRegistry:
    def __init__(self):
        self._jobs: dict[str, JobState] = {}

    def create(self) -> JobState:
        job_id = str(uuid4())
        state = JobState(id=job_id)
        self._jobs[job_id] = state
        return state

    def get(self, job_id: str) -> JobState | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        return [j.to_dict() for j in self._jobs.values()]


JOBS = JobRegistry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update(job: JobState, *, status: str | None = None, progress: float | None = None,
            message: str | None = None):
    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = max(0.0, min(1.0, progress))
    if message is not None:
        job.message = message
    job.updated_at = datetime.now(timezone.utc).isoformat()


def _public_files(job_id: str, out_dir: str, written: dict) -> dict:
    """
    Convert local file paths to {key: filename} dicts the API serves via
    /api/files/{job_id}/{filename}. Only file BASENAMES are returned to the
    frontend, never absolute paths.
    """
    return {key: os.path.basename(path) for key, path in written.items()
            if path and os.path.exists(path)}


def _build_outputs_for_class_map(class_map: np.ndarray,
                                  flood_mask: np.ndarray,
                                  perm_mask: np.ndarray,
                                  profile: dict,
                                  out_dir: str,
                                  base_name: str,
                                  sentinel2_path: str | None = None) -> tuple[dict, dict]:
    """
    Persist all GeoTIFFs + PNGs + zipped shapefiles, compute area statistics,
    return (stats, files_written).

    Outputs (relative to out_dir):
      - <base>_class_map.tif        : single-band uint8 0/1/2
      - <base>_flood_mask.tif       : binary flood
      - <base>_permanent_water.tif  : binary permanent water
      - <base>_flood_map.png        : "report style" RGB PNG (red/blue/gray)
      - <base>_overlay.png          : RGBA PNG, transparent non-water — for map
      - <base>_overlay.tif          : RGB GeoTIFF, S2 base + red/blue burned in
      - <base>_flood_polygons.zip   : flood polygons as zipped shapefile
      - <base>_permanent_polygons.zip
    """
    profile_out = profile.copy()
    profile_out.update({"driver": "GTiff", "count": 1, "dtype": "uint8"})
    profile_out.pop("nodata", None)

    written = {}
    written["class_tif"]         = os.path.join(out_dir, f"{base_name}_class_map.tif")
    written["flood_tif"]         = os.path.join(out_dir, f"{base_name}_flood_mask.tif")
    written["perm_tif"]          = os.path.join(out_dir, f"{base_name}_permanent_water.tif")
    written["png"]               = os.path.join(out_dir, f"{base_name}_flood_map.png")
    written["overlay_water_png"] = os.path.join(out_dir, f"{base_name}_overlay_water.png")
    written["overlay_landmask_png"] = os.path.join(out_dir, f"{base_name}_overlay_landmask.png")
    written["overlay_png"]       = os.path.join(out_dir, f"{base_name}_overlay.png")
    written["overlay_tif"]       = os.path.join(out_dir, f"{base_name}_overlay.tif")
    written["overlay_color_tif"] = os.path.join(out_dir, f"{base_name}_overlay_color.tif")

    save_geotiff(class_map.astype(np.uint8), profile_out, written["class_tif"])
    save_geotiff(flood_mask.astype(np.uint8), profile_out, written["flood_tif"])
    save_geotiff(perm_mask.astype(np.uint8),  profile_out, written["perm_tif"])

    # Visualization outputs:
    #  - report PNG (white/red/blue, for embedding in reports)
    #  - water-only RGBA PNG (constant opacity — slider must NOT affect this)
    #  - landmask RGBA PNG (slider DOES affect this)
    #  - combined overlay PNG (legacy, kept for fallback)
    #  - overlay TIF (S2 base + red/blue burned in, self-contained QGIS layer)
    #  - overlay color TIF (RGBA, transparent non-water — drop on top of any basemap)
    visualize_flood_map(flood_mask, perm_mask, written["png"],
                        title=f"Flood Map - {base_name}")
    make_overlay_water_png(flood_mask, perm_mask, written["overlay_water_png"])
    make_overlay_landmask_png(flood_mask, perm_mask, written["overlay_landmask_png"])
    make_overlay_png(flood_mask, perm_mask, written["overlay_png"])
    make_overlay_tif(class_map, profile, written["overlay_tif"],
                      sentinel2_path=sentinel2_path)
    make_overlay_color_tif(class_map, profile, written["overlay_color_tif"])

    # Vector exports (zipped .shp bundles)
    shp_files = class_map_to_shapefiles(
        class_map, profile["transform"], profile["crs"], out_dir, base_name
    )
    written["flood_shp_zip"] = shp_files.get("flood")
    written["perm_shp_zip"]  = shp_files.get("permanent")

    # Area statistics
    flood_km2 = compute_area_km2(flood_mask, profile["transform"], profile["crs"])
    perm_km2  = compute_area_km2(perm_mask,  profile["transform"], profile["crs"])
    total_km2 = compute_total_extent_km2(class_map, profile["transform"], profile["crs"])
    stats = {
        "flood_km2":     round(flood_km2, 4),
        "permanent_km2": round(perm_km2, 4),
        "total_km2":     round(total_km2, 4),
        "flood_pct":     round(100 * flood_km2 / max(total_km2, 1e-9), 3),
        "permanent_pct": round(100 * perm_km2  / max(total_km2, 1e-9), 3),
    }
    return stats, written


# ---------------------------------------------------------------------------
# Job runners
# ---------------------------------------------------------------------------

def run_coordinates_job(job: JobState,
                         predictor: FloodPredictor,
                         lon_min: float, lat_min: float,
                         lon_max: float, lat_max: float,
                         flood_date: str,
                         out_dir: str):
    """Run a bbox+date prediction end-to-end and update `job` as it progresses."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        base_name = f"job_{job.id[:8]}"

        _update(job, status="running", progress=0.05, message="Fetching imagery from Earth Engine...")
        gee_tif = os.path.join(out_dir, f"{base_name}_input.tif")
        predictor.fetcher.fetch(lon_min, lat_min, lon_max, lat_max,
                                flood_date, gee_tif)

        _update(job, progress=0.40, message="Running flood-segmentation model...")
        # Read first 6 bands (geedim sometimes appends a fill mask band)
        with rasterio.open(gee_tif) as src:
            profile = src.profile.copy()
            raw = src.read().astype(np.float32)
        if raw.shape[0] < 6:
            raise RuntimeError(
                f"GEE returned only {raw.shape[0]} bands; expected 6 "
                f"(VV, VH, DEM, Slope, JRC, HAND). Most likely cause: "
                f"no Sentinel-1 acquisition in the bbox+date window. "
                f"Try a different date or a slightly larger bbox."
            )
        from dataset import normalize_band
        BAND_NAMES = ["s1_vv", "s1_vh", "dem", "slope", "jrc", "hand"]
        normalized = np.stack(
            [normalize_band(raw[i], BAND_NAMES[i]) for i in range(6)], axis=-1
        )
        prob_map = sliding_window_predict(
            predictor.model, normalized, predictor.device,
            window=predictor.window_size, stride=predictor.stride,
        )

        _update(job, progress=0.75, message="Postprocessing predictions...")
        class_map, flood_mask, perm_mask = postprocess(prob_map, raw[3], raw[4])

        # Fetch Sentinel-2 RGB for the overlay visualization (best-effort).
        # If S2 has no clear image in ±30 days, overlay TIF falls back to flat gray.
        _update(job, progress=0.80, message="Fetching Sentinel-2 RGB for overlay...")
        s2_tif = os.path.join(out_dir, f"{base_name}_s2_rgb.tif")
        s2_path = predictor.fetcher.fetch_sentinel2_rgb(
            lon_min, lat_min, lon_max, lat_max, flood_date, s2_tif
        )

        _update(job, progress=0.90, message="Computing area + writing outputs...")
        stats, written = _build_outputs_for_class_map(
            class_map, flood_mask, perm_mask, profile, out_dir, base_name,
            sentinel2_path=s2_path,
        )
        stats["bbox"] = [lon_min, lat_min, lon_max, lat_max]
        stats["date"] = flood_date
        stats["has_satellite_overlay"] = bool(s2_path)

        job.stats = stats
        job.files = _public_files(job.id, out_dir, written)
        _update(job, status="done", progress=1.0,
                message=f"Done. Flood area: {stats['flood_km2']:.2f} km^2.")

    except Exception as e:
        job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _update(job, status="error", message=f"Failed: {e}")


def run_shapefile_job(job: JobState,
                       predictor: FloodPredictor,
                       shp_zip_path: str,
                       flood_date: str,
                       out_dir: str):
    """
    Read uploaded zipped shapefile → run prediction over its bbox →
    clip rasters to the polygon → produce stats/exports.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        base_name = f"job_{job.id[:8]}"

        _update(job, status="running", progress=0.05,
                message="Reading shapefile...")
        polygon, bbox, src_crs = read_shapefile_zip(
            shp_zip_path, work_dir=os.path.join(out_dir, "_shp_extract")
        )
        lon_min, lat_min, lon_max, lat_max = bbox
        width_km, height_km = bbox_size_km(bbox)
        _update(job, progress=0.10,
                message=f"Bbox {width_km:.1f}km × {height_km:.1f}km. Fetching imagery...")

        gee_tif = os.path.join(out_dir, f"{base_name}_input.tif")
        predictor.fetcher.fetch(lon_min, lat_min, lon_max, lat_max,
                                flood_date, gee_tif)

        _update(job, progress=0.45, message="Running flood-segmentation model...")
        with rasterio.open(gee_tif) as src:
            profile = src.profile.copy()
            raw = src.read().astype(np.float32)
        from dataset import normalize_band
        BAND_NAMES = ["s1_vv", "s1_vh", "dem", "slope", "jrc", "hand"]
        normalized = np.stack(
            [normalize_band(raw[i], BAND_NAMES[i]) for i in range(6)], axis=-1
        )
        prob_map = sliding_window_predict(
            predictor.model, normalized, predictor.device,
            window=predictor.window_size, stride=predictor.stride,
        )

        _update(job, progress=0.78, message="Postprocessing + clipping to polygon...")
        class_map, flood_mask, perm_mask = postprocess(prob_map, raw[3], raw[4])

        # Persist a temporary unclipped class_tif so we can clip it via rasterio.mask
        tmp_class_tif = os.path.join(out_dir, f"{base_name}_class_unclipped.tif")
        profile_uint8 = profile.copy()
        profile_uint8.update({"driver": "GTiff", "count": 1, "dtype": "uint8"})
        profile_uint8.pop("nodata", None)
        save_geotiff(class_map.astype(np.uint8), profile_uint8, tmp_class_tif)

        clipped_class_tif = os.path.join(out_dir, f"{base_name}_class_clipped.tif")
        clip_raster_to_polygon(tmp_class_tif, polygon, "EPSG:4326",
                                clipped_class_tif, nodata_value=0)
        os.remove(tmp_class_tif)

        # Re-derive masks from the clipped class map (so all outputs agree)
        with rasterio.open(clipped_class_tif) as src:
            class_map = src.read(1)
            profile_clipped = src.profile.copy()
        flood_mask = (class_map == 1).astype(np.uint8)
        perm_mask  = (class_map == 2).astype(np.uint8)

        # Fetch Sentinel-2 RGB for overlay (best-effort)
        _update(job, progress=0.84, message="Fetching Sentinel-2 RGB for overlay...")
        s2_tif = os.path.join(out_dir, f"{base_name}_s2_rgb.tif")
        s2_path = predictor.fetcher.fetch_sentinel2_rgb(
            lon_min, lat_min, lon_max, lat_max, flood_date, s2_tif
        )

        _update(job, progress=0.92, message="Computing area + writing outputs...")
        stats, written = _build_outputs_for_class_map(
            class_map, flood_mask, perm_mask, profile_clipped, out_dir, base_name,
            sentinel2_path=s2_path,
        )
        # Replace the (already-saved) clipped class_tif with the canonical name
        os.replace(clipped_class_tif, written["class_tif"])

        stats["bbox"] = [lon_min, lat_min, lon_max, lat_max]
        stats["date"] = flood_date
        stats["bbox_size_km"] = [round(width_km, 2), round(height_km, 2)]
        stats["source_crs"] = str(src_crs) if src_crs else "EPSG:4326 (assumed)"
        stats["has_satellite_overlay"] = bool(s2_path)

        job.stats = stats
        job.files = _public_files(job.id, out_dir, written)
        _update(job, status="done", progress=1.0,
                message=f"Done. Flood area: {stats['flood_km2']:.2f} km^2.")

    except Exception as e:
        job.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        _update(job, status="error", message=f"Failed: {e}")


# ---------------------------------------------------------------------------
# Quick smoke test (no model, just registry)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    job = JOBS.create()
    print(f"Created job: {job.id}")
    _update(job, status="running", progress=0.5, message="Half done")
    print(JOBS.get(job.id).to_dict())
