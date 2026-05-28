"""
FastAPI app for the flood-prediction web service.

Run locally:
    uvicorn api:app --host 0.0.0.0 --port 8000

Then in a second terminal:
    ngrok http 8000

Copy the https://...ngrok-free.app URL into the frontend's NEXT_PUBLIC_API_BASE.

Environment variables (optional):
    FLOOD_CHECKPOINT     — path to .pth checkpoint   (default: ./checkpoints-v3/best_dice.pth)
    FLOOD_GEE_KEY        — path to GEE service-account JSON
    FLOOD_JOB_DIR        — base dir for job outputs  (default: /tmp/flood_jobs)
    FLOOD_FRONTEND_ORIGINS — comma-separated CORS origins
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Optional

# Make stdout/stderr UTF-8 on Windows so Unicode glyphs in our log messages
# (→, ✓, ✗, etc.) don't crash predictions inside background tasks.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:
    pass

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))

from job_runner import JOBS, JobState, run_coordinates_job, run_shapefile_job


# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------

DEFAULT_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints-v3", "best_dice.pth"
)
if not os.path.exists(DEFAULT_CKPT):
    _alt_ckpt = os.path.join(
        os.path.dirname(__file__), "..", "checkpoint-v3", "best_dice.pth"
    )
    if os.path.exists(_alt_ckpt):
        DEFAULT_CKPT = _alt_ckpt

CHECKPOINT_PATH = os.environ.get("FLOOD_CHECKPOINT", DEFAULT_CKPT)

DEFAULT_GEE_KEY = os.path.join(os.path.dirname(__file__), "..", "gee-key.json")
GEE_KEY_PATH = os.environ.get("FLOOD_GEE_KEY", DEFAULT_GEE_KEY if os.path.exists(DEFAULT_GEE_KEY) else "")

# Support loading GEE credentials from a raw JSON environment variable string (useful for serverless/Docker deploys)
_gee_json = os.environ.get("FLOOD_GEE_KEY_JSON")
if _gee_json:
    try:
        _temp_key = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        _temp_key.write(_gee_json.encode("utf-8"))
        _temp_key.close()
        GEE_KEY_PATH = _temp_key.name
    except Exception as _e:
        print(f"[api] Error writing FLOOD_GEE_KEY_JSON env var to temp file: {_e}")

JOB_BASE_DIR    = os.environ.get("FLOOD_JOB_DIR", os.path.join(tempfile.gettempdir(), "flood_jobs"))

_cors_env = os.environ.get("FLOOD_FRONTEND_ORIGINS",
                           "http://localhost:3000,http://127.0.0.1:3000")
CORS_ORIGINS = [o.strip() for o in _cors_env.split(",") if o.strip()]

# Vercel preview URLs are random per deploy — allow them with a regex
CORS_ORIGIN_REGEX = r"https://.*\.vercel\.app"

os.makedirs(JOB_BASE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# FastAPI app + CORS
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Flood Prediction API",
    description="3-class flood vs permanent water segmentation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Model loading (once, on startup)
# ---------------------------------------------------------------------------

_predictor = None
_init_error: str | None = None


@app.on_event("startup")
def _load_predictor():
    global _predictor, _init_error
    try:
        from inference import FloodPredictor
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}. "
                f"Set FLOOD_CHECKPOINT env var or place best_dice.pth there."
            )
        _predictor = FloodPredictor(CHECKPOINT_PATH)

        # If a GEE key was provided, init Earth Engine with the service account.
        # Otherwise the GEE fetcher will lazy-init via ee.Authenticate() (only
        # works on a machine with a valid auth token).
        if GEE_KEY_PATH and os.path.exists(GEE_KEY_PATH):
            import ee
            credentials = ee.ServiceAccountCredentials(None, GEE_KEY_PATH)
            ee.Initialize(credentials)
            _predictor.fetcher._ee_initialized = True
            print(f"[api] GEE initialized with service account: {GEE_KEY_PATH}")
        else:
            print("[api] No GEE service-account key provided; GEE fetcher will "
                  "attempt lazy init via ee.Authenticate() if needed.")

        print(f"[api] Predictor ready. Checkpoint: {CHECKPOINT_PATH}")
        print(f"[api] Job output dir: {JOB_BASE_DIR}")
    except Exception as e:
        _init_error = f"{type(e).__name__}: {e}"
        print(f"[api] STARTUP ERROR: {_init_error}")


def _require_predictor():
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. {_init_error or ''}",
        )


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class CoordinatesRequest(BaseModel):
    lon_min: float = Field(..., ge=-180, le=180)
    lat_min: float = Field(..., ge=-90,  le=90)
    lon_max: float = Field(..., ge=-180, le=180)
    lat_max: float = Field(..., ge=-90,  le=90)
    date:    str   = Field(..., description="YYYY-MM-DD")


class JobCreateResponse(BaseModel):
    job_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "ok":               _predictor is not None,
        "model_loaded":     _predictor is not None,
        "gee_initialized":  bool(_predictor and _predictor.fetcher._ee_initialized),
        "checkpoint":       os.path.basename(CHECKPOINT_PATH),
        "init_error":       _init_error,
    }


@app.post("/api/predict/coordinates", response_model=JobCreateResponse)
def predict_coordinates(req: CoordinatesRequest, background_tasks: BackgroundTasks):
    _require_predictor()

    if req.lon_min >= req.lon_max or req.lat_min >= req.lat_max:
        raise HTTPException(400, "Invalid bbox: min must be < max for both lon and lat")

    job = JOBS.create()
    out_dir = os.path.join(JOB_BASE_DIR, job.id)

    background_tasks.add_task(
        run_coordinates_job, job, _predictor,
        req.lon_min, req.lat_min, req.lon_max, req.lat_max,
        req.date, out_dir,
    )
    return {"job_id": job.id}


@app.post("/api/predict/shapefile", response_model=JobCreateResponse)
async def predict_shapefile(
    background_tasks: BackgroundTasks,
    shp_zip: UploadFile = File(..., description="Zipped .shp + .shx + .dbf + .prj"),
    date:    str        = Form(..., description="YYYY-MM-DD"),
):
    _require_predictor()

    if not shp_zip.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Upload must be a .zip containing the shapefile components.")

    job = JOBS.create()
    out_dir = os.path.join(JOB_BASE_DIR, job.id)
    os.makedirs(out_dir, exist_ok=True)

    # Stream the upload to disk so the worker can read it
    saved_zip = os.path.join(out_dir, "input_shapefile.zip")
    with open(saved_zip, "wb") as f:
        shutil.copyfileobj(shp_zip.file, f)

    background_tasks.add_task(
        run_shapefile_job, job, _predictor, saved_zip, date, out_dir,
    )
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.get("/api/jobs")
def list_jobs():
    return {"jobs": JOBS.list()}


@app.get("/api/files/{job_id}/{filename}")
def get_file(job_id: str, filename: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != "done":
        raise HTTPException(409, f"Job is {job.status}; files not ready yet.")

    # Whitelist by basename — never let the client traverse paths
    safe_name = os.path.basename(filename)
    if safe_name != filename or ".." in safe_name:
        raise HTTPException(400, "Invalid filename")

    full_path = os.path.join(JOB_BASE_DIR, job_id, safe_name)
    if not os.path.exists(full_path):
        raise HTTPException(404, f"File '{safe_name}' not found for this job")

    media_type = {
        ".tif":  "image/tiff",
        ".tiff": "image/tiff",
        ".png":  "image/png",
        ".zip":  "application/zip",
    }.get(Path(safe_name).suffix.lower(), "application/octet-stream")

    return FileResponse(full_path, media_type=media_type, filename=safe_name)


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    out_dir = os.path.join(JOB_BASE_DIR, job_id)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    # Note: we leave the JobState in the registry for status history;
    # client just won't be able to fetch its files anymore.
    return {"deleted": True, "job_id": job_id}


@app.post("/api/jobs/cleanup")
def cleanup_old_jobs(keep_last: int = 5):
    """
    Delete all but the `keep_last` most recent job folders. Each job's
    outputs can be ~500 MB for a large bbox; this prevents the disk from
    filling up after many demos.
    """
    if not os.path.isdir(JOB_BASE_DIR):
        return {"deleted": 0, "kept": 0}

    entries = []
    for name in os.listdir(JOB_BASE_DIR):
        p = os.path.join(JOB_BASE_DIR, name)
        if os.path.isdir(p):
            entries.append((p, os.path.getmtime(p)))
    entries.sort(key=lambda e: e[1], reverse=True)   # newest first

    to_keep, to_delete = entries[:keep_last], entries[keep_last:]
    bytes_freed = 0
    for path, _ in to_delete:
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    bytes_freed += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        shutil.rmtree(path, ignore_errors=True)

    return {
        "deleted":  len(to_delete),
        "kept":     len(to_keep),
        "freed_mb": round(bytes_freed / 1e6, 1),
    }


# ---------------------------------------------------------------------------
# Local dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
