"""
Inference pipeline for the trained ResNet50 + UNet++ flood model.

Implements (from DeepSARFlood paper 2025 — extended to 3-class):
  1. GEE data fetching — Sentinel-1 VV/VH, DEM, Slope, JRC, HAND
  2. Preprocessing & band normalization
  3. Sliding window inference (512×512, stride=400) — full softmax averaged
  4. Post-processing: argmax over {non-water, flood, permanent}; slope > 5% → non-water
  5. Output: per-class GeoTIFFs + PNG visualization (red=flood, blue=permanent water)

Usage:
    from inference import FloodPredictor
    predictor = FloodPredictor("checkpoints/model_soup.pth")
    predictor.predict_from_geotiff("input.tif", "output_dir/")
    # or
    predictor.predict_from_gee(lon_min, lat_min, lon_max, lat_max,
                                flood_date="2022-08-06", output_dir="output_dir/")
"""

import os
import sys
import numpy as np
import torch
import rasterio
from rasterio.transform import from_bounds
# Visualization uses PIL (Pillow) — no matplotlib dependency. PIL works
# with uint8 RGB directly, uses ~14× less memory than matplotlib's float64
# RGBA pipeline, and avoids matplotlib's Tk backend (which can't run from
# FastAPI BackgroundTask worker threads).

sys.path.insert(0, os.path.dirname(__file__))

from model   import build_model
from dataset import normalize_band, BAND_STATS


# ---------------------------------------------------------------------------
# GEE Data Fetcher
# ---------------------------------------------------------------------------

class GEEDataFetcher:
    """
    Fetches all required bands from Google Earth Engine for a given
    bounding box and flood date.
    """

    def __init__(self):
        self._ee_initialized = False

    def _init_ee(self):
        if not self._ee_initialized:
            import ee
            try:
                ee.Initialize()
            except Exception:
                ee.Authenticate()
                ee.Initialize()
            self._ee_initialized = True
            print("[GEE] Earth Engine initialized.")

    @staticmethod
    def _run_in_fresh_thread(fn, *args, **kwargs):
        """
        Run `fn` in a brand-new thread and wait for it. Used to give geedim's
        internal asyncio.Runner a clean asyncio context for every download —
        FastAPI's threadpool workers are reused, and geedim leaves the loop
        in a "running" state which breaks the *next* call from the same thread.
        A fresh OS thread has no asyncio state at all, so each download starts
        clean.
        """
        import threading
        result_box: dict = {}
        def target():
            try:
                result_box["value"] = fn(*args, **kwargs)
            except BaseException as e:
                result_box["error"] = e
        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join()
        if "error" in result_box:
            raise result_box["error"]
        return result_box.get("value")

    def fetch(self,
              lon_min: float, lat_min: float,
              lon_max: float, lat_max: float,
              flood_date: str,
              output_path: str,
              scale: int = 10) -> str:
        """
        Downloads a 6-band GeoTIFF for the given bbox and date.

        Args:
            lon_min, lat_min, lon_max, lat_max : bounding box (degrees)
            flood_date  : date string "YYYY-MM-DD"
            output_path : where to save the downloaded GeoTIFF
            scale       : spatial resolution in meters (default 10m)
        Returns:
            output_path
        """
        self._init_ee()
        import ee
        import geedim

        region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        # --- Sentinel-1 ---
        # S1 revisit is 6-12 days; widen the search window so arbitrary
        # bbox+date combinations don't return an empty collection.
        from datetime import datetime, timedelta
        d = datetime.strptime(flood_date, "%Y-%m-%d")
        date_start = (d - timedelta(days=6)).strftime("%Y-%m-%d")
        date_end   = (d + timedelta(days=6)).strftime("%Y-%m-%d")
        s1_coll = (ee.ImageCollection("COPERNICUS/S1_GRD")
                   .filterBounds(region)
                   .filterDate(date_start, date_end)
                   .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                   .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
                   .filter(ee.Filter.eq("instrumentMode", "IW")))

        n_s1 = s1_coll.size().getInfo()
        if n_s1 == 0:
            raise RuntimeError(
                f"No Sentinel-1 images found for bbox "
                f"({lon_min}, {lat_min}, {lon_max}, {lat_max}) within "
                f"±6 days of {flood_date}. Try a different date — "
                f"S1 revisit is 6-12 days."
            )
        print(f"[GEE] Found {n_s1} S1 acquisition(s) in {date_start}..{date_end}")

        s1 = s1_coll.select(["VV", "VH"]).mean().clip(region)

        # --- NASA DEM ---
        dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation").clip(region)
        slope = ee.Terrain.slope(dem).clip(region)

        # --- JRC Permanent Water (seasonality) ---
        jrc = (ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
               .select("seasonality")
               .unmask(0)
               .clip(region))

        # --- HAND ---
        hand = (ee.ImageCollection("users/gena/global-hand/hand-100")
                .mosaic()
                .unmask(0)
                .clip(region))

        # Stack all bands: VV, VH, DEM, Slope, JRC, HAND
        stacked = (s1.rename(["VV", "VH"])
                   .addBands(dem.rename("DEM"))
                   .addBands(slope.rename("Slope"))
                   .addBands(jrc.rename("JRC"))
                   .addBands(hand.float().rename("HAND")))

        # Download using geedim. Wrap in a fresh OS thread so geedim's
        # asyncio.Runner gets a clean context (the FastAPI threadpool's
        # worker threads are reused, which leaks asyncio state).
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        img = geedim.MaskedImage(stacked)
        region_info = region.getInfo()  # do this on calling thread (no asyncio)

        def _do_download():
            img.download(output_path, region=region_info, scale=scale,
                         crs="EPSG:4326", overwrite=True)
        self._run_in_fresh_thread(_do_download)

        print(f"[GEE] Downloaded 6-band image -> {output_path}")
        return output_path

    def fetch_sentinel2_rgb(self,
                             lon_min: float, lat_min: float,
                             lon_max: float, lat_max: float,
                             flood_date: str,
                             output_path: str,
                             scale: int = 10,
                             window_days: int = 30) -> str | None:
        """
        Fetch a cloud-free Sentinel-2 RGB composite for the bbox + date range.

        Used purely for visualization (overlay base image). Median composite
        over [date - window_days, date + window_days], filtered for low-cloud
        scenes. Returns the output path on success, or None if no usable
        Sentinel-2 imagery exists for the request (caller falls back to flat
        gray base in the overlay TIF).
        """
        self._init_ee()
        import ee
        import geedim

        from datetime import datetime, timedelta
        d = datetime.strptime(flood_date, "%Y-%m-%d")
        date_start = (d - timedelta(days=window_days)).strftime("%Y-%m-%d")
        date_end   = (d + timedelta(days=window_days)).strftime("%Y-%m-%d")

        region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(region)
              .filterDate(date_start, date_end)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))

        n_s2 = s2.size().getInfo()
        if n_s2 == 0:
            print(f"[GEE] No Sentinel-2 images found for "
                  f"{date_start}..{date_end} (cloudy <60%). "
                  f"Overlay TIF will use flat-gray base.")
            return None

        composite = (s2.median()
                     .select(["B4", "B3", "B2"])    # red, green, blue
                     .clip(region))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        img = geedim.MaskedImage(composite)
        region_info = region.getInfo()

        def _do_download():
            img.download(output_path, region=region_info, scale=scale,
                         crs="EPSG:4326", overwrite=True)
        try:
            self._run_in_fresh_thread(_do_download)
        except Exception as e:
            print(f"[GEE] Sentinel-2 fetch failed: {e}. Overlay will use flat-gray base.")
            return None

        print(f"[GEE] Downloaded Sentinel-2 RGB ({n_s2} scenes composited) -> {output_path}")
        return output_path


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def load_and_normalize(tif_path: str) -> tuple[np.ndarray, dict]:
    """
    Load a 6-band GeoTIFF (VV, VH, DEM, Slope, JRC, HAND) and normalize
    each band to [0, 1].

    Returns:
        image      : (H, W, 6) float32 numpy array
        profile    : rasterio profile dict (for writing output GeoTIFF)
    """
    band_names = ["s1_vv", "s1_vh", "dem", "slope", "jrc", "hand"]
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        bands = src.read().astype(np.float32)   # (C, H, W)

    # GEE's `geedim` downloads sometimes append an extra fill-mask band, so
    # the file may have 7 bands instead of 6. Only the first 6 are the data
    # bands we stacked in the fetcher: [VV, VH, DEM, Slope, JRC, HAND].
    assert bands.shape[0] >= 6, \
        f"Expected >=6 bands (VV,VH,DEM,Slope,JRC,HAND), got {bands.shape[0]}"

    normalized = np.stack(
        [normalize_band(bands[i], band_names[i]) for i in range(6)],
        axis=-1
    )   # (H, W, 6)
    return normalized, profile


# ---------------------------------------------------------------------------
# Sliding window inference
# ---------------------------------------------------------------------------

def sliding_window_predict(model: torch.nn.Module,
                            image: np.ndarray,
                            device: torch.device,
                            window: int = 512,
                            stride: int = 400,
                            batch_size: int = 8,
                            num_classes: int = 3) -> np.ndarray:
    """
    Run model inference using overlapping sliding windows.

    Args:
        model       : trained FloodSegmentationModel
        image       : (H, W, 6) normalized float32 array
        device      : torch device
        window      : patch size (512 as per paper)
        stride      : step size (400 as per paper — overlap ensures edge coverage)
        batch_size  : how many patches to process at once
        num_classes : number of output classes the model produces (default 3)
    Returns:
        prob_map  : (num_classes, H, W) float32 — per-class softmax probabilities
                    averaged over overlapping windows.
    """
    H, W, C = image.shape
    model.eval()

    # Pad image so every pixel is covered by at least one full window
    pad_h = max(0, window - H % stride) if H % stride != 0 else 0
    pad_w = max(0, window - W % stride) if W % stride != 0 else 0
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    pH, pW, _ = image.shape
    prob_sum  = np.zeros((num_classes, pH, pW), dtype=np.float32)
    count_map = np.zeros((pH, pW), dtype=np.float32)

    # Collect all patch coordinates
    coords = []
    for y in range(0, pH - window + 1, stride):
        for x in range(0, pW - window + 1, stride):
            coords.append((y, x))

    # Process in batches
    patches = []
    patch_coords = []

    def _run_batch(patches, patch_coords):
        batch = torch.from_numpy(
            np.stack(patches, axis=0).transpose(0, 3, 1, 2)  # (B, C, H, W)
        ).to(device)
        with torch.no_grad():
            logits, _ = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B, K, H, W)
        for prob, (y, x) in zip(probs, patch_coords):
            prob_sum[:, y:y+window, x:x+window] += prob
            count_map[y:y+window, x:x+window]   += 1.0

    for y, x in coords:
        patch = image[y:y+window, x:x+window]
        patches.append(patch)
        patch_coords.append((y, x))

        if len(patches) == batch_size:
            _run_batch(patches, patch_coords)
            patches, patch_coords = [], []

    if patches:
        _run_batch(patches, patch_coords)

    # Average overlapping predictions per class
    count_map = np.maximum(count_map, 1.0)
    prob_map = prob_sum / count_map[None, ...]

    # Crop back to original size
    return prob_map[:, :H, :W]


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def postprocess(prob_map: np.ndarray,
                slope_band: np.ndarray,
                jrc_band: np.ndarray | None = None,
                max_slope_pct: float = 5.0,
                jrc_seasonality_threshold: float = 5.0,
                jrc_override: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (C, H, W) per-class probabilities into binary flood / permanent masks.

    Args:
        prob_map    : (C, H, W) softmax probabilities. Channels must be
                      [non-water, flood, permanent].
        slope_band  : (H, W) slope in degrees/percent (raw, unnormalized).
                      Pixels with slope > max_slope_pct are forced to non-water.
        jrc_band    : (H, W) JRC seasonality (0–12, unnormalized). Optional.
                      Used as a sanity override when jrc_override=True.
        max_slope_pct             : pixels with slope > this → non-water (paper: 5%)
        jrc_seasonality_threshold : JRC seasonality cutoff for permanent water
        jrc_override              : if True and jrc_band is given, any pixel the
                                    model called "flood" but where JRC indicates
                                    long-term water is reassigned to permanent.
    Returns:
        class_map      : (H, W) uint8 — 0=non-water, 1=flood, 2=permanent
        flood_mask     : (H, W) uint8 — 1 where class_map==1
        permanent_mask : (H, W) uint8 — 1 where class_map==2
    """
    assert prob_map.ndim == 3 and prob_map.shape[0] >= 3, \
        f"Expected (C>=3, H, W) prob_map, got {prob_map.shape}"

    class_map = prob_map.argmax(axis=0).astype(np.uint8)

    # Remove high-slope pixels (water doesn't accumulate on steep terrain)
    high_slope = slope_band > max_slope_pct
    class_map[high_slope] = 0

    # JRC override: if model said "flood" but JRC has long seasonality, that's
    # actually a permanent river the model missed reassigning.
    if jrc_override and jrc_band is not None:
        permanent_from_jrc = jrc_band > jrc_seasonality_threshold
        misclassified = (class_map == 1) & permanent_from_jrc
        class_map[misclassified] = 2

    flood_mask     = (class_map == 1).astype(np.uint8)
    permanent_mask = (class_map == 2).astype(np.uint8)
    return class_map, flood_mask, permanent_mask


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

FLOOD_RGB     = (0.722, 0.110, 0.110)   # #B91C1C — Tailwind red-700, dark red
PERMANENT_RGB = (0.114, 0.306, 0.847)   # #1D4ED8 — Tailwind blue-700, dark blue
BACKGROUND_RGB = (0.92,  0.92,  0.92)   # light gray (non-water mask)


_FLOOD_U8     = np.array([int(c * 255) for c in FLOOD_RGB],     dtype=np.uint8)
_PERMANENT_U8 = np.array([int(c * 255) for c in PERMANENT_RGB], dtype=np.uint8)
_BG_U8        = np.array([int(c * 255) for c in BACKGROUND_RGB], dtype=np.uint8)


def _maybe_downsample(masks: list[np.ndarray], max_dim: int):
    """Stride-downsample a list of HxW masks together if any dim > max_dim."""
    H, W = masks[0].shape
    if max(H, W) <= max_dim:
        return masks
    stride = max(1, max(H, W) // max_dim)
    return [m[::stride, ::stride] for m in masks]


def visualize_flood_map(flood_mask: np.ndarray,
                         permanent_mask: np.ndarray,
                         output_path: str,
                         title: str = "Flood Inundation Map",
                         max_dim: int = 4096):
    """
    "Report style" PNG: solid red flood, blue permanent, light-gray land.
    Used for embedding in reports and slide decks.
    """
    from PIL import Image
    flood_mask, permanent_mask = _maybe_downsample(
        [flood_mask, permanent_mask], max_dim
    )
    H, W = flood_mask.shape

    rgb = np.broadcast_to(_BG_U8, (H, W, 3)).copy()
    rgb[permanent_mask == 1] = _PERMANENT_U8
    rgb[flood_mask     == 1] = _FLOOD_U8

    Image.fromarray(rgb, mode="RGB").save(output_path, "PNG", optimize=True)
    print(f"[Viz] report PNG  -> {output_path}  ({W}x{H})")


def _smooth_edges(mask_u8: np.ndarray, blur_radius: float = 0.0) -> np.ndarray:
    """
    Optional light Gaussian smoothing on a 0/255 alpha channel.

    Disabled by default — for water-dense scenes (coastal regions, large
    floodplains) even a 0.5-pixel blur bleeds the red/blue tint into the
    surrounding land, making the satellite imagery look stained. Keeping
    the masks crisp is the better trade-off; the slight pixelation at
    very-high zoom comes from the model's native 10 m resolution and can't
    be fixed with smoothing.
    """
    if blur_radius <= 0:
        return mask_u8
    from PIL import Image, ImageFilter
    return np.array(
        Image.fromarray(mask_u8, mode="L").filter(
            ImageFilter.GaussianBlur(radius=blur_radius)
        )
    )


def make_overlay_water_png(flood_mask: np.ndarray,
                            permanent_mask: np.ndarray,
                            output_path: str,
                            max_dim: int = 8192):
    """
    Map overlay layer 1 — only flood (red) + permanent (blue), full opacity,
    transparent everywhere else. The user's opacity slider should NEVER
    change this layer's appearance.
    """
    from PIL import Image
    flood_mask, permanent_mask = _maybe_downsample(
        [flood_mask, permanent_mask], max_dim
    )
    H, W = flood_mask.shape

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    fl = flood_mask     == 1
    pm = permanent_mask == 1
    rgba[fl, 0:3] = _FLOOD_U8
    rgba[pm, 0:3] = _PERMANENT_U8

    alpha = np.zeros((H, W), dtype=np.uint8)
    alpha[fl | pm] = 255
    alpha = _smooth_edges(alpha, blur_radius=0.0)
    rgba[..., 3] = alpha

    Image.fromarray(rgba, mode="RGBA").save(output_path, "PNG", optimize=True)
    print(f"[Viz] water-only PNG    -> {output_path}  ({W}x{H}, RGBA, opaque red/blue)")


def make_overlay_landmask_png(flood_mask: np.ndarray,
                               permanent_mask: np.ndarray,
                               output_path: str,
                               max_dim: int = 8192,
                               land_alpha: int = 200):
    """
    Map overlay layer 2 — only the light-gray non-water mask. Transparent
    over flood/permanent water pixels. The user's opacity slider controls
    THIS layer's `raster-opacity` so the satellite basemap fades in/out
    on the dry land while red/blue water stays constant.
    """
    from PIL import Image
    flood_mask, permanent_mask = _maybe_downsample(
        [flood_mask, permanent_mask], max_dim
    )
    H, W = flood_mask.shape

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    non_water = (flood_mask == 0) & (permanent_mask == 0)
    rgba[non_water, 0:3] = _BG_U8

    alpha = np.zeros((H, W), dtype=np.uint8)
    alpha[non_water] = land_alpha
    alpha = _smooth_edges(alpha, blur_radius=0.0)
    rgba[..., 3] = alpha

    Image.fromarray(rgba, mode="RGBA").save(output_path, "PNG", optimize=True)
    print(f"[Viz] landmask PNG      -> {output_path}  ({W}x{H}, RGBA, slider-controlled)")


def make_overlay_png(flood_mask: np.ndarray,
                      permanent_mask: np.ndarray,
                      output_path: str,
                      max_dim: int = 8192,
                      water_alpha: int = 230):
    """
    Combined overlay PNG (kept for backward compatibility): RGBA with
    transparent non-water + opaque red/blue water. Frontend prefers the
    two-layer split (water + landmask) so the slider only affects the gray.
    """
    from PIL import Image
    flood_mask, permanent_mask = _maybe_downsample(
        [flood_mask, permanent_mask], max_dim
    )
    H, W = flood_mask.shape

    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    fl = flood_mask     == 1
    pm = permanent_mask == 1
    rgba[fl, 0:3] = _FLOOD_U8
    rgba[pm, 0:3] = _PERMANENT_U8
    alpha = np.zeros((H, W), dtype=np.uint8)
    alpha[fl | pm] = water_alpha
    alpha = _smooth_edges(alpha, blur_radius=0.0)
    rgba[..., 3] = alpha

    Image.fromarray(rgba, mode="RGBA").save(output_path, "PNG", optimize=True)
    print(f"[Viz] overlay PNG       -> {output_path}  ({W}x{H}, RGBA, combined)")


def make_overlay_tif(class_map: np.ndarray,
                      profile: dict,
                      output_path: str,
                      sentinel2_path: str | None = None,
                      water_alpha: float = 0.85):
    """
    Self-contained colored GeoTIFF: Sentinel-2 RGB underneath + dark red /
    dark blue burned in where flood/permanent. Non-flood pixels keep the
    real satellite imagery — NO gray overlay. Opens directly in QGIS.

    Args:
        class_map        : (H, W) uint8 with 0/1/2
        profile          : rasterio profile of class_map (transform, crs, etc.)
        output_path      : where to write the RGB GeoTIFF
        sentinel2_path   : optional path to a 3-band S2 RGB GeoTIFF.
                           If None, uses a flat dark-gray base.
        water_alpha      : how strongly to tint water pixels (0=none, 1=solid).
                           Default 0.85 — almost-solid red/blue with a hint
                           of the underlying satellite for realism.
    """
    from rasterio.warp import reproject, Resampling

    H, W = class_map.shape

    # 1. Build / load the base RGB layer (uint8, HxWx3)
    if sentinel2_path and os.path.exists(sentinel2_path):
        with rasterio.open(sentinel2_path) as s2_src:
            s2_data = np.zeros((3, H, W), dtype=np.float32)
            for i in range(3):
                reproject(
                    source=rasterio.band(s2_src, i + 1),
                    destination=s2_data[i],
                    src_transform=s2_src.transform, src_crs=s2_src.crs,
                    dst_transform=profile["transform"], dst_crs=profile["crs"],
                    resampling=Resampling.bilinear,
                )
        # Per-band 2-98 percentile stretch — adapts to whatever range the
        # GEE S2 asset returns (water-dominated scenes have very low
        # reflectance, vegetation/urban much higher).
        bands = []
        for i in range(3):
            band = s2_data[i]
            valid = band[band > 0]   # drop nodata
            if valid.size == 0:
                bands.append(np.zeros((H, W), dtype=np.uint8))
                continue
            lo, hi = np.percentile(valid, [2, 98])
            if hi - lo < 1:
                hi = lo + 1   # avoid div-by-zero on flat patches
            stretched = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255)
            bands.append(stretched.astype(np.uint8))
        base_rgb = np.stack(bands, axis=-1)   # HxWx3
    else:
        # Flat dark-gray base when no S2 imagery available
        base_rgb = np.full((H, W, 3), 60, dtype=np.uint8)

    # 2. Blend the prediction colors on top
    out_rgb = base_rgb.astype(np.float32)
    flood_mask     = class_map == 1
    permanent_mask = class_map == 2

    # Per-pixel: (1 - alpha) * base + alpha * tint_color
    if flood_mask.any():
        out_rgb[flood_mask] = (
            (1 - water_alpha) * base_rgb[flood_mask].astype(np.float32)
            + water_alpha * _FLOOD_U8.astype(np.float32)
        )
    if permanent_mask.any():
        out_rgb[permanent_mask] = (
            (1 - water_alpha) * base_rgb[permanent_mask].astype(np.float32)
            + water_alpha * _PERMANENT_U8.astype(np.float32)
        )
    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)

    # 3. Write as 3-band uint8 GeoTIFF
    out_profile = profile.copy()
    out_profile.update({
        "driver":   "GTiff",
        "count":    3,
        "dtype":    "uint8",
        "compress": "deflate",
        "photometric": "RGB",
    })
    out_profile.pop("nodata", None)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        for i in range(3):
            dst.write(out_rgb[..., i], i + 1)
    print(f"[Viz] overlay TIF -> {output_path}  "
          f"(S2 base: {'yes' if sentinel2_path else 'no'})")


def make_overlay_color_tif(class_map: np.ndarray,
                            profile: dict,
                            output_path: str):
    """
    RGBA GeoTIFF with **only** red/blue prediction polygons — non-water is
    fully transparent (alpha=0). Open this on top of ANY satellite basemap
    in QGIS and you'll see the prediction as a transparent overlay, with
    your basemap visible through dry land.

    Use this instead of overlay.tif when you don't want any baked-in
    satellite imagery — e.g. you have your own base layer in QGIS already.
    """
    H, W = class_map.shape
    rgba = np.zeros((4, H, W), dtype=np.uint8)
    fl = class_map == 1
    pm = class_map == 2

    rgba[0][fl] = _FLOOD_U8[0];     rgba[1][fl] = _FLOOD_U8[1];     rgba[2][fl] = _FLOOD_U8[2]
    rgba[0][pm] = _PERMANENT_U8[0]; rgba[1][pm] = _PERMANENT_U8[1]; rgba[2][pm] = _PERMANENT_U8[2]
    rgba[3][fl | pm] = 255   # fully opaque on water
    # non-water alpha stays 0

    out_profile = profile.copy()
    out_profile.update({
        "driver":   "GTiff",
        "count":    4,
        "dtype":    "uint8",
        "compress": "deflate",
        "photometric": "RGB",
        "alpha":    "yes",
    })
    out_profile.pop("nodata", None)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **out_profile) as dst:
        for i in range(4):
            dst.write(rgba[i], i + 1)
        dst.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        )
    print(f"[Viz] color-only TIF -> {output_path}  (RGBA, transparent non-water)")


def save_geotiff(array: np.ndarray, profile: dict, output_path: str):
    """Save a 2D or 3D numpy array as GeoTIFF."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    profile = profile.copy()
    if array.ndim == 2:
        array = array[np.newaxis]   # (1, H, W)
    profile.update({"count": array.shape[0], "dtype": str(array.dtype)})

    # Drop nodata if it doesn't fit the output dtype (e.g. NaN copied from a
    # float input profile but we're writing uint8 class/mask outputs).
    nd = profile.get("nodata")
    if nd is not None:
        try:
            np.array([nd]).astype(array.dtype, casting="safe")
        except (TypeError, ValueError):
            profile.pop("nodata", None)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array)
    print(f"[GeoTIFF] Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class FloodPredictor:
    """
    High-level flood prediction interface.

    Args:
        checkpoint_path : path to .pth checkpoint (model soup or single model)
        device          : "cuda" | "cpu" | "auto" (default: auto)
        window_size     : sliding window size (default 512)
        stride          : sliding window stride (default 400)
    """

    def __init__(self,
                 checkpoint_path: str,
                 device: str = "auto",
                 window_size: int = 512,
                 stride: int = 400):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.window_size = window_size
        self.stride = stride
        self.model = self._load_model(checkpoint_path)
        self.fetcher = GEEDataFetcher()
        print(f"[FloodPredictor] Ready on {self.device}")

    def _load_model(self, path: str) -> torch.nn.Module:
        ckpt = torch.load(path, map_location=self.device)
        model = build_model(in_channels=6, pretrained=False).to(self.device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        model.eval()
        print(f"[FloodPredictor] Model loaded from {path}")
        return model

    def predict_from_geotiff(self,
                              input_tif: str,
                              output_dir: str) -> dict:
        """
        Run 3-class inference on a pre-downloaded 6-band GeoTIFF
        (channel order: VV, VH, DEM, Slope, JRC, HAND).

        Returns dict with paths to output files.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"[FloodPredictor] Loading {input_tif}…")

        image, profile = load_and_normalize(input_tif)

        # Extract raw slope & JRC for post-processing (before normalization)
        with rasterio.open(input_tif) as src:
            raw_slope = src.read(4).astype(np.float32)   # band 4 = Slope
            raw_jrc   = src.read(5).astype(np.float32)   # band 5 = JRC

        print("[FloodPredictor] Running sliding-window inference…")
        prob_map = sliding_window_predict(
            self.model, image, self.device,
            window=self.window_size, stride=self.stride
        )   # (3, H, W)

        class_map, flood_mask, permanent_mask = postprocess(
            prob_map, raw_slope, raw_jrc
        )

        # Save outputs
        base = os.path.splitext(os.path.basename(input_tif))[0]

        flood_prob_path = os.path.join(output_dir, f"{base}_flood_prob.tif")
        perm_prob_path  = os.path.join(output_dir, f"{base}_permanent_prob.tif")
        class_path      = os.path.join(output_dir, f"{base}_class_map.tif")
        flood_path      = os.path.join(output_dir, f"{base}_flood_mask.tif")
        perm_path       = os.path.join(output_dir, f"{base}_permanent_water.tif")
        png_path        = os.path.join(output_dir, f"{base}_flood_map.png")

        profile_out = profile.copy()
        profile_out.update({"driver": "GTiff", "count": 1})

        save_geotiff(prob_map[1].astype(np.float32), profile_out, flood_prob_path)
        save_geotiff(prob_map[2].astype(np.float32), profile_out, perm_prob_path)
        save_geotiff(class_map.astype(np.uint8),      profile_out, class_path)
        save_geotiff(flood_mask.astype(np.uint8),     profile_out, flood_path)
        save_geotiff(permanent_mask.astype(np.uint8), profile_out, perm_path)
        visualize_flood_map(flood_mask, permanent_mask, png_path,
                             title=f"Flood Map — {base}")

        stats = {
            "flood_pixels":     int(flood_mask.sum()),
            "permanent_pixels": int(permanent_mask.sum()),
            "total_pixels":     int(flood_mask.size),
            "flood_pct":        float(flood_mask.mean() * 100),
            "permanent_pct":    float(permanent_mask.mean() * 100),
        }
        print(f"[FloodPredictor] Flood area    : {stats['flood_pct']:.2f}% of scene")
        print(f"[FloodPredictor] Permanent area: {stats['permanent_pct']:.2f}% of scene")

        return {"flood_prob_tif": flood_prob_path,
                "permanent_prob_tif": perm_prob_path,
                "class_tif": class_path,
                "flood_tif": flood_path,
                "perm_tif": perm_path,
                "png": png_path, "stats": stats}

    def predict_from_gee(self,
                          lon_min: float, lat_min: float,
                          lon_max: float, lat_max: float,
                          flood_date: str,
                          output_dir: str) -> dict:
        """
        Fetch data from GEE then run inference.

        Args:
            lon_min, lat_min, lon_max, lat_max : bounding box in degrees
            flood_date  : "YYYY-MM-DD"
            output_dir  : where to save outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        tif_path = os.path.join(output_dir,
                                f"gee_{flood_date}_{lon_min:.2f}_{lat_min:.2f}.tif")

        self.fetcher.fetch(lon_min, lat_min, lon_max, lat_max,
                           flood_date, tif_path)

        return self.predict_from_geotiff(tif_path, output_dir)

    def predict_from_polygon(self,
                              polygon_4326,
                              flood_date: str,
                              output_dir: str,
                              clip_to_polygon: bool = True) -> dict:
        """
        Predict over a shapely polygon (in EPSG:4326): fetches its bounding
        box from GEE, runs inference, then optionally clips the rasters to
        the actual polygon shape (not just the bbox).

        For polygons larger than ~30 km wide, prefer the tiled predictor
        (see tiled_predictor.predict_large_bbox) — this method does a single
        GEE fetch and may hit GEE size limits for large regions.
        """
        bbox = polygon_4326.bounds   # (minx, miny, maxx, maxy)
        results = self.predict_from_gee(bbox[0], bbox[1], bbox[2], bbox[3],
                                         flood_date, output_dir)
        if not clip_to_polygon:
            return results

        # Clip each raster output to the polygon's actual shape
        from shapefile_handler import clip_raster_to_polygon  # local import
        for key in ("class_tif", "flood_tif", "perm_tif",
                    "flood_prob_tif", "permanent_prob_tif"):
            if key in results and os.path.exists(results[key]):
                clipped = results[key].replace(".tif", "_clipped.tif")
                nodata_val = 0 if "tif" in key and "prob" not in key else 0.0
                clip_raster_to_polygon(results[key], polygon_4326,
                                        "EPSG:4326", clipped,
                                        nodata_value=nodata_val)
                os.replace(clipped, results[key])
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flood inference tool")
    sub = parser.add_subparsers(dest="mode")

    # Mode 1: predict from local GeoTIFF
    p_tif = sub.add_parser("tif", help="Predict from local 6-band GeoTIFF")
    p_tif.add_argument("--input",      required=True)
    p_tif.add_argument("--output_dir", default="outputs/")
    p_tif.add_argument("--checkpoint", required=True)

    # Mode 2: fetch from GEE and predict
    p_gee = sub.add_parser("gee", help="Fetch from Google Earth Engine and predict")
    p_gee.add_argument("--lon_min",    type=float, required=True)
    p_gee.add_argument("--lat_min",    type=float, required=True)
    p_gee.add_argument("--lon_max",    type=float, required=True)
    p_gee.add_argument("--lat_max",    type=float, required=True)
    p_gee.add_argument("--date",       required=True, help="YYYY-MM-DD")
    p_gee.add_argument("--output_dir", default="outputs/")
    p_gee.add_argument("--checkpoint", required=True)

    args = parser.parse_args()

    if args.mode == "tif":
        predictor = FloodPredictor(args.checkpoint)
        results = predictor.predict_from_geotiff(args.input, args.output_dir)
    elif args.mode == "gee":
        predictor = FloodPredictor(args.checkpoint)
        results = predictor.predict_from_gee(
            args.lon_min, args.lat_min, args.lon_max, args.lat_max,
            args.date, args.output_dir,
        )
    else:
        parser.print_help()
