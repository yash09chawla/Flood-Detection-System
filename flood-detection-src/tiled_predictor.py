"""
Predict over a bbox larger than GEE's single-image download limit (~30 km wide
at 10 m resolution, ~50 MB cap) by tiling the request, predicting each tile,
and mosaicking the resulting class maps into one GeoTIFF.

The model itself doesn't need chunking — `sliding_window_predict` already
handles arbitrary-size single tiles internally. This module exists purely to
work around GEE's per-request size cap.
"""

import os
import sys
import math

import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge

sys.path.insert(0, os.path.dirname(__file__))


def _tile_bbox(lon_min: float, lat_min: float,
               lon_max: float, lat_max: float,
               max_tile_deg: float = 0.3) -> list[tuple]:
    """
    Split a bbox into a grid of tiles each at most `max_tile_deg` wide.

    Returns a list of (lon_min, lat_min, lon_max, lat_max) sub-bboxes covering
    the full original bbox without overlap.
    """
    n_lon = max(1, math.ceil((lon_max - lon_min) / max_tile_deg))
    n_lat = max(1, math.ceil((lat_max - lat_min) / max_tile_deg))
    tile_w = (lon_max - lon_min) / n_lon
    tile_h = (lat_max - lat_min) / n_lat

    tiles = []
    for i in range(n_lon):
        for j in range(n_lat):
            tiles.append((
                lon_min + i * tile_w,
                lat_min + j * tile_h,
                lon_min + (i + 1) * tile_w,
                lat_min + (j + 1) * tile_h,
            ))
    return tiles


def predict_large_bbox(predictor,
                        lon_min: float, lat_min: float,
                        lon_max: float, lat_max: float,
                        flood_date: str,
                        output_dir: str,
                        max_tile_deg: float = 0.3,
                        progress_callback=None) -> dict:
    """
    Predict over a large bbox by tiling, then mosaic the resulting class maps.

    Args:
        predictor       : a loaded inference.FloodPredictor
        lon_min, ...    : full bounding box (degrees, EPSG:4326)
        flood_date      : "YYYY-MM-DD"
        output_dir      : where to write the mosaicked outputs
        max_tile_deg    : max width of any sub-tile (default 0.3° ≈ 33 km)
        progress_callback : optional fn(progress: float, message: str) for status updates
    Returns:
        Dict matching predict_from_geotiff's output: paths to class_tif,
        flood_tif, perm_tif, png, etc.
    """
    from inference         import (sliding_window_predict, postprocess,
                                    visualize_flood_map, save_geotiff)
    from raster_to_vector  import class_map_to_shapefiles
    from area_calculator   import compute_area_km2

    os.makedirs(output_dir, exist_ok=True)
    tiles = _tile_bbox(lon_min, lat_min, lon_max, lat_max, max_tile_deg)
    n_tiles = len(tiles)
    if progress_callback:
        progress_callback(0.0, f"Splitting into {n_tiles} tiles...")

    if n_tiles == 1:
        # Just delegate to the simple single-fetch path
        if progress_callback:
            progress_callback(0.1, "Single-tile fast path...")
        return predictor.predict_from_gee(
            lon_min, lat_min, lon_max, lat_max, flood_date, output_dir
        )

    tile_dir = os.path.join(output_dir, "_tiles")
    os.makedirs(tile_dir, exist_ok=True)

    # Predict each tile and stash its class_map TIF for later mosaicking
    tile_class_paths = []
    for idx, (l_min, b_min, l_max, b_max) in enumerate(tiles, 1):
        if progress_callback:
            progress_callback(idx / (n_tiles + 1),
                              f"Tile {idx}/{n_tiles}: fetch + predict...")

        tile_subdir = os.path.join(tile_dir, f"tile_{idx:03d}")
        os.makedirs(tile_subdir, exist_ok=True)
        tile_results = predictor.predict_from_gee(
            l_min, b_min, l_max, b_max, flood_date, tile_subdir,
        )
        tile_class_paths.append(tile_results["class_tif"])

    # Mosaic all the tile class maps into a single TIF
    if progress_callback:
        progress_callback(0.92, "Mosaicking tiles...")

    open_files = [rasterio.open(p) for p in tile_class_paths]
    mosaic, mosaic_transform = rio_merge(open_files, method="first")
    mosaic = mosaic[0].astype(np.uint8)   # (1, H, W) → (H, W)
    mosaic_meta = open_files[0].meta.copy()
    for f in open_files:
        f.close()

    mosaic_meta.update({
        "height":    mosaic.shape[0],
        "width":     mosaic.shape[1],
        "transform": mosaic_transform,
        "count":     1,
        "dtype":     "uint8",
        "driver":    "GTiff",
    })
    mosaic_meta.pop("nodata", None)

    # Write standard output set from the mosaic
    base = "mosaic"
    written = {
        "class_tif": os.path.join(output_dir, f"{base}_class_map.tif"),
        "flood_tif": os.path.join(output_dir, f"{base}_flood_mask.tif"),
        "perm_tif":  os.path.join(output_dir, f"{base}_permanent_water.tif"),
        "png":       os.path.join(output_dir, f"{base}_flood_map.png"),
    }

    flood_mask = (mosaic == 1).astype(np.uint8)
    perm_mask  = (mosaic == 2).astype(np.uint8)

    save_geotiff(mosaic,                                mosaic_meta, written["class_tif"])
    save_geotiff(flood_mask,                            mosaic_meta, written["flood_tif"])
    save_geotiff(perm_mask,                             mosaic_meta, written["perm_tif"])
    visualize_flood_map(flood_mask, perm_mask, written["png"],
                        title=f"Flood Map — {n_tiles}-tile mosaic")

    # Vector exports
    shp_files = class_map_to_shapefiles(
        mosaic, mosaic_meta["transform"], mosaic_meta["crs"], output_dir, base,
    )
    written["flood_shp_zip"] = shp_files.get("flood")
    written["perm_shp_zip"]  = shp_files.get("permanent")

    # Stats
    flood_km2 = compute_area_km2(flood_mask, mosaic_meta["transform"], mosaic_meta["crs"])
    perm_km2  = compute_area_km2(perm_mask,  mosaic_meta["transform"], mosaic_meta["crs"])
    total_km2 = compute_area_km2(np.ones_like(mosaic, dtype=np.uint8),
                                  mosaic_meta["transform"], mosaic_meta["crs"])
    stats = {
        "flood_km2":     round(flood_km2, 4),
        "permanent_km2": round(perm_km2, 4),
        "total_km2":     round(total_km2, 4),
        "flood_pct":     round(100 * flood_km2 / max(total_km2, 1e-9), 3),
        "permanent_pct": round(100 * perm_km2  / max(total_km2, 1e-9), 3),
        "n_tiles":       n_tiles,
    }
    written["stats"] = stats

    if progress_callback:
        progress_callback(1.0, f"Done. {n_tiles} tiles mosaicked.")
    return written
