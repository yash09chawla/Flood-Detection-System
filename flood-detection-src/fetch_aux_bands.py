"""
One-time downloader for missing auxiliary bands (DEM / Slope / HAND).

For each Sen1Floods11 chip already on disk under
`<data_root>/HandLabeled/S1Hand/<base_id>_S1Hand.tif`, fetch the matching
NASADEM elevation, terrain slope, and Height-Above-Nearest-Drainage from
Google Earth Engine and write them into:

    <data_root>/HandLabeled/DEM/<base_id>_DEM.tif
    <data_root>/HandLabeled/Slope/<base_id>_Slope.tif
    <data_root>/HandLabeled/HAND/<base_id>_HAND.tif

Each output is reprojected to the S1Hand chip's CRS, transform, and shape so
training-time alignment with VV/VH is exact. Re-running skips any chip that
already has all three outputs (idempotent — safe to resume).

Usage:
    python fetch_aux_bands.py --data_root ../BTP-SenData
    python fetch_aux_bands.py --data_root ../BTP-SenData --limit 5    # smoke test
    python fetch_aux_bands.py --data_root ../BTP-SenData --bands DEM Slope
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

sys.path.insert(0, os.path.dirname(__file__))


BAND_SPECS = {
    "DEM":   {"asset": "NASA/NASADEM_HGT/001",          "select": "elevation"},
    "Slope": {"asset": None,                             "select": None},  # derived
    "HAND":  {"asset": "users/gena/global-hand/hand-100","select": None,
              "is_collection": True},
}


def _init_ee():
    import ee

    SERVICE_ACCOUNT_FILE = '/kaggle/input/datasets/yash10chawla/last-try-key/gee-kaggle-project-fbe0b036073d.json'

    credentials = ee.ServiceAccountCredentials(
        None,
        SERVICE_ACCOUNT_FILE
    )

    ee.Initialize(credentials)
    print("[GEE] Earth Engine initialized (service account).")


def _bbox_from_tif(path: str):
    """Return (minx, miny, maxx, maxy) in EPSG:4326 plus the source profile."""
    from rasterio.warp import transform_bounds
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        bounds = src.bounds
        bounds_4326 = transform_bounds(src.crs, "EPSG:4326",
                                       *bounds, densify_pts=21)
    return bounds_4326, profile


def _build_ee_image(band: str, region):
    import ee
    if band == "DEM":
        return ee.Image(BAND_SPECS["DEM"]["asset"]).select(
            BAND_SPECS["DEM"]["select"]
        ).clip(region).rename("DEM")
    if band == "Slope":
        dem = ee.Image(BAND_SPECS["DEM"]["asset"]).select(
            BAND_SPECS["DEM"]["select"]
        )
        return ee.Terrain.slope(dem).clip(region).rename("Slope")
    if band == "HAND":
        return (ee.ImageCollection(BAND_SPECS["HAND"]["asset"])
                .mosaic()
                .unmask(0)
                .float()
                .clip(region)
                .rename("HAND"))
    raise ValueError(f"Unknown band: {band}")


def _download_band(band: str, bounds_4326, ref_profile, out_path: str,
                   scale: int = 10):
    """Download one band as GeoTIFF reprojected to ref_profile's grid."""
    import ee
    import geedim

    region = ee.Geometry.Rectangle([
        bounds_4326[0], bounds_4326[1], bounds_4326[2], bounds_4326[3]
    ])
    img = _build_ee_image(band, region)
    masked = geedim.MaskedImage(img)

    tmp_path = out_path + ".tmp.tif"
    masked.download(tmp_path, region=region.getInfo(), scale=scale,
                    crs="EPSG:4326", overwrite=True)

    # Reproject + resample to the S1Hand grid so pixels align exactly.
    with rasterio.open(tmp_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs

    dst_profile = ref_profile.copy()
    dst_profile.update({"count": 1, "dtype": "float32",
                        "driver": "GTiff", "compress": "deflate"})
    dst = np.zeros((dst_profile["height"], dst_profile["width"]),
                   dtype=np.float32)
    reproject(
        source=src_data, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_profile["transform"], dst_crs=dst_profile["crs"],
        resampling=Resampling.bilinear,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **dst_profile) as out:
        out.write(dst, 1)
    os.remove(tmp_path)


def _list_chips(data_root: str) -> list[str]:
    s1_dir = os.path.join(data_root, "HandLabeled", "S1Hand")
    if not os.path.isdir(s1_dir):
        raise SystemExit(f"S1Hand directory not found: {s1_dir}")
    chips = []
    for f in sorted(os.listdir(s1_dir)):
        if f.endswith("_S1Hand.tif"):
            chips.append(f[: -len("_S1Hand.tif")])
    return chips


def _output_path(data_root: str, base_id: str, band: str) -> str:
    return os.path.join(data_root, "HandLabeled", band,
                        f"{base_id}_{band}.tif")


def _all_outputs_exist(data_root: str, base_id: str, bands: list[str]) -> bool:
    return all(os.path.exists(_output_path(data_root, base_id, b)) for b in bands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Path containing HandLabeled/ subdir (e.g. BTP-SenData)")
    ap.add_argument("--bands", nargs="+", default=["DEM", "Slope", "HAND"],
                    choices=["DEM", "Slope", "HAND"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N chips (for smoke testing)")
    ap.add_argument("--scale", type=int, default=10,
                    help="GEE download scale in meters (default 10)")
    args = ap.parse_args()

    chips = _list_chips(args.data_root)
    if args.limit:
        chips = chips[: args.limit]
    print(f"[fetch_aux] {len(chips)} chips to consider | bands={args.bands}")

    _init_ee()

    n_done, n_skipped, n_failed = 0, 0, 0
    for i, base_id in enumerate(chips, 1):
        if _all_outputs_exist(args.data_root, base_id, args.bands):
            n_skipped += 1
            continue

        s1_path = os.path.join(args.data_root, "HandLabeled", "S1Hand",
                               f"{base_id}_S1Hand.tif")
        bounds_4326, ref_profile = _bbox_from_tif(s1_path)

        for band in args.bands:
            out_path = _output_path(args.data_root, base_id, band)
            if os.path.exists(out_path):
                continue
            try:
                _download_band(band, bounds_4326, ref_profile, out_path,
                               scale=args.scale)
                print(f"  [{i:4d}/{len(chips)}] {base_id} :: {band} ✓")
            except Exception as e:
                n_failed += 1
                print(f"  [{i:4d}/{len(chips)}] {base_id} :: {band} ✗  {e}")
                # Brief backoff so a transient error doesn't snowball.
                time.sleep(2)
        n_done += 1

    print(f"\n[fetch_aux] done. processed={n_done}  "
          f"skipped(already-present)={n_skipped}  failed={n_failed}")


if __name__ == "__main__":
    main()
