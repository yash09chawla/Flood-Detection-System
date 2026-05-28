"""
Read user-uploaded shapefiles (zipped) and clip prediction rasters to polygons.

Two responsibilities:
  - Unzip a user upload (which may include junk like __MACOSX/ or nested
    folders), find the .shp inside, dissolve all features into one polygon,
    return the polygon + bbox in EPSG:4326 (so it can drive a GEE fetch).
  - After prediction, clip the resulting raster to the actual polygon shape
    (not just its bounding box) so the user gets predictions inside their
    region of interest only.
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask


def _find_shp_in_dir(directory: str) -> str:
    """Recursively find the first .shp file under `directory`."""
    for root, _dirs, files in os.walk(directory):
        # Skip macOS metadata folder if present
        if "__MACOSX" in root.split(os.sep):
            continue
        for f in files:
            if f.lower().endswith(".shp"):
                return os.path.join(root, f)
    raise ValueError(f"No .shp file found in {directory}")


def read_shapefile_zip(zip_path: str, work_dir: str | None = None):
    """
    Extract a zipped shapefile, dissolve all features into one polygon,
    and return (polygon_4326, bbox_4326, original_crs).

    Args:
        zip_path : path to the .zip the user uploaded
        work_dir : where to extract files (defaults to a sibling of zip_path)
    Returns:
        polygon_4326 : shapely (Multi)Polygon in EPSG:4326
        bbox_4326    : (lon_min, lat_min, lon_max, lat_max) in EPSG:4326
        original_crs : the CRS the shapefile arrived in (for metadata)
    """
    import geopandas as gpd

    zip_path = str(zip_path)
    if work_dir is None:
        work_dir = str(Path(zip_path).with_suffix("")) + "_extracted"
    os.makedirs(work_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work_dir)

    shp_path = _find_shp_in_dir(work_dir)
    gdf = gpd.read_file(shp_path)

    if gdf.empty:
        raise ValueError(f"Shapefile at {shp_path} contains no features.")

    original_crs = gdf.crs
    if gdf.crs is None:
        # Some user shapefiles ship without a .prj — default to WGS84 and warn
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # Dissolve all features into one geometry (handles multi-feature shapefiles)
    polygon_4326 = gdf.unary_union
    bbox_4326 = polygon_4326.bounds   # (minx, miny, maxx, maxy)
    return polygon_4326, bbox_4326, original_crs


def clip_raster_to_polygon(tif_path: str,
                           polygon,
                           polygon_crs,
                           out_path: str,
                           nodata_value=0) -> str:
    """
    Clip a GeoTIFF to a polygon's exact shape (not just bbox).

    Args:
        tif_path     : input raster
        polygon      : shapely (Multi)Polygon
        polygon_crs  : CRS of `polygon` (will be reprojected to raster's CRS)
        out_path     : where to write the clipped raster
        nodata_value : value written outside the polygon (0 = non-water for class maps)
    Returns:
        out_path
    """
    import geopandas as gpd

    with rasterio.open(tif_path) as src:
        # Reproject the polygon into the raster's CRS so mask() works correctly
        gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs=polygon_crs)
        gdf = gdf.to_crs(src.crs)
        shapes = [g.__geo_interface__ for g in gdf.geometry]

        out_image, out_transform = rio_mask(
            src, shapes, crop=False, nodata=nodata_value,
        )
        out_meta = src.meta.copy()

    out_meta.update({
        "height":    out_image.shape[1],
        "width":     out_image.shape[2],
        "transform": out_transform,
        "nodata":    nodata_value if np.issubdtype(out_image.dtype, np.floating) else None,
    })
    if out_meta["nodata"] is None:
        out_meta.pop("nodata", None)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(out_image)
    return out_path


def bbox_size_km(bbox_4326) -> tuple[float, float]:
    """
    Approximate width and height of a bbox in km (rough — ok for routing logic).
    Used to decide whether to use a single GEE fetch or the tiled predictor.
    """
    lon_min, lat_min, lon_max, lat_max = bbox_4326
    mid_lat = (lat_min + lat_max) / 2.0
    # 1 deg latitude ≈ 110.574 km; 1 deg longitude ≈ 111.320 * cos(lat) km
    height_km = (lat_max - lat_min) * 110.574
    width_km  = (lon_max - lon_min) * 111.320 * abs(np.cos(np.radians(mid_lat)))
    return width_km, height_km


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    from shapely.geometry import Polygon

    # Build a tiny shapefile in a temp dir, zip it, then read it back
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not installed; skipping smoke test.")
        raise SystemExit(0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Create a small polygon over Bolivia
        poly = Polygon([(-66.0, -13.7), (-65.6, -13.7),
                        (-65.6, -13.4), (-66.0, -13.4)])
        gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")
        shp = tmp / "test.shp"
        gdf.to_file(shp, driver="ESRI Shapefile")

        # Zip the four sidecars
        zip_path = tmp / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for ext in (".shp", ".shx", ".dbf", ".prj"):
                f = tmp / f"test{ext}"
                if f.exists():
                    zf.write(f, arcname=f.name)

        polygon, bbox, crs = read_shapefile_zip(str(zip_path),
                                                work_dir=str(tmp / "extract"))
        print(f"Read polygon : {polygon.area:.4f} sq deg")
        print(f"Bbox         : {bbox}")
        print(f"Original CRS : {crs}")
        w, h = bbox_size_km(bbox)
        print(f"Bbox size    : {w:.1f} km wide × {h:.1f} km tall")
