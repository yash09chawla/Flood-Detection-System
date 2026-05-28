"""
Convert a class-map raster (uint8 with values 0/1/2) into shapefiles.

We extract polygons per non-zero class using `rasterio.features.shapes()`,
write them via geopandas, and zip the four shapefile components
(.shp + .shx + .dbf + .prj) so the frontend can serve a single download.
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape


CLASS_NAMES = {1: "flood", 2: "permanent"}


def mask_to_geodataframe(mask: np.ndarray,
                         transform,
                         crs,
                         class_value: int):
    """
    Extract polygons for a single class from the mask.

    Returns a GeoDataFrame with columns ['class', 'class_id', 'geometry'].
    Empty if no pixels match.
    """
    import geopandas as gpd  # imported lazily so the module import is cheap

    binary = (mask == class_value).astype(np.uint8)
    if binary.sum() == 0:
        return gpd.GeoDataFrame(
            {"class": [], "class_id": [], "geometry": []}, crs=crs
        )

    polys = []
    for geom, val in features.shapes(binary, mask=binary.astype(bool),
                                      transform=transform):
        if int(val) == 1:
            polys.append(shape(geom))

    if not polys:
        return gpd.GeoDataFrame(
            {"class": [], "class_id": [], "geometry": []}, crs=crs
        )

    name = CLASS_NAMES.get(class_value, f"class_{class_value}")
    return gpd.GeoDataFrame(
        {"class": [name] * len(polys),
         "class_id": [class_value] * len(polys),
         "geometry": polys},
        crs=crs,
    )


def write_shapefile_zip(gdf, out_zip_path: str, layer_name: str = "polygons") -> str:
    """
    Write the GeoDataFrame to a shapefile (4 sidecar files), then zip them
    into a single .zip the frontend can offer as one download.

    Returns the path to the .zip file.
    """
    out_zip_path = str(out_zip_path)
    out_dir = os.path.dirname(out_zip_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    if gdf.empty:
        # Still produce an empty zip so the API contract holds — frontend can
        # show "0 polygons extracted" without error-handling a missing file.
        with zipfile.ZipFile(out_zip_path, "w") as zf:
            zf.writestr(f"{layer_name}_README.txt",
                        "No polygons of this class were detected.\n")
        return out_zip_path

    # geopandas writes into a directory of 4 files; we assemble them in /tmp
    # and then zip just those four (avoid pulling in unrelated files).
    tmp_dir = Path(out_dir) / f".__tmp_shp_{layer_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    shp_path = tmp_dir / f"{layer_name}.shp"
    gdf.to_file(shp_path, driver="ESRI Shapefile")

    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            f = tmp_dir / f"{layer_name}{ext}"
            if f.exists():
                zf.write(f, arcname=f.name)

    # Tidy up the temp dir
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    return out_zip_path


def class_map_to_shapefiles(class_map: np.ndarray,
                             transform,
                             crs,
                             output_dir: str,
                             base_name: str = "result") -> dict:
    """
    Convert a 3-class map into two zipped shapefiles (flood + permanent).

    Returns a dict {class_name: zip_path} for the frontend to expose as
    download buttons.
    """
    os.makedirs(output_dir, exist_ok=True)
    out = {}
    for class_value, class_name in CLASS_NAMES.items():
        gdf = mask_to_geodataframe(class_map, transform, crs, class_value)
        zip_path = os.path.join(output_dir, f"{base_name}_{class_name}_polygons.zip")
        write_shapefile_zip(gdf, zip_path, layer_name=f"{class_name}_polygons")
        out[class_name] = zip_path
    return out


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    # 100×100 fake class map: a 30×30 flood block, a 20×20 permanent block
    cm = np.zeros((100, 100), dtype=np.uint8)
    cm[10:40, 10:40] = 1   # flood
    cm[60:80, 60:80] = 2   # permanent

    transform = rasterio.transform.from_bounds(0.0, 0.0, 1.0, 1.0, 100, 100)

    with tempfile.TemporaryDirectory() as tmp:
        files = class_map_to_shapefiles(cm, transform, "EPSG:4326", tmp, "test")
        for name, path in files.items():
            size = os.path.getsize(path)
            print(f"  {name:10s} → {os.path.basename(path)}  ({size} bytes)")
        # Verify the flood zip contains the expected sidecars
        with zipfile.ZipFile(files["flood"]) as zf:
            print(f"  flood zip contents: {zf.namelist()}")
