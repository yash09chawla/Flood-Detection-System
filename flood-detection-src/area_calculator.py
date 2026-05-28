"""
Compute real-world area (km²) from a binary raster mask + its georeference.

The trick: pixel counts × pixel-size-in-degrees doesn't give area in km², because
1 degree of longitude varies with latitude. To get a correct area we either
  (a) project the mask to an equal-area CRS and sum its pixel area there, or
  (b) integrate the per-pixel area in geographic coords using a geodetic ellipsoid.

We use approach (a) with EPSG:6933 (NSIDC EASE-Grid 2.0 Global), which is a true
equal-area projection covering the whole globe. Reprojecting a binary mask is
fast even for 10000×10000 inputs.
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


# Equal-area CRS used for area summation. EPSG:6933 = NSIDC EASE-Grid 2.0 Global.
EQUAL_AREA_CRS = "EPSG:6933"


def compute_area_km2(mask: np.ndarray,
                     transform,
                     src_crs,
                     positive_value: int = 1) -> float:
    """
    Sum the area (in km²) of all pixels in `mask` that equal `positive_value`.

    Args:
        mask           : 2D uint8/int array (e.g. flood_mask, permanent_mask)
        transform      : rasterio Affine transform of the mask
        src_crs        : the mask's CRS (rasterio CRS or pyproj CRS string)
        positive_value : the value in mask that counts as "true" (default 1)
    Returns:
        Area in km² (float).
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    # Extract just the binary positive layer
    binary = (mask == positive_value).astype(np.uint8)
    if binary.sum() == 0:
        return 0.0

    src_height, src_width = binary.shape

    # Build the equal-area destination transform that covers the same footprint
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, EQUAL_AREA_CRS,
        src_width, src_height,
        *rasterio.transform.array_bounds(src_height, src_width, transform),
    )

    reprojected = np.zeros((dst_height, dst_width), dtype=np.uint8)
    reproject(
        source=binary,
        destination=reprojected,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=EQUAL_AREA_CRS,
        resampling=Resampling.nearest,
    )

    # In an equal-area CRS, every pixel has the same area = |a*e - b*d|
    # where (a, b, _, d, e, _) are the affine coefficients.
    pixel_area_m2 = abs(dst_transform.a * dst_transform.e
                        - dst_transform.b * dst_transform.d)
    n_pixels = int(reprojected.sum())
    area_m2 = n_pixels * pixel_area_m2
    return area_m2 / 1e6   # m² → km²


def compute_total_extent_km2(mask: np.ndarray, transform, src_crs) -> float:
    """Total area covered by the raster (regardless of mask values), in km²."""
    return compute_area_km2(np.ones_like(mask, dtype=np.uint8), transform, src_crs)


# ---------------------------------------------------------------------------
# Quick sanity test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a 100x100 dummy mask covering ~1 km × 1 km near the equator
    # (1 degree of latitude ≈ 111 km, so 0.009° ≈ 1 km)
    H, W = 100, 100
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[25:75, 25:75] = 1   # central 50% = 0.25 km² of the 1 km² scene

    transform = rasterio.transform.from_bounds(
        west=0.0, south=0.0, east=0.009, north=0.009,
        width=W, height=H,
    )

    area = compute_area_km2(mask, transform, "EPSG:4326")
    total = compute_total_extent_km2(mask, transform, "EPSG:4326")
    print(f"Mask area : {area:.4f} km²")
    print(f"Total area: {total:.4f} km²")
    print(f"Fraction  : {area/total:.4f}  (expected ~0.25)")
