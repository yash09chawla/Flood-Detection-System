"""
PyTorch Dataset for the Sen1Floods11 dataset.

Sen1Floods11 structure (v1.1) — actual layout on Google Drive:
  v1.1/
    data/flood_events/
      HandLabeled/
        S1Hand/           — Sentinel-1 GRD chips (VV + VH), 2-channel GeoTIFF
        LabelHand/        — Hand-labeled flood masks, 1-channel GeoTIFF {-1, 0, 1}
        S2Hand/           — Sentinel-2 chips (for MNDWI label generation)
        JRCWaterHand/     — JRC permanent water patches
        S1OtsuLabelHand/  — (not used)
      WeaklyLabeled/
        S1Weak/           — Sentinel-1 GRD chips for weakly-labeled data
        S2Weak/           — Sentinel-2 chips
        S1OtsuLabelWeak/  — (not used)
        S2IndexLabelWeak/ — (not used)
    splits/
      flood_handlabeled/
        flood_train_data.csv
        flood_valid_data.csv
        flood_test_data.csv

  File naming convention:
    S1Hand:       {Region}_{ID}_S1Hand.tif
    LabelHand:    {Region}_{ID}_LabelHand.tif
    JRCWaterHand: {Region}_{ID}_JRCWaterHand.tif
    S2Hand:       {Region}_{ID}_S2Hand.tif

Channels assembled per sample:
  [0] S1-VV    (dB, range approx -25 to 0)
  [1] S1-VH    (dB)
  [2] DEM      (meters)           — optional, fallback to 0.5 if missing
  [3] Slope    (degrees or %)     — optional, fallback to 0.0 if missing
  [4] JRC      (water seasonality, 0-12 months)
  [5] HAND     (meters above nearest drainage) — optional, fallback to 0.5 if missing

All bands normalized to [0, 1] using per-band statistics (min/max).

Labels (3-class, derived at load time from LabelHand + JRCWaterHand):
   2 = permanent water  (LabelHand == 1 AND JRCWaterHand >= JRC_PERMANENT_THRESHOLD)
   1 = flood            (LabelHand == 1 AND JRCWaterHand <  JRC_PERMANENT_THRESHOLD)
   0 = non-water        (LabelHand == 0)
  -1 = masked/no-data   (LabelHand == -1, ignored in loss/metric computation)

Note: Sen1Floods11 ships JRCWaterHand as a binary mask {0, 1}, so the default
threshold (0.5) treats every JRC==1 pixel as permanent. About ~3% of labeled
water pixels are permanent in this dataset.
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

from augmentations import get_train_transforms, get_val_transforms, apply_transforms


# ---------------------------------------------------------------------------
# Per-band normalization statistics (from Sen1Floods11 / paper)
# These are approximate global values; will be refined during training.
# ---------------------------------------------------------------------------

BAND_STATS = {
    "s1_vv":  {"min": -50.0,  "max": 1.0},
    "s1_vh":  {"min": -50.0,  "max": 1.0},
    "dem":    {"min": -100.0, "max": 6000.0},
    "slope":  {"min": 0.0,    "max": 90.0},
    "jrc":    {"min": 0.0,    "max": 100.0},
    "hand":   {"min": 0.0,    "max": 200.0},
}

BAND_ORDER = ["s1_vv", "s1_vh", "dem", "slope", "jrc", "hand"]

# JRC threshold above which a labeled water pixel is treated as permanent
# rather than flood. Sen1Floods11 ships JRCWaterHand as a BINARY mask
# {0 = no permanent water, 1 = permanent water} — confirmed by inspecting the
# tifs (max value across the dataset is 1). So 0.5 cleanly captures the "1"
# bucket while staying robust to dtype rounding.
# (Note: GEE's "seasonality" band at inference time is 0–12 instead — that
#  uses a separate threshold inside inference.postprocess.)
JRC_PERMANENT_THRESHOLD = 0.5


def build_three_class_label(label_hand: np.ndarray,
                             jrc_raw: np.ndarray | None,
                             threshold: float = JRC_PERMANENT_THRESHOLD) -> np.ndarray:
    """
    Combine LabelHand {-1, 0, 1} with raw JRC seasonality (0-12) into a
    3-class label {-1, 0, 1, 2}.

    If jrc_raw is None (missing locally), all label==1 pixels stay as flood (1)
    so the dataloader still works, but the model can't learn the permanent class.
    """
    out = label_hand.astype(np.int64).copy()
    if jrc_raw is None:
        return out
    permanent = (out == 1) & (jrc_raw >= threshold)
    out[permanent] = 2
    return out




def normalize_band(arr: np.ndarray, band_name: str) -> np.ndarray:
    """Normalize a single band to [0, 1] using global statistics."""
    stats = BAND_STATS[band_name]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)  # kill NaNs first
    arr = np.clip(arr, stats["min"], stats["max"])
    return (arr - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)






# ---------------------------------------------------------------------------
# Rasterio helpers
# ---------------------------------------------------------------------------

def read_tif(path: str, band: int = 1) -> np.ndarray:
    """Read one band from a GeoTIFF. Returns float32 (H, W)."""
    with rasterio.open(path) as src:
        data = src.read(band).astype(np.float32)
    return data


def read_tif_all(path: str) -> np.ndarray:
    """Read all bands from a GeoTIFF. Returns float32 (C, H, W)."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
    return data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FloodDataset(Dataset):
    """
    Args:
        data_root   : path containing HandLabeled/ and WeaklyLabeled/ subdirs
                      e.g. "Sen1Floods11/v1.1/data/flood_events/"
        split_file  : path to split CSV or TXT file
        split       : "train" | "val" | "test"
        patch_size  : spatial size of output patches (default 512)
        aux_root    : path to directory containing DEM/Slope/HAND subdirs.
                      If None, assumes they are inside data_root/HandLabeled/.
    """

    def __init__(self,
                 data_root: str,
                 split_file: str,
                 split: str = "train",
                 patch_size: int = 512,
                 aux_root: str | None = None):
        self.data_root = data_root
        self.split     = split
        self.patch_size = patch_size
        self.aux_root  = aux_root

        # Build paths to HandLabeled and WeaklyLabeled dirs
        self.hand_dir = os.path.join(data_root, "HandLabeled")
        self.weak_dir = os.path.join(data_root, "WeaklyLabeled")

        # Transforms
        if split == "train":
            self.transforms = get_train_transforms(patch_size)
        else:
            self.transforms = get_val_transforms(patch_size)

        # Load sample list
        self.samples = self._load_split(split_file)
        if len(self.samples) == 0:
            raise RuntimeError(
                f"[FloodDataset] {split}: 0 samples loaded from {split_file}. "
                f"Check that the split file exists and contains recognizable "
                f"chip names (e.g. 'Bolivia_103757_S1Hand.tif')."
            )
        print(f"[FloodDataset] {split}: {len(self.samples)} samples loaded")

        # One-time audit of which input dirs are populated. Done per split so the
        # log is loud at the very start of training rather than failing silently
        # mid-epoch with constant-fallback aux bands.
        self._audit_dataset()

    def _audit_dataset(self):
        """Log file counts per HandLabeled/<band> subdir; warn on missing aux."""
        required = ("S1Hand", "LabelHand")
        recommended = ("DEM", "Slope", "HAND", "JRCWaterHand")
        optional = ("S2Hand",)

        def _count(subdir: str) -> int:
            p = os.path.join(self.hand_dir, subdir)
            if not os.path.isdir(p):
                return -1
            return sum(1 for f in os.listdir(p) if f.endswith(".tif"))

        print(f"[FloodDataset] {self.split}: dataset audit ({self.hand_dir})")
        for name in required + recommended + optional:
            n = _count(name)
            if n < 0:
                tag = "MISSING"
            elif n == 0:
                tag = "EMPTY  "
            else:
                tag = f"{n:5d} "
            kind = ("required" if name in required
                    else "recommended" if name in recommended
                    else "optional")
            print(f"  - {name:14s} {tag}  [{kind}]")

        # Hard-fail on missing required dirs.
        for name in required:
            if _count(name) <= 0:
                raise RuntimeError(
                    f"[FloodDataset] required band directory '{name}' is "
                    f"missing or empty under {self.hand_dir}. "
                    f"Cannot train without S1Hand + LabelHand."
                )

        # Loud warning if any aux band is silently missing — this is exactly
        # the failure that produced the previous conflated-class problem.
        for name in recommended:
            if _count(name) <= 0:
                print(f"  ⚠️  WARNING: '{name}' is missing — "
                      f"dataloader will silently substitute a constant plane "
                      f"and the 3-class label can't separate permanent water "
                      f"properly. Run fetch_aux_bands.py before training.")

    @staticmethod
    def _extract_base_id(val: str) -> str | None:
        """
        Extract base ID from a value like 'Bolivia_103757_S1Hand.tif'.
        Returns 'Bolivia_103757' or None if not a recognizable sample name.
        """
        val = val.strip()
        if not val:
            return None
        # Remove .tif extension
        val = val.replace(".tif", "")
        # Strip known suffixes
        for suffix in ("_S1Hand", "_S1Weak", "_S2Hand", "_S2Weak",
                        "_LabelHand", "_JRCWaterHand", "_S1OtsuLabelHand"):
            val = val.replace(suffix, "")
        # Should be left with something like "Bolivia_103757"
        if val and "_" in val:
            return val
        return None

    def _load_split(self, split_file: str) -> list[str]:
        """
        Read split file → list of sample base IDs (e.g. 'Bolivia_103757').

        Supports multiple formats:
          - CSV with header: scans all columns for values containing S1/Label refs
          - CSV without header: tries first column
          - Plain TXT: one stem per line

        Returns base IDs with suffixes stripped (e.g. 'Bolivia_103757').
        """
        samples = []

        if split_file.endswith(".csv"):
            with open(split_file, newline="") as f:
                # Peek to detect if there's a header
                first_line = f.readline().strip()
                f.seek(0)

                reader = csv.reader(f)
                header = next(reader, None)

                # Check if the first row looks like a header or data
                is_header = header and any(
                    h.lower() in ("s1hand", "label", "s1", "image", "filename",
                                  "s1_hand", "file", "name")
                    for h in header
                )

                rows_to_process = list(reader)
                if not is_header and header:
                    # First row was data, not a header — include it
                    rows_to_process.insert(0, header)

                for row in rows_to_process:
                    # Try each cell in the row to find a sample name
                    for cell in row:
                        base_id = self._extract_base_id(cell)
                        if base_id:
                            samples.append(base_id)
                            break
        else:
            # Plain text file: one stem per line
            with open(split_file) as f:
                for ln in f:
                    base_id = self._extract_base_id(ln)
                    if base_id:
                        samples.append(base_id)

        return samples

    def _get_hand_path(self, base_id: str, subdir: str, suffix_tag: str) -> str:
        """Build path like: HandLabeled/S1Hand/Bolivia_103757_S1Hand.tif"""
        return os.path.join(self.hand_dir, subdir,
                            f"{base_id}_{suffix_tag}.tif")

    def _get_weak_path(self, base_id: str, subdir: str, suffix_tag: str) -> str:
        """Build path like: WeaklyLabeled/S1Weak/Bolivia_18962_S1Weak.tif"""
        return os.path.join(self.weak_dir, subdir,
                            f"{base_id}_{suffix_tag}.tif")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        base_id = self.samples[idx]

        # --- Load Sentinel-1 (2 bands: VV, VH) ---
        s1_path = self._get_hand_path(base_id, "S1Hand", "S1Hand")
        if not os.path.exists(s1_path):
            s1_path = self._get_weak_path(base_id, "S1Weak", "S1Weak")

        s1 = read_tif_all(s1_path)   # (2, H, W)
        vv = normalize_band(s1[0], "s1_vv")
        vh = normalize_band(s1[1], "s1_vh")

        # --- Load auxiliary bands ---
        # _load_aux_raw returns the raw float32 array (or None if missing).
        # We need raw JRC for label derivation, so loading raw and normalizing
        # separately keeps both uses in sync without double I/O.
        def _load_aux_raw(subdir: str, suffix_tag: str) -> np.ndarray | None:
            p = os.path.join(self.hand_dir, subdir,
                             f"{base_id}_{suffix_tag}.tif")
            if os.path.exists(p):
                return read_tif(p)
            if self.aux_root:
                p2 = os.path.join(self.aux_root, subdir,
                                  f"{base_id}_{suffix_tag}.tif")
                if os.path.exists(p2):
                    return read_tif(p2)
            return None

        def _norm_or_fallback(raw: np.ndarray | None, band_name: str,
                              fallback_val: float) -> np.ndarray:
            if raw is None:
                return np.full_like(vv, fallback_val, dtype=np.float32)
            return normalize_band(raw, band_name)

        dem_raw   = _load_aux_raw("DEM",          "DEM")
        slope_raw = _load_aux_raw("Slope",        "Slope")
        jrc_raw   = _load_aux_raw("JRCWaterHand", "JRCWaterHand")
        hand_raw  = _load_aux_raw("HAND",         "HAND")

        dem   = _norm_or_fallback(dem_raw,   "dem",   0.5)
        slope = _norm_or_fallback(slope_raw, "slope", 0.0)
        jrc   = _norm_or_fallback(jrc_raw,   "jrc",   0.0)
        hand  = _norm_or_fallback(hand_raw,  "hand",  0.5)

        # --- Stack 6 channels: (H, W, 6) for albumentations ---
        H, W = vv.shape
        image = np.stack([vv, vh, dem, slope, jrc, hand], axis=-1).astype(np.float32)

        # --- Load flood label ---
        label_path = self._get_hand_path(base_id, "LabelHand", "LabelHand")
        if not os.path.exists(label_path):
            # Try weakly-labeled (no direct LabelWeak in dataset, but handle gracefully)
            label_path = self._get_weak_path(base_id, "S1OtsuLabelWeak",
                                             "S1OtsuLabelWeak")
        label_hand = read_tif(label_path).astype(np.int64)   # {-1, 0, 1}
        # Derive 3-class label using raw JRC: class 2 = permanent water
        label = build_three_class_label(label_hand, jrc_raw)

        # --- Load MNDWI target (from S2, optional) ---
        mndwi = None
        s2_path = self._get_hand_path(base_id, "S2Hand", "S2Hand")
        if not os.path.exists(s2_path):
            s2_path = self._get_weak_path(base_id, "S2Weak", "S2Weak")
        if os.path.exists(s2_path):
            s2 = read_tif_all(s2_path)   # expects (C, H, W) with SWIR=band1, Green=band2
            if s2.shape[0] >= 2:
                swir  = s2[0].astype(np.float32)
                green = s2[1].astype(np.float32)
                denom = swir + green + 1e-8
                mndwi_raw = (green - swir) / denom
                mndwi = np.clip(mndwi_raw, 0.0, 0.5).astype(np.float32)

        # --- Apply transforms (image + mask) ---
        image_tensor, label_tensor = apply_transforms(self.transforms, image, label)

        # --- Handle MNDWI tensor ---
        if mndwi is not None:
            # Apply same spatial transforms to mndwi (treat as extra mask channel)
            _, mndwi_tensor = apply_transforms(self.transforms,
                                               np.expand_dims(mndwi, -1), label)
            mndwi_tensor = mndwi_tensor.unsqueeze(0).float()  # (1, H, W)
        else:
            mndwi_tensor = torch.zeros(1, self.patch_size, self.patch_size)

        return {
            "image":  image_tensor.float(),      # (6, H, W)
            "label":  label_tensor,              # (H, W)  LongTensor {-1, 0, 1}
            "mndwi":  mndwi_tensor,              # (1, H, W) float
            "stem":   base_id,
        }


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def build_datasets(data_root: str,
                   splits_root: str,
                   patch_size: int = 512,
                   aux_root: str | None = None) -> dict[str, FloodDataset]:
    """
    Build train/val/test datasets using Sen1Floods11 split files.

    Args:
        data_root   : path to directory containing HandLabeled/ subdirectory.
                      e.g. on Kaggle: '/kaggle/working/sen1floods11'
                      e.g. full path: 'Sen1Floods11/v1.1/data/flood_events'
        splits_root : path to directory containing flood_train_data.csv,
                      flood_valid_data.csv, flood_test_data.csv.
                      e.g. on Kaggle: '/kaggle/working/sen1floods11/splits'
        patch_size  : crop size (default 512)
        aux_root    : optional path to DEM/Slope/HAND auxiliary data directory

    Supports both CSV split files (actual Sen1Floods11 layout) and
    legacy TXT split files.
    """
    datasets = {}

    # Try CSV files first (actual Sen1Floods11 structure), fall back to TXT
    train_csv = os.path.join(splits_root, "flood_train_data.csv")
    val_csv   = os.path.join(splits_root, "flood_valid_data.csv")
    test_csv  = os.path.join(splits_root, "flood_test_data.csv")

    if os.path.exists(train_csv):
        train_file = train_csv
        val_file   = val_csv
        test_file  = test_csv
    else:
        # Legacy TXT format
        train_file = os.path.join(splits_root, "flood_handlabeled_train.txt")
        val_file   = os.path.join(splits_root, "flood_handlabeled_S1_64px_valid.txt")
        test_file  = os.path.join(splits_root, "flood_handlabeled_S1_64px_test.txt")

    datasets["train"] = FloodDataset(data_root, train_file, "train",
                                     patch_size, aux_root)
    datasets["val"]   = FloodDataset(data_root, val_file,   "val",
                                     patch_size, aux_root)
    datasets["test"]  = FloodDataset(data_root, test_file,  "test",
                                     patch_size, aux_root)
    return datasets


# ---------------------------------------------------------------------------
# Quick test (run without dataset — shape check only)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os
    import rasterio
    from rasterio.transform import from_bounds

    # Create a minimal dummy GeoTIFF for testing
    def _write_dummy(path, n_bands, H=512, W=512, dtype="float32"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        profile = {
            "driver": "GTiff", "height": H, "width": W,
            "count": n_bands, "dtype": dtype,
            "crs": "EPSG:4326",
            "transform": from_bounds(0, 0, 1, 1, W, H),
        }
        with rasterio.open(path, "w", **profile) as dst:
            for i in range(1, n_bands + 1):
                dst.write(np.random.rand(H, W).astype(dtype), i)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_id = "TestRegion_001"
        # Create HandLabeled directory structure matching real dataset
        hand_dir = os.path.join(tmpdir, "HandLabeled")
        _write_dummy(os.path.join(hand_dir, "S1Hand",
                                  f"{base_id}_S1Hand.tif"), 2)
        label_arr = np.random.choice([-1, 0, 1], (512, 512)).astype("int16")
        lpath = os.path.join(hand_dir, "LabelHand",
                             f"{base_id}_LabelHand.tif")
        os.makedirs(os.path.dirname(lpath), exist_ok=True)
        profile = {"driver": "GTiff", "height": 512, "width": 512, "count": 1,
                   "dtype": "int16", "crs": "EPSG:4326",
                   "transform": from_bounds(0, 0, 1, 1, 512, 512)}
        with rasterio.open(lpath, "w", **profile) as dst:
            dst.write(label_arr, 1)

        # Write a JRC seasonality band so the 3-class label can include class 2
        jrc_arr = np.random.randint(0, 13, (512, 512)).astype("int16")
        jpath = os.path.join(hand_dir, "JRCWaterHand",
                             f"{base_id}_JRCWaterHand.tif")
        os.makedirs(os.path.dirname(jpath), exist_ok=True)
        with rasterio.open(jpath, "w", **profile) as dst:
            dst.write(jrc_arr, 1)

        split_path = os.path.join(tmpdir, "split.txt")
        with open(split_path, "w") as f:
            f.write(f"{base_id}_S1Hand\n")

        ds = FloodDataset(tmpdir, split_path, split="val", patch_size=512)
        sample = ds[0]
        print(f"image : {sample['image'].shape} {sample['image'].dtype}")
        print(f"label : {sample['label'].shape} {sample['label'].dtype}")
        print(f"mndwi : {sample['mndwi'].shape}")

        unique = sorted(torch.unique(sample["label"]).tolist())
        print(f"label unique values: {unique}")
        assert set(unique).issubset({-1, 0, 1, 2}), \
            f"3-class label expected in {{-1,0,1,2}}, got {unique}"
