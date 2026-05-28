"""
Data augmentation pipeline using Albumentations.

Paper settings (DeepSARFlood 2025):
  - Augmentations: crop, horizontal flip, vertical flip, random rotation
  - Probability per augmentation: 0.1 – 0.3
  - Non-augmented data remains the predominant part of training set

All augmentations are applied identically to both the image (6-channel)
and the corresponding flood label mask.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Training augmentation pipeline
# ---------------------------------------------------------------------------



def get_train_transforms(patch_size: int = 512) -> A.Compose:
    return A.Compose([
        A.RandomCrop(height=patch_size, width=patch_size),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.2),
        A.Rotate(limit=15, p=0.1, border_mode=0),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})



# ---------------------------------------------------------------------------
# Validation / test pipeline (no augmentation, just tensor conversion)
# ---------------------------------------------------------------------------



def get_val_transforms(patch_size: int = 512) -> A.Compose:
    return A.Compose([
        A.CenterCrop(height=patch_size, width=patch_size),
        ToTensorV2(),
    ], additional_targets={"mask": "mask"})



# ---------------------------------------------------------------------------
# Apply helper (numpy arrays)
# ---------------------------------------------------------------------------

def apply_transforms(transforms: A.Compose,
                     image: np.ndarray,
                     mask: np.ndarray | None = None) -> tuple:
    """
    Apply Albumentations transforms.

    Args:
        transforms : Albumentations Compose pipeline
        image      : (H, W, C) float32 numpy array
        mask       : (H, W)    int or float numpy array (label)
    Returns:
        image_tensor : (C, H, W) torch.Tensor
        mask_tensor  : (H, W)   torch.LongTensor  or None
    """
    if mask is not None:
        result = transforms(image=image, mask=mask)
        return result["image"], result["mask"].long()
    else:
        result = transforms(image=image)
        return result["image"], None


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    img  = np.random.rand(512, 512, 6).astype(np.float32)
    mask = np.random.randint(0, 2, (512, 512)).astype(np.int64)

    train_tf = get_train_transforms()
    img_t, mask_t = apply_transforms(train_tf, img, mask)
    print(f"Train image tensor : {img_t.shape} {img_t.dtype}")
    print(f"Train mask tensor  : {mask_t.shape} {mask_t.dtype}")

    val_tf = get_val_transforms()
    img_v, mask_v = apply_transforms(val_tf, img, mask)
    print(f"Val   image tensor : {img_v.shape}")
    print(f"Val   mask tensor  : {mask_v.shape}")
