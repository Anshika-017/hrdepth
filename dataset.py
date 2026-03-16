"""
dataset.py — Improved IndoorStereoDataset with augmentation & validation split.

Key improvements over original:
  - Deterministic train/val split (80/20) so we can monitor generalisation.
  - Normalisation using ImageNet statistics (matches SAM2 pre-training).
  - Rich albumentations pipeline: flips, rotations, colour jitter, elastic
    deformation, grid distortion, coarse dropout — all mask-safe.
  - Patch-based sampling: crops a random 1024×1024 patch from the full-res
    4112×3008 image, dramatically increasing effective dataset diversity.
  - Binary mask thresholding to guard against anti-aliased mask edges.
  - cv2.INTER_AREA used for downsampling (avoids aliasing artefacts).
"""

import os
import random
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("[WARNING] albumentations not installed – using basic augmentation fallback.")


# ── ImageNet mean/std (SAM2 was pre-trained with these) ──────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 1024   # SAM2 native input resolution


def _build_train_transform() -> Optional[object]:
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        # ── Spatial ──────────────────────────────────────────────────────────
        A.RandomResizedCrop(
            height=IMG_SIZE, width=IMG_SIZE,
            scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1,
            rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5
        ),
        A.ElasticTransform(alpha=80, sigma=10, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),

        # ── Photometric ──────────────────────────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10,
                             sat_shift_limit=20,
                             val_shift_limit=10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),

        # ── Regularisation ───────────────────────────────────────────────────
        A.CoarseDropout(
            max_holes=8, max_height=64, max_width=64,
            min_holes=1, fill_value=0, mask_fill_value=0, p=0.3
        ),

        # ── Normalise & convert ───────────────────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ToTensorV2(),
    ])


def _build_val_transform() -> Optional[object]:
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ToTensorV2(),
    ])


def _collect_samples(root: str) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    for scene in sorted(os.listdir(root)):
        cam_dir  = os.path.join(root, scene, "camera_00")
        mask_path = os.path.join(root, scene, "mask_00.png")
        if not os.path.isdir(cam_dir) or not os.path.exists(mask_path):
            continue
        for fname in sorted(os.listdir(cam_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                samples.append((os.path.join(cam_dir, fname), mask_path))
    return samples


class IndoorDataset(Dataset):
    """
    Parameters
    ----------
    root        : path to dataset split directory (e.g. 'dataset/train')
    split       : 'train' or 'val'
    val_ratio   : fraction of scenes held out for validation
    seed        : random seed for reproducible split
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        assert split in ("train", "val"), "split must be 'train' or 'val'"

        all_samples = _collect_samples(root)

        # ── Scene-level split (avoid data leakage between cameras) ───────────
        scenes = sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                         for p, _ in all_samples})
        rng = random.Random(seed)
        rng.shuffle(scenes)
        n_val   = max(1, int(len(scenes) * val_ratio))
        val_set = set(scenes[:n_val])

        self.samples = [
            (img, mask) for img, mask in all_samples
            if (os.path.basename(os.path.dirname(os.path.dirname(img))) in val_set)
            == (split == "val")
        ]

        self.transform = (
            _build_train_transform() if split == "train"
            else _build_val_transform()
        )
        self.split = split

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_pair(self, img_path: str, mask_path: str):
        img  = cv2.imread(img_path)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Resize mask to match image spatial dimensions if needed
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Binarise (threshold at 127 to handle anti-aliased edges)
        mask = (mask > 127).astype(np.uint8)
        return img, mask

    def _basic_transform(self, img: np.ndarray, mask: np.ndarray):
        """Fallback when albumentations is not available."""
        img  = cv2.resize(img,  (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        img_t  = torch.tensor(img).permute(2, 0, 1).float()
        mask_t = torch.tensor(mask).unsqueeze(0).float()
        return img_t, mask_t

    # ── Dataset API ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        img, mask = self._load_pair(img_path, mask_path)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img_t  = augmented["image"].float()           # (3, H, W)  normalised
            mask_t = augmented["mask"].unsqueeze(0).float()  # (1, H, W)  binary
        else:
            img_t, mask_t = self._basic_transform(img, mask)

        return img_t, mask_t
