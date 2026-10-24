# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""PSNR / SSIM / LPIPS helpers and the shared coverage mask.

Library module used by eval_ablation.py and synthetic/run_experiment.py.
"""
from __future__ import annotations

import numpy as np
import cv2
import torch
import lpips

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pathlib import Path
from typing import Any, Optional, Tuple
import argparse
import json

# -------------------------
# Utility
# -------------------------

def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear RGB [0,1]."""
    threshold = 0.04045
    linear = np.where(
        img <= threshold,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4,
    )
    return linear


def load_rgb_image(path: Path) -> np.ndarray:
    """Load RGB image as float32 [0,1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def load_mask(path: Path, threshold: int = 1) -> np.ndarray:
    """Load FG mask as bool array."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask >= threshold


# -------------------------
# Metrics
# -------------------------

def compute_psnr(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    if mask is not None:
        gt = gt[mask]
        pred = pred[mask]
    return peak_signal_noise_ratio(gt, pred, data_range=1.0)


def compute_ssim(
    gt: np.ndarray,
    pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    if mask is None:
        return structural_similarity(
            gt, pred, channel_axis=2, data_range=1.0
        )

    # SSIM does not natively support sparse masks,
    # so we compute SSIM on masked images.
    gt_masked = gt * mask[..., None]
    pred_masked = pred * mask[..., None]

    return structural_similarity(
        gt_masked,
        pred_masked,
        channel_axis=2,
        data_range=1.0,
    )


class LPIPSComputer:
    def __init__(self, net: str = "alex", device: str = "cuda"):
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.loss_fn.eval()

    @torch.no_grad()
    def compute(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """
        gt, pred: [H, W, 3] in [0,1]
        """
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0)
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)

        # [0,1] -> [-1,1]
        gt_t = gt_t * 2.0 - 1.0
        pred_t = pred_t * 2.0 - 1.0

        gt_t = gt_t.to(self.device)
        pred_t = pred_t.to(self.device)

        if mask is not None:
            mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            mask_t = mask_t.to(self.device)

            gt_t = gt_t * mask_t
            pred_t = pred_t * mask_t

        dist = self.loss_fn(gt_t, pred_t)
        return float(dist.item())


# -------------------------
# Main evaluation
# -------------------------

def evaluate_pair(
    gt_path: Path,
    pred_path: Path,
    mask_path: Optional[Path] = None,
    use_linear_rgb: bool = True,
    lpips_computer: Optional[LPIPSComputer] = None,
    whole_image: bool = False,
) -> Tuple[float, float, float]:
    gt = load_rgb_image(gt_path)
    pred = load_rgb_image(pred_path)

    # if use_linear_rgb:
    #     gt = srgb_to_linear(gt)
    #     pred = srgb_to_linear(pred)
    if whole_image:
        mask = np.ones(gt.shape[:2], dtype=bool)
    else:
        mask = load_mask(mask_path) if mask_path is not None else None
        if mask is None:
            mask = (pred[..., 0] > 0) & (pred[..., 1] > 0) & (pred[..., 2] > 0)  # simple non-black mask
            mask = mask.astype(np.uint8)
            #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
            mask = mask.astype(bool)

    # cv2.imshow("GT", cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # cv2.imshow("Pred", cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # cv2.imshow("Mask", (mask.astype(np.uint8) * 255))
    # cv2.waitKey(1)
    # cv2.imwrite("debug_gt.png", cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("debug_pred.png", cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # cv2.imwrite("debug_mask.png", (mask.astype(np.uint8) * 255))


    psnr = compute_psnr(gt, pred, mask)
    ssim = compute_ssim(gt, pred, mask)

    if lpips_computer is None:
        lpips_computer = LPIPSComputer()

    lpips_val = lpips_computer.compute(gt, pred, mask)

    return psnr, ssim, lpips_val


# -------------------------
# Example usage
# -------------------------

def get_gt_path(frame_id: int, cam_id: int, gt_dir: Path) -> Path:
    gt_frame_dir = gt_dir / f"hd_{frame_id:08d}"
    gt_path = gt_frame_dir / f"k{cam_id:02d}_rgb.jpg.undist.jpg"
    return gt_path

def get_pred_path(frame_id: int, cam_id: int, pred_dir: Path) -> Path:
    pred_path = pred_dir / str(cam_id) / f"{frame_id:06d}.png"
    return pred_path

def make_mask(pred_paths):
    mask_combined = None
    for pred_path in pred_paths:
        pred = load_rgb_image(pred_path)
        mask = (pred[..., 0] > 0) & (pred[..., 1] > 0) & (pred[..., 2] > 0)  # simple non-black mask
        mask = mask.astype(np.uint8)
        #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=2)
        mask = mask.astype(bool)
        if mask_combined is None:
            mask_combined = mask
        else:
            mask_combined = mask_combined & mask
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
    return mask_combined
