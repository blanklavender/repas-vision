"""
Two-Stage Leaf Segmentation Pipeline
=====================================
Stage 1: Single track point → whole plant mask (EfficientSAM ViT-T)
Stage 2: Automatic mask generation within plant region → individual leaf masks
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
from torchvision.transforms import ToTensor
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


# ── CONFIG ──────────────────────────────────────────────────────────────────

@dataclass
class Config:
    efficient_sam_repo_path: str = r"../../../EfficientSAM"
    model_type: str = "vits"  # "vitt" or "vits"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    image_path: str = r"../../../captures/03-04_FLORA_captures/capture_20260304_124617_01/color_20260304_124617.png"
    track_point: Tuple[int, int] = (618, 408)  # (x, y) on the plant

    # Stage 2: automatic mask generation parameters
    points_per_side: int = 32         # grid density for auto prompting
    pred_iou_thresh: float = 0.7      # minimum predicted IoU to keep a mask
    stability_score_thresh: float = 0.8
    min_mask_region_area: int = 200    # discard tiny fragments (pixels)


# ── EfficientSAM Segmenter ─────────────────────────────────────────────────

class EfficientSAMSegmenter:
    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model()

    def _load_model(self):
        repo_path = Path(self.config.efficient_sam_repo_path).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"EfficientSAM repo not found at {repo_path}. "
                "Clone: git clone https://github.com/yformer/EfficientSAM.git"
            )
        sys.path.insert(0, str(repo_path))
        orig_dir = os.getcwd()
        os.chdir(repo_path)
        try:
            from efficient_sam.build_efficient_sam import (
                build_efficient_sam_vitt, build_efficient_sam_vits
            )
            model = (build_efficient_sam_vitt() if self.config.model_type == "vitt"
                     else build_efficient_sam_vits())
            model = model.to(self.config.device).eval()
            print(f"[OK] EfficientSAM ({self.config.model_type}) on {self.config.device}")
            return model
        finally:
            os.chdir(orig_dir)

    def _run_model(self, image_rgb: np.ndarray, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Run model and return the best mask (highest IoU)."""
        img_t = ToTensor()(image_rgb).to(self.config.device)
        pts = torch.tensor(points).reshape(1, 1, -1, 2).float().to(self.config.device)
        lbl = torch.tensor(labels).reshape(1, 1, -1).int().to(self.config.device)
        with torch.no_grad():
            logits, iou = self.model(img_t[None], pts, lbl)
        sorted_ids = torch.argsort(iou, dim=-1, descending=True)
        logits = torch.take_along_dim(logits, sorted_ids[..., None, None], dim=2)
        return torch.ge(logits[0, 0, 0], 0).cpu().numpy()

    def _run_model_with_iou(self, image_rgb: np.ndarray, points: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run model and return (best_mask, best_iou_score)."""
        img_t = ToTensor()(image_rgb).to(self.config.device)
        pts = torch.tensor(points).reshape(1, 1, -1, 2).float().to(self.config.device)
        lbl = torch.tensor(labels).reshape(1, 1, -1).int().to(self.config.device)
        with torch.no_grad():
            logits, iou = self.model(img_t[None], pts, lbl)
        sorted_ids = torch.argsort(iou, dim=-1, descending=True)
        logits = torch.take_along_dim(logits, sorted_ids[..., None, None], dim=2)
        best_mask = torch.ge(logits[0, 0, 0], 0).cpu().numpy()
        best_iou = iou.sort(dim=-1, descending=True).values[0, 0, 0].item()
        return best_mask, best_iou

    def segment_with_points(self, image_rgb: np.ndarray, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self._run_model(image_rgb, points, labels)

    def segment_with_box(self, image_rgb: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return self._run_model(image_rgb, np.array([[x1, y1], [x2, y2]]), np.array([2, 3]))


# ── STAGE 1: Plant mask from single track point ────────────────────────────

def stage1_plant_mask(
    segmenter: EfficientSAMSegmenter,
    image_bgr: np.ndarray,
    point_xy: Tuple[int, int],
) -> np.ndarray:
    """Segment the whole plant with a single foreground point prompt."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    points = np.array([[point_xy[0], point_xy[1]]])
    labels = np.array([1])  # foreground

    plant_mask = segmenter.segment_with_points(image_rgb, points, labels).astype(np.uint8)

    # Clean up with morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)

    return plant_mask


# ── STAGE 2: Automatic leaf masks within plant region ──────────────────────

def stage2_leaf_masks(
    segmenter: EfficientSAMSegmenter,
    image_bgr: np.ndarray,
    plant_mask: np.ndarray,
    config: Config,
) -> List[np.ndarray]:
    """
    Automatic mask generation constrained to the plant region.

    Strategy: lay a grid of point prompts inside the plant mask,
    run each through the model, filter and deduplicate.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = plant_mask.shape

    # ── Generate grid points inside the plant mask ──
    xs = np.linspace(0, w - 1, config.points_per_side, dtype=int)
    ys = np.linspace(0, h - 1, config.points_per_side, dtype=int)
    grid_points = [(x, y) for y in ys for x in xs if plant_mask[y, x] > 0]
    print(f"[Stage 2] {len(grid_points)} grid points inside plant mask")

    # ── Run each point prompt and collect candidate masks ──
    candidates = []  # list of (mask, iou_score)

    for px, py in grid_points:
        points = np.array([[px, py]])
        labels = np.array([1])

        mask, iou = segmenter._run_model_with_iou(image_rgb, points, labels)
        mask = mask.astype(np.uint8) & plant_mask  # constrain to plant region

        if iou < config.pred_iou_thresh:
            continue
        if mask.sum() < config.min_mask_region_area:
            continue

        candidates.append((mask, iou))

    print(f"[Stage 2] {len(candidates)} candidate masks after IoU/area filtering")

    # ── Deduplicate via NMS on masks ──
    leaf_masks = _mask_nms(candidates, iou_threshold=0.5, stability_thresh=config.stability_score_thresh)

    # Filter out any mask that covers nearly the entire plant
    plant_area = plant_mask.sum()
    leaf_masks = [m for m in leaf_masks if m.sum() < 0.85 * plant_area]

    print(f"[Stage 2] {len(leaf_masks)} final leaf masks after NMS + whole-plant filter")
    return leaf_masks


def _mask_nms(
    candidates: List[Tuple[np.ndarray, float]],
    iou_threshold: float = 0.5,
    stability_thresh: float = 0.8,
) -> List[np.ndarray]:
    """Greedy non-maximum suppression on binary masks, sorted by IoU score."""
    candidates.sort(key=lambda x: x[1], reverse=True)

    kept: List[np.ndarray] = []
    for mask, score in candidates:
        if score < stability_thresh:
            continue
        suppress = False
        for existing in kept:
            intersection = (mask & existing).sum()
            union = (mask | existing).sum()
            if union > 0 and intersection / union > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(mask)
    return kept


# ── VISUALIZATION ───────────────────────────────────────────────────────────

def visualize(
    image_bgr: np.ndarray,
    plant_mask: np.ndarray,
    leaf_masks: List[np.ndarray],
    point_xy: Tuple[int, int],
    save_path: str = None,
):
    """Show plant mask, individual leaf masks overlaid on the color image."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: original + track point
    axes[0].imshow(image_rgb)
    axes[0].plot(point_xy[0], point_xy[1], "r*", markersize=15, markeredgecolor="white")
    axes[0].set_title("Input + Track Point")
    axes[0].axis("off")

    # Panel 2: plant mask overlay
    overlay_plant = image_rgb.copy()
    green_tint = np.zeros_like(image_rgb)
    green_tint[:, :, 1] = 255
    overlay_plant[plant_mask > 0] = (
        0.6 * overlay_plant[plant_mask > 0] + 0.4 * green_tint[plant_mask > 0]
    ).astype(np.uint8)
    axes[1].imshow(overlay_plant)
    axes[1].set_title("Stage 1: Plant Mask")
    axes[1].axis("off")

    # Panel 3: leaf masks overlaid with distinct colors
    overlay_leaves = image_rgb.copy().astype(np.float64)
    n_leaves = len(leaf_masks)
    for i, lmask in enumerate(leaf_masks):
        hue = i / max(n_leaves, 1)
        color_rgb = np.array(hsv_to_rgb((hue, 0.9, 1.0))) * 255
        color_layer = np.zeros_like(image_rgb, dtype=np.float64)
        color_layer[:] = color_rgb
        region = lmask > 0
        overlay_leaves[region] = 0.55 * overlay_leaves[region] + 0.45 * color_layer[region]

    overlay_leaves_uint8 = overlay_leaves.astype(np.uint8)
    for lmask in leaf_masks:
        contours, _ = cv2.findContours(lmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_leaves_uint8, contours, -1, (255, 255, 255), 1)

    axes[2].imshow(overlay_leaves_uint8)
    axes[2].set_title(f"Stage 2: {n_leaves} Leaf Segments")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved visualization → {save_path}")
    plt.show()


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    config = Config()
    print(f"Device: {config.device}")

    image_bgr = cv2.imread(config.image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {config.image_path}")
    print(f"Image shape: {image_bgr.shape}")

    segmenter = EfficientSAMSegmenter(config)

    # Stage 1: whole plant
    print("\n── Stage 1: Plant segmentation ──")
    plant_mask = stage1_plant_mask(segmenter, image_bgr, config.track_point)
    plant_area = plant_mask.sum()
    total_px = image_bgr.shape[0] * image_bgr.shape[1]
    print(f"Plant mask area: {plant_area} px  ({100 * plant_area / total_px:.1f}% of image)")

    # Stage 2: leaf-level masks within plant
    print("\n── Stage 2: Leaf segmentation (automatic) ──")
    leaf_masks = stage2_leaf_masks(segmenter, image_bgr, plant_mask, config)

    # Visualize
    save_path = config.image_path.rsplit(".", 1)[0] + "_leaf_segmentation.png"
    visualize(image_bgr, plant_mask, leaf_masks, config.track_point, save_path=save_path)


if __name__ == "__main__":
    main()