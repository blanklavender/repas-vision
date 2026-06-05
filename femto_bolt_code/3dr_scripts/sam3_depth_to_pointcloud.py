#!/usr/bin/env python3
"""
SAM3 Segmentation → Depth → Point Cloud Generator
===================================================
Segments objects in an RGB image using SAM3 (text or box prompts),
projects each mask into 3D using a co-registered depth image,
and exports colored PLY point clouds viewable in MeshLab.

Usage:
    python sam3_depth_to_pointcloud.py \
        --color  color_image.png \
        --depth  depth_image.png \
        --prompt "leaf" \
        --confidence 0.5 \
        --fx 612.0 --fy 612.0 --cx 320.0 --cy 240.0 \
        --depth_scale 1000.0 \
        --outdir pointclouds

    # If using a RealSense camera you can also pass intrinsics via JSON:
    python sam3_depth_to_pointcloud.py \
        --color  color.png \
        --depth  depth.png \
        --prompt "leaf" \
        --intrinsics_json intrinsics.json \
        --outdir pointclouds
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_mask, plot_bbox, COLORS


# ─────────────────────────────────────────────
# 1.  SAM3 segmentation
# ─────────────────────────────────────────────
def run_sam3_segmentation(
    color_path: str,
    prompt: str,
    confidence: float = 0.5,
    bpe_path: str | None = None,
) -> dict:
    """Run SAM3 text-prompt segmentation and return results dict."""

    image = Image.open(color_path).convert("RGB")

    # Locate BPE vocab shipped with sam3
    if bpe_path is None:
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

    # Enable bfloat16 on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path="./sam3_weights/sam3.pt",
    load_from_HF=False,)
    processor = Sam3Processor(model, confidence_threshold=confidence)
    state = processor.set_image(image)

    processor.reset_all_prompts(state)
    state = processor.set_text_prompt(state=state, prompt=prompt)

    n = len(state["scores"])
    print(f"[SAM3] Found {n} object(s) for prompt '{prompt}'")

    return state, image


# ─────────────────────────────────────────────
# 2.  Depth → 3D back-projection
# ─────────────────────────────────────────────
def depth_to_pointcloud(
    depth_img: np.ndarray,
    color_img: np.ndarray,
    mask: np.ndarray,
    fx: float, fy: float,
    cx: float, cy: float,
    depth_scale: float = 1000.0,
    min_depth: float = 0.01,
    max_depth: float = 10.0,
) -> o3d.geometry.PointCloud:
    """
    Back-project masked depth pixels into a colored Open3D point cloud.

    Parameters
    ----------
    depth_img : (H, W) uint16 or float depth image
    color_img : (H, W, 3) uint8 RGB image
    mask      : (H, W) boolean mask (True = keep)
    fx, fy    : focal lengths in pixels
    cx, cy    : principal point in pixels
    depth_scale : divisor to convert raw depth to metres  (1000 for mm→m)
    """
    h, w = depth_img.shape[:2]

    # Pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Convert depth to metres
    z = depth_img.astype(np.float64) / depth_scale

    # Apply mask and depth range filter
    valid = mask & (z > min_depth) & (z < max_depth)

    z = z[valid]
    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)

    # Back-project
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)  # (N, 3)

    # Colours normalised to [0, 1]
    colors = color_img[valid].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ─────────────────────────────────────────────
# 3.  Visualisation helper (saves a figure)
# ─────────────────────────────────────────────
def save_segmentation_vis(image: Image.Image, results: dict, out_path: str):
    """Save a matplotlib figure showing all masks + boxes."""
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    w_img, h_img = image.size
    for i in range(len(results["scores"])):
        color = COLORS[i % len(COLORS)]
        plot_mask(results["masks"][i].squeeze(0).cpu(), color=color)
        plot_bbox(
            h_img, w_img,
            results["boxes"][i].cpu(),
            text=f"obj_{i}  {results['scores'][i]:.2f}",
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[VIS] Saved segmentation overlay → {out_path}")


# ─────────────────────────────────────────────
# 4.  Main pipeline
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SAM3 segmentation + depth → per-mask point clouds (.ply)"
    )
    parser.add_argument("--color", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to depth image (16-bit PNG)")
    parser.add_argument("--prompt", required=True, help="Text prompt for SAM3 (e.g. 'leaf')")
    parser.add_argument("--confidence", type=float, default=0.5)

    # Camera intrinsics (manual)
    parser.add_argument("--fx", type=float, default=748.89, help="Focal length x (px)")
    parser.add_argument("--fy", type=float, default=748.35, help="Focal length y (px)")
    parser.add_argument("--cx", type=float, default=639.86, help="Principal point x (defaults to W/2)")
    parser.add_argument("--cy", type=float, default=361.95, help="Principal point y (defaults to H/2)")
    parser.add_argument("--depth_scale", type=float, default=1000.0,
                        help="Divisor to convert depth values to metres (1000 for mm)")
    parser.add_argument("--min_depth", type=float, default=0.01, help="Min valid depth (m)")
    parser.add_argument("--max_depth", type=float, default=10.0, help="Max valid depth (m)")

    # Or load from a RealSense-style JSON
    parser.add_argument("--intrinsics_json", default=None,
                        help="JSON file with keys: fx, fy, cx, cy, depth_scale")

    parser.add_argument("--outdir", default="pointclouds", help="Output directory")
    parser.add_argument("--save_combined", action="store_true",
                        help="Also save a single PLY with all masks combined")
    parser.add_argument("--save_full_scene", action="store_true",
                        help="Also save unmasked full-scene point cloud")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load images ──────────────────────────
    color_bgr = cv2.imread(args.color, cv2.IMREAD_COLOR)
    if color_bgr is None:
        sys.exit(f"ERROR: cannot read color image: {args.color}")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    depth_raw = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        sys.exit(f"ERROR: cannot read depth image: {args.depth}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]  # take first channel if multi-channel

    h, w = depth_raw.shape[:2]
    print(f"[IO] Color shape: {color_rgb.shape}  Depth shape: {depth_raw.shape}")

    # Resize colour to match depth if needed
    if color_rgb.shape[:2] != depth_raw.shape[:2]:
        print(f"[IO] Resizing color {color_rgb.shape[:2]} → depth {depth_raw.shape[:2]}")
        color_rgb = cv2.resize(color_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    # ── Intrinsics ───────────────────────────
    if args.intrinsics_json:
        with open(args.intrinsics_json) as f:
            intr = json.load(f)
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr.get("cx", w / 2.0), intr.get("cy", h / 2.0)
        depth_scale = intr.get("depth_scale", args.depth_scale)
    else:
        fx, fy = args.fx, args.fy
        cx = args.cx if args.cx is not None else w / 2.0
        cy = args.cy if args.cy is not None else h / 2.0
        depth_scale = args.depth_scale

    print(f"[CAM] fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}  scale={depth_scale}")

    # ── Run SAM3 ─────────────────────────────
    results, pil_image = run_sam3_segmentation(
        args.color, args.prompt, args.confidence
    )

    n_masks = len(results["scores"])
    if n_masks == 0:
        sys.exit("[SAM3] No objects detected — try a different prompt or lower confidence.")

    # Save visualisation
    vis_path = os.path.join(args.outdir, "segmentation_overlay.png")
    save_segmentation_vis(pil_image, results, vis_path)

    # ── Generate per-mask point clouds ───────
    all_pcds = []
    for i in range(n_masks):
        mask_tensor = results["masks"][i].squeeze(0).cpu().numpy()  # (H_mask, W_mask)

        # Resize mask to depth dimensions if needed
        if mask_tensor.shape != depth_raw.shape:
            mask_tensor = cv2.resize(
                mask_tensor.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            mask_tensor = mask_tensor.astype(bool)

        score = results["scores"][i]
        if isinstance(score, torch.Tensor):
            score = score.item()

        pcd = depth_to_pointcloud(
            depth_raw, color_rgb, mask_tensor,
            fx, fy, cx, cy,
            depth_scale, args.min_depth, args.max_depth,
        )

        n_pts = len(pcd.points)
        ply_path = os.path.join(args.outdir, f"mask_{i:02d}_score{score:.2f}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"[PLY] mask {i}  score={score:.2f}  points={n_pts:,}  → {ply_path}")

        all_pcds.append(pcd)

    # ── Optional: combined + full-scene ──────
    if args.save_combined and all_pcds:
        combined = o3d.geometry.PointCloud()
        for p in all_pcds:
            combined += p
        comb_path = os.path.join(args.outdir, "all_masks_combined.ply")
        o3d.io.write_point_cloud(comb_path, combined)
        print(f"[PLY] Combined all masks → {comb_path}  ({len(combined.points):,} pts)")

    if args.save_full_scene:
        full_mask = np.ones(depth_raw.shape[:2], dtype=bool)
        full_pcd = depth_to_pointcloud(
            depth_raw, color_rgb, full_mask,
            fx, fy, cx, cy,
            depth_scale, args.min_depth, args.max_depth,
        )
        full_path = os.path.join(args.outdir, "full_scene.ply")
        o3d.io.write_point_cloud(full_path, full_pcd)
        print(f"[PLY] Full scene → {full_path}  ({len(full_pcd.points):,} pts)")

    print(f"\n✓ Done. Open PLY files in MeshLab:  meshlab {args.outdir}/*.ply")


if __name__ == "__main__":
    main()