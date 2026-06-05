#!/usr/bin/env python3
"""
Offline: visualize the gradient mask effect as a point cloud.
Shows three Open3D windows side-by-side:
  (1) baseline:   depth threshold only
  (2) filtered:   + gradient mask
  (3) rejected:   points the mask threw out (red) overlaid on kept points (natural color)

Usage:
    python view_gradient_mask_pcd.py <session_dir> <angle_str e.g. 000>
"""
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d

# ---- tunables ----
GRAD_THRESH_MM_PER_PX = 20.0
Z_MIN_M, Z_MAX_M      = 0.20, 0.80
ERODE_PX              = 0


def load_capture(session_dir: Path, angle_str: str):
    color = cv2.imread(str(session_dir / f'color_{angle_str}.png'), cv2.IMREAD_COLOR)
    depth_u16 = cv2.imread(str(session_dir / f'depth_{angle_str}.png'), cv2.IMREAD_UNCHANGED)
    assert depth_u16 is not None and depth_u16.dtype == np.uint16
    with open(session_dir / 'meta.json') as f:
        meta = json.load(f)
    return color, depth_u16.astype(np.float32) * meta['depth_scale_m_per_unit'], meta


def compute_masks(depth_m, grad_thresh_m, z_min, z_max, erode_px):
    range_mask = (depth_m > z_min) & (depth_m < z_max)

    d = np.where(depth_m > 0, depth_m, np.nan)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.nan_to_num(np.sqrt(gx*gx + gy*gy), nan=np.inf)

    grad_mask = grad < grad_thresh_m
    baseline = range_mask.copy()
    filtered = (range_mask & grad_mask)

    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1))
        filtered = cv2.erode(filtered.astype(np.uint8), k, iterations=1).astype(bool)

    rejected = baseline & ~filtered   # in baseline but killed by gradient mask
    return baseline, filtered, rejected


def backproject(depth_m, color_bgr, mask, K):
    h, w = depth_m.shape
    fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']
    ys, xs = np.where(mask)
    zs = depth_m[ys, xs]
    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    pts = np.stack([X, Y, zs], axis=1).astype(np.float32)
    cols_bgr = color_bgr[ys, xs].astype(np.float32) / 255.0
    cols_rgb = cols_bgr[:, ::-1]   # BGR -> RGB for Open3D
    return pts, cols_rgb


def make_pcd(pts, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    if len(sys.argv) < 3:
        print("usage: python view_gradient_mask_pcd.py <session_dir> <angle_000|005|...>")
        sys.exit(1)
    session = Path(sys.argv[1])
    angle = sys.argv[2]

    color, depth_m, meta = load_capture(session, angle)
    K = meta['color_intrinsics']

    baseline, filtered, rejected = compute_masks(
        depth_m, GRAD_THRESH_MM_PER_PX / 1000.0, Z_MIN_M, Z_MAX_M, ERODE_PX)

    print(f"baseline: {baseline.sum():>7}   "
          f"filtered: {filtered.sum():>7}   "
          f"rejected: {rejected.sum():>7}   "
          f"({100*rejected.sum()/max(baseline.sum(),1):.1f}% of baseline dropped)")

    # (1) baseline cloud
    pts_b, col_b = backproject(depth_m, color, baseline, K)
    pcd_baseline = make_pcd(pts_b, col_b)

    # (2) filtered cloud (what you'd keep)
    pts_f, col_f = backproject(depth_m, color, filtered, K)
    pcd_filtered = make_pcd(pts_f, col_f)

    # (3) kept (natural color) + rejected (red) overlay
    pts_r, _ = backproject(depth_m, color, rejected, K)
    col_r = np.tile(np.array([[1.0, 0.0, 0.0]], np.float32), (pts_r.shape[0], 1))
    pcd_overlay = pcd_filtered + make_pcd(pts_r, col_r)

    print("\nopening three windows. close each to proceed to the next.")
    print(" 1) baseline (depth threshold only) — fringe visible")
    o3d.visualization.draw_geometries([pcd_baseline],
        window_name="baseline  (depth threshold only)", width=900, height=800)

    print(" 2) filtered (+ gradient mask)")
    o3d.visualization.draw_geometries([pcd_filtered],
        window_name=f"filtered  (grad mask @ {GRAD_THRESH_MM_PER_PX} mm/px)",
        width=900, height=800)

    print(" 3) kept (natural color) + rejected (red)")
    o3d.visualization.draw_geometries([pcd_overlay],
        window_name="rejected points shown in red", width=900, height=800)


if __name__ == '__main__':
    main()