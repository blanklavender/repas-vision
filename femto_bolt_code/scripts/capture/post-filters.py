#!/usr/bin/env python3
"""
Offline test: depth-gradient mask on a single capture.
Loads one color_XXX.png + depth_XXX.png pair from a session and shows
the effect of the gradient mask at a chosen threshold.

Usage:
    python post-filters.py /path/to/session 000
"""
import sys
import json
from pathlib import Path
import cv2
import numpy as np

# ---- tunables ----
GRAD_THRESH_MM_PER_PX = 15.0   # start here; try 5, 10, 20
Z_MIN_M, Z_MAX_M      = 0.20, 0.80   # plant depth range
ERODE_PX              = 0     # extra morphological erosion after gradient mask (0 = off)


def load_capture(session_dir: Path, angle_str: str):
    color = cv2.imread(str(session_dir / f'color_{angle_str}.png'), cv2.IMREAD_COLOR)
    depth_u16 = cv2.imread(str(session_dir / f'depth_{angle_str}.png'), cv2.IMREAD_UNCHANGED)
    assert depth_u16 is not None and depth_u16.dtype == np.uint16, \
        "depth PNG didn't load as uint16 — check IMREAD_UNCHANGED"
    with open(session_dir / 'meta.json') as f:
        meta = json.load(f)
    scale = meta['depth_scale_m_per_unit']   # 0.001 for Femto Bolt
    depth_m = depth_u16.astype(np.float32) * scale
    return color, depth_m, depth_u16


def gradient_mask(depth_m, z_min, z_max, grad_thresh_m_per_px, erode_px=0):
    # depth-range mask
    range_mask = (depth_m > z_min) & (depth_m < z_max)

    # gradient on valid depth only (NaN out zeros so Sobel ignores them)
    d = np.where(depth_m > 0, depth_m, np.nan)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    # reject discontinuity pixels and any NaN-tainted neighbors
    grad_mask = np.nan_to_num(grad, nan=np.inf) < grad_thresh_m_per_px

    mask = (range_mask & grad_mask).astype(np.uint8)

    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        mask = cv2.erode(mask, k, iterations=1)

    return mask.astype(bool), grad


def visualize(color, depth_m, mask, grad):
    h, w = depth_m.shape

    # baseline: depth-threshold only, no gradient filter
    baseline = ((depth_m > Z_MIN_M) & (depth_m < Z_MAX_M))

    # colored cutouts
    color_baseline = color.copy()
    color_baseline[~baseline] = 0
    color_masked = color.copy()
    color_masked[~mask] = 0

    # gradient visualization (clipped for display)
    g = np.clip(grad / (2 * GRAD_THRESH_MM_PER_PX / 1000.0), 0, 1) * 255
    grad_vis = cv2.applyColorMap(g.astype(np.uint8), cv2.COLORMAP_HOT)

    # label and stack
    def label(img, text):
        out = img.copy()
        cv2.putText(out, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    top = np.hstack([label(color, 'original color'),
                     label(color_baseline, 'depth threshold only (fringe visible)')])
    bot = np.hstack([label(grad_vis, f'|grad Z|  (bright = discontinuity)'),
                     label(color_masked, f'+ gradient mask @ {GRAD_THRESH_MM_PER_PX} mm/px')])
    return np.vstack([top, bot])


def main():
    if len(sys.argv) < 3:
        print("usage: python test_gradient_mask.py <session_dir> <angle_000|005|...>")
        sys.exit(1)
    session_dir = Path(sys.argv[1])
    angle_str = sys.argv[2]

    color, depth_m, _ = load_capture(session_dir, angle_str)
    mask, grad = gradient_mask(depth_m, Z_MIN_M, Z_MAX_M,
                               GRAD_THRESH_MM_PER_PX / 1000.0, ERODE_PX)
    vis = visualize(color, depth_m, mask, grad)

    # fit to screen
    scale = min(1.0, 1600 / vis.shape[1])
    if scale < 1.0:
        vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))

    cv2.imshow('gradient mask test', vis)
    print(f"threshold: {GRAD_THRESH_MM_PER_PX} mm/px   |   valid pixels: "
          f"{mask.sum()} / {mask.size} ({100*mask.mean():.1f}%)")
    print("press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()