
"""
edge_overlay_check.py — D2C alignment diagnostic.

Renders three views from an aligned RGB + uint16 depth pair:
    1. Colorized depth alone.
    2. Color image overlaid on colorized depth at user-defined alpha.
    3. Color Canny edges drawn on colorized depth — the most diagnostic view:
       misalignment shows as color edges sitting off real depth boundaries
       (e.g. leaf silhouette edges that don't ride the depth discontinuity).

Usage:
    python edge_overlay_check.py --color color_000.png --depth depth_000.png
    python edge_overlay_check.py --color color_000.png --depth depth_000.png \
        --depth-min 300 --depth-max 600 --alpha 0.5

Assumes:
    - Color is a standard 8-bit BGR image (PNG/JPG).
    - Depth is a uint16 PNG with values in millimetres (Femto Bolt /
      RealSense default after hardware D2C).
    - After D2C alignment, color and depth share the same resolution and
      intrinsics, so corresponding pixels are at the same (u, v).
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def colorize_depth(depth_mm, dmin=None, dmax=None, cmap=cv2.COLORMAP_TURBO):
    """uint16 depth in mm -> BGR uint8 colorized; invalid (zero) stays black."""
    valid = depth_mm > 0
    if not valid.any():
        return np.zeros((*depth_mm.shape, 3), np.uint8), 0.0, 0.0
    if dmin is None:
        dmin = float(np.percentile(depth_mm[valid], 2))
    if dmax is None:
        dmax = float(np.percentile(depth_mm[valid], 98))
    norm = np.clip((depth_mm.astype(np.float32) - dmin) / max(dmax - dmin, 1e-6), 0, 1)
    out = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    out[~valid] = 0
    return out, dmin, dmax


def edge_overlay_check(color_path, depth_path, dmin=None, dmax=None,
                       alpha=0.5, edge_color=(255, 255, 255),
                       canny_lo=80, canny_hi=160):
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)        # BGR uint8
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)    # uint16, mm
    if color is None:
        raise FileNotFoundError(color_path)
    if depth is None:
        raise FileNotFoundError(depth_path)
    if color.shape[:2] != depth.shape[:2]:
        raise ValueError(
            f"Shape mismatch: color {color.shape[:2]} vs depth {depth.shape[:2]}. "
            "After hardware D2C alignment, the two should match exactly."
        )

    colorized, dmin, dmax = colorize_depth(depth, dmin, dmax)

    # Alpha blend.
    blend = cv2.addWeighted(colorized, 1.0 - alpha, color, alpha, 0)

    # Canny on greyscale color, drawn over colorized depth in a flat color.
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    edge_view = colorized.copy()
    edge_view[edges > 0] = edge_color

    to_rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(to_rgb(colorized))
    axes[0].set_title(f"Colorized depth [{dmin:.0f}–{dmax:.0f} mm]")
    axes[1].imshow(to_rgb(blend))
    axes[1].set_title(f"Color overlay @ alpha={alpha:.2f}")
    axes[2].imshow(to_rgb(edge_view))
    axes[2].set_title("Color Canny edges on depth\n(misalignment = edges off boundaries)")
    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--color", required=True, type=Path,
                   help="Aligned color image path (PNG/JPG).")
    p.add_argument("--depth", required=True, type=Path,
                   help="Aligned depth image path (uint16 PNG, mm).")
    p.add_argument("--depth-min", type=float, default=None,
                   help="Colormap min in mm (default: 2nd percentile of valid depth).")
    p.add_argument("--depth-max", type=float, default=None,
                   help="Colormap max in mm (default: 98th percentile of valid depth).")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Color image alpha in the blend (0..1).")
    args = p.parse_args()
    edge_overlay_check(args.color, args.depth,
                       dmin=args.depth_min, dmax=args.depth_max, alpha=args.alpha)


if __name__ == "__main__":
    main()