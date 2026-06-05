import cv2
import numpy as np
from pathlib import Path

# --- Config ---
session_dir = Path("../../captures/04-17_flora_matte_test/capture_20260417_114924/")
out_dir = session_dir / "masked_calibrated"
out_dir.mkdir(exist_ok=True)

# Color intrinsics (aligned depth uses these)
fx, fy = 748.8987, 748.3514
cx, cy = 639.8700, 361.9517

# Thresholds (mm)
max_depth = 600
x_left   = -130
x_right  =  140
y_bottom =  130

# Precompute coordinate maps once (same for all frames)
h, w = 720, 1280
u_map, v_map = np.meshgrid(np.arange(w), np.arange(h))

angles = list(range(0, 360, 5))
success = 0

for ang in angles:
    astr = f"{ang:03d}"
    color_path = session_dir / f"color_{astr}.png"
    depth_path = session_dir / f"depth_{astr}.png"

    if not color_path.exists() or not depth_path.exists():
        print(f"  [SKIP] {ang:>3}° — files missing")
        continue

    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    color = cv2.imread(str(color_path))

    Z = depth.astype(np.float64)
    X = (u_map - cx) * Z / fx
    Y = (v_map - cy) * Z / fy

    mask = (Z > 0) & (Z <= max_depth) & (X >= x_left) & (X <= x_right) & (Y <= y_bottom)

    # Save masked depth (16-bit, zeros = no data)
    depth_masked = depth.copy()
    depth_masked[~mask] = 0
    cv2.imwrite(str(out_dir / f"depth_{astr}.png"), depth_masked)

    # Save masked color (black background)
    color_masked = color.copy()
    color_masked[~mask] = 0
    cv2.imwrite(str(out_dir / f"color_{astr}.png"), color_masked)

    # Save binary mask
    cv2.imwrite(str(out_dir / f"mask_{astr}.png"), (mask * 255).astype(np.uint8))

    success += 1

print(f"\n[DONE] {success}/{len(angles)} frames masked → {out_dir}")