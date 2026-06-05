import cv2
import numpy as np

depth = cv2.imread("../../captures/04-17_flora_matte_test/capture_20260417_115206/aligned_depth_raw_20260417_115206.png", cv2.IMREAD_UNCHANGED)

# Depth intrinsics
fx, fy = 748.8987, 748.3513
cx, cy = 639.8699, 361.9516
h, w = depth.shape
print(f"Depth image size: {w}x{h}")

# Build 3D coordinate maps (in mm)
u = np.arange(w)
v = np.arange(h)
u_map, v_map = np.meshgrid(u, v)

Z = depth.astype(np.float64)
X = (u_map - cx) * Z / fx   # left/right from camera center
Y = (v_map - cy) * Z / fy   # up/down from camera center

# --- Tune these thresholds (all in mm) ---
max_depth = 600       # Z: max distance from camera
x_left   = -130       # X: left boundary (negative = left of center)
x_right  =  140       # X: right boundary
y_bottom =  130        # Y: bottom cutoff (positive = below optical center)
# ------------------------------------------

mask = (
    (Z > 0) &
    (Z <= max_depth) &
    (X >= x_left) &
    (X <= x_right) &
    (Y <= y_bottom)
)

# Apply and save
depth_masked = depth.copy()
depth_masked[~mask] = 0

# Visualization
depth_vis = (depth_masked.astype(float) / max_depth * 255).clip(0, 255).astype(np.uint8)
cv2.imwrite("depth_000_masked.png", depth_masked)
cv2.imwrite("depth_000_masked_vis.png", depth_vis)

# Also save the mask itself for applying to color later
cv2.imwrite("mask_000.png", (mask * 255).astype(np.uint8))

print(f"X range in foreground: {X[Z>0].min():.0f} to {X[Z>0].max():.0f} mm")
print(f"Y range in foreground: {Y[Z>0].min():.0f} to {Y[Z>0].max():.0f} mm")