import cv2
import numpy as np

FX, FY = 748.8987, 748.3514
CX, CY = 639.8700, 361.9517

for ang in [0, 180]:
    depth = cv2.imread(f"../../../captures/2026-04-02_flora_02_led/masked/depth_{ang:03d}.png", cv2.IMREAD_UNCHANGED)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64) / 1000.0
    valid = z > 0
    x = (u - CX) * z / FX
    pts = np.stack([x[valid], v[valid], z[valid]], axis=-1)  # v just for counting
    
    z_vals = z[valid]
    x_vals = x[valid]
    print(f"\n--- Frame {ang}° ---")
    print(f"  Z range: {z_vals.min():.4f} - {z_vals.max():.4f}")
    print(f"  Z centroid: {z_vals.mean():.4f}")
    print(f"  X centroid: {x_vals.mean():.4f}")

# Derive pivot
# PIVOT_Z ≈ (max_Z_frame0 + min_Z_frame180) / 2
# PIVOT_X ≈ (X_centroid_frame0 + X_centroid_frame180) / 2