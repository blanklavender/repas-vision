"""
AprilTag-based pivot optimizer.
Detects tags across all frames, uses depth-derived 3D corners
to find the pivot (X, Z) that minimizes alignment error.
"""
import cv2
import numpy as np
from pupil_apriltags import Detector
from scipy.optimize import minimize
from pathlib import Path

DATA_DIR = Path("../../../captures/2026-04-02_flora_02_led/masked")  # ← update this
FX, FY = 748.8987, 748.3514
CX, CY = 639.8700, 361.9517
ANGLES = list(range(0, 360, 5))

detector = Detector(families="tag36h11")

def get_tag_corners_3d(angle_deg):
    """Detect tags, return their corners as 3D points via depth lookup."""
    astr = f"{angle_deg:03d}"
    color = cv2.imread(str(DATA_DIR / f"color_{astr}.png"))
    depth = cv2.imread(str(DATA_DIR / f"depth_{astr}.png"), cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    
    tags = detector.detect(gray)
    points_3d = []
    
    for tag in tags:
        for corner in tag.corners:
            u, v = int(round(corner[0])), int(round(corner[1]))
            if 0 <= u < 1280 and 0 <= v < 720:
                z_mm = depth[v, u]
                if z_mm > 0:
                    z = z_mm / 1000.0
                    x = (u - CX) * z / FX
                    y = (v - CY) * z / FY
                    points_3d.append([x, y, z])
    
    return np.array(points_3d) if points_3d else None


def undo_rotation(pts, angle_deg, pivot_x, pivot_z):
    """Rotate points back by -angle around pivot."""
    pivot = np.array([pivot_x, 0.0, pivot_z])
    pts_c = pts - pivot
    theta = np.radians(-angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (Ry @ pts_c.T).T + pivot


def alignment_error(params):
    """
    For each consecutive frame pair, un-rotate both to frame 0,
    measure how well nearby points agree.
    """
    px, pz = params
    all_aligned = {}
    
    for angle_deg in ANGLES:
        pts = tag_data.get(angle_deg)
        if pts is None:
            continue
        aligned = undo_rotation(pts, angle_deg, px, pz)
        all_aligned[angle_deg] = aligned
    
    # Compare all frames against frame 0's neighborhood
    if 0 not in all_aligned or len(all_aligned) < 10:
        return 1e6
    
    ref = all_aligned[0]
    total_err = 0
    count = 0
    
    for ang, pts in all_aligned.items():
        if ang == 0:
            continue
        # For each point, find nearest neighbor in ref
        for p in pts:
            dists = np.linalg.norm(ref - p, axis=1)
            total_err += dists.min()
            count += 1
    
    return total_err / max(count, 1)


# Collect tag detections from all frames
print("Detecting AprilTags across all frames...")
tag_data = {}
for angle_deg in ANGLES:
    pts = get_tag_corners_3d(angle_deg)
    if pts is not None:
        tag_data[angle_deg] = pts
        print(f"  {angle_deg:3d}°: {len(pts)} corner points")
    else:
        print(f"  {angle_deg:3d}°: no tags detected")

print(f"\nTags detected in {len(tag_data)}/{len(ANGLES)} frames")

# Grid search first (coarse), then refine with Nelder-Mead
print("\nGrid search...")
best_err = 1e9
best_pz, best_px = 0.549, -0.016  # start from centroid estimate

for pz in np.arange(0.50, 0.60, 0.005):
    for px in np.arange(-0.05, 0.02, 0.005):
        err = alignment_error([px, pz])
        if err < best_err:
            best_err = err
            best_px, best_pz = px, pz

print(f"Grid best: PIVOT_X={best_px:.4f}, PIVOT_Z={best_pz:.4f}, err={best_err:.6f}")

# Refine
print("\nRefining with Nelder-Mead...")
result = minimize(alignment_error, [best_px, best_pz], method='Nelder-Mead',
                  options={'xatol': 0.0001, 'fatol': 1e-8, 'maxiter': 5000})

opt_px, opt_pz = result.x
print(f"\nOptimized: PIVOT_X={opt_px:.4f}, PIVOT_Z={opt_pz:.4f}, err={result.fun:.6f}")
print(f"\nUpdate your script with:")
print(f"  PIVOT_Z = {opt_pz:.4f}")
print(f"  PIVOT_X = {opt_px:.4f}")