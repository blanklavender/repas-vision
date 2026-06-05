"""
OPTIMIZE PIVOT for turntable point cloud alignment.

Uses consecutive frame pairs (0°+5°, 5°+10°, etc.) and finds the
PIVOT_X, PIVOT_Z that minimizes alignment error across all pairs.

The idea: if the pivot is correct, undo_plant_rotation with the correct
pivot will perfectly align consecutive frames. If the pivot is wrong,
there'll be a residual shift. We search for the pivot that minimizes
the total residual.

Method: scipy.optimize.minimize on the mean nearest-neighbor distance
between consecutive aligned frame pairs.
"""

import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from scipy.optimize import minimize
from scipy.spatial import cKDTree

DATA_DIR = Path("../../../captures/2026-03-21_flora_01_led/new_masked")
OUTPUT_DIR = Path("./diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FX, FY = 748.8987, 748.3514
CX, CY = 639.8700, 361.9517

# Starting pivot (your current values)
PIVOT_X_INIT = -0.007
PIVOT_Z_INIT = 0.5125

# Use a subset of frame pairs for speed (every 15° = 24 pairs)
PAIR_STEP = 15
PAIRS = [(a, a + PAIR_STEP) for a in range(0, 360 - PAIR_STEP, PAIR_STEP)]


def depth_to_points(depth_path, color_path):
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64) / 1000.0
    valid = z > 0
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    return points


def undo_rotation(points, angle_deg, pivot_x, pivot_z):
    """Undo plant rotation with given pivot."""
    pivot = np.array([pivot_x, 0.0, pivot_z])
    pts = points - pivot
    theta = np.radians(-angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    Ry = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])
    pts_rotated = (Ry @ pts.T).T
    return pts_rotated + pivot


# ============================================================
# PRELOAD all frames we need
# ============================================================
print("Loading frames...")
frame_points = {}
for a, b in PAIRS:
    for angle in [a, b]:
        if angle not in frame_points:
            astr = f"{angle:03d}"
            depth_path = DATA_DIR / f"depth_{astr}.png"
            color_path = DATA_DIR / f"color_{astr}.png"
            if depth_path.exists():
                pts = depth_to_points(depth_path, color_path)
                # Downsample for speed — keep every 10th point
                frame_points[angle] = pts[::10]
                print(f"  {angle:3d}°: {len(frame_points[angle]):,} points (subsampled)")

print(f"Loaded {len(frame_points)} frames for {len(PAIRS)} pairs")


# ============================================================
# COST FUNCTION: mean nearest-neighbor distance across pairs
# ============================================================
def alignment_cost(params):
    pivot_x, pivot_z = params
    total_dist = 0.0
    count = 0

    for a, b in PAIRS:
        if a not in frame_points or b not in frame_points:
            continue

        # Align both frames to frame 0 space using candidate pivot
        pts_a = undo_rotation(frame_points[a], a, pivot_x, pivot_z)
        pts_b = undo_rotation(frame_points[b], b, pivot_x, pivot_z)

        # Mean nearest-neighbor distance (a → b)
        tree = cKDTree(pts_b)
        dists, _ = tree.query(pts_a, k=1)

        # Use median instead of mean — robust to outliers from
        # different visible regions
        total_dist += np.median(dists)
        count += 1

    return total_dist / count if count > 0 else 1e10


# ============================================================
# OPTIMIZE
# ============================================================
print("\n" + "=" * 60)
print("OPTIMIZING PIVOT")
print("=" * 60)

# Evaluate starting point
initial_cost = alignment_cost([PIVOT_X_INIT, PIVOT_Z_INIT])
print(f"Initial pivot: X={PIVOT_X_INIT:.4f}, Z={PIVOT_Z_INIT:.4f}")
print(f"Initial cost:  {initial_cost:.6f} m (median NN distance)")

# Run optimization
print("\nRunning Nelder-Mead optimization...")
result = minimize(
    alignment_cost,
    x0=[PIVOT_X_INIT, PIVOT_Z_INIT],
    method='Nelder-Mead',
    options={'xatol': 0.0001, 'fatol': 1e-7, 'maxiter': 500, 'disp': True}
)

opt_x, opt_z = result.x
opt_cost = result.fun
print(f"\nOptimized pivot: X={opt_x:.6f}, Z={opt_z:.6f}")
print(f"Optimized cost:  {opt_cost:.6f} m")
print(f"Improvement:     {initial_cost - opt_cost:.6f} m ({(1 - opt_cost/initial_cost)*100:.1f}%)")
print(f"Pivot change:    dX={opt_x - PIVOT_X_INIT:+.4f}, dZ={opt_z - PIVOT_Z_INIT:+.4f}")


# ============================================================
# ALSO try grid search around current values for robustness
# ============================================================
print("\n" + "=" * 60)
print("GRID SEARCH (backup verification)")
print("=" * 60)

best_cost = initial_cost
best_x, best_z = PIVOT_X_INIT, PIVOT_Z_INIT

# Search ±5cm around current pivot in 5mm steps
x_range = np.arange(PIVOT_X_INIT - 0.05, PIVOT_X_INIT + 0.05, 0.005)
z_range = np.arange(PIVOT_Z_INIT - 0.05, PIVOT_Z_INIT + 0.05, 0.005)

total_evals = len(x_range) * len(z_range)
print(f"Evaluating {total_evals} grid points...")

for i, px in enumerate(x_range):
    for pz in z_range:
        cost = alignment_cost([px, pz])
        if cost < best_cost:
            best_cost = cost
            best_x, best_z = px, pz
    if (i + 1) % 5 == 0:
        print(f"  {(i+1)*len(z_range)}/{total_evals} evaluated, best so far: X={best_x:.4f} Z={best_z:.4f} cost={best_cost:.6f}")

print(f"\nGrid search best: X={best_x:.6f}, Z={best_z:.6f}, cost={best_cost:.6f}")


# ============================================================
# USE BEST RESULT
# ============================================================
# Pick whichever method gave better result
if opt_cost <= best_cost:
    final_x, final_z, final_cost = opt_x, opt_z, opt_cost
    method = "Nelder-Mead"
else:
    final_x, final_z, final_cost = best_x, best_z, best_cost
    method = "Grid search"

print(f"\n{'='*60}")
print(f"BEST PIVOT ({method}): X={final_x:.6f}, Z={final_z:.6f}")
print(f"Cost: {final_cost:.6f} m")
print(f"{'='*60}")


# ============================================================
# BUILD VALIDATION CLOUDS with optimized pivot
# ============================================================
print("\nBuilding validation point clouds with optimized pivot...")

PIVOT_NEW = np.array([final_x, 0.0, final_z])

def build_validation_cloud(angles, label, pivot_x, pivot_z):
    all_pts = []
    all_cols = []
    for angle_deg in angles:
        astr = f"{angle_deg:03d}"
        depth_path = DATA_DIR / f"depth_{astr}.png"
        color_path = DATA_DIR / f"color_{astr}.png"
        if not depth_path.exists():
            continue
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        color = cv2.imread(str(color_path))
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float64) / 1000.0
        valid = z > 0
        x = (u - CX) * z / FX
        y = (v - CY) * z / FY
        pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
        cols = color_rgb[valid].astype(np.float64) / 255.0

        pts_aligned = undo_rotation(pts, angle_deg, pivot_x, pivot_z)
        all_pts.append(pts_aligned)
        all_cols.append(cols)

    all_pts = np.concatenate(all_pts, axis=0)
    all_cols = np.concatenate(all_cols, axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_cols)
    pcd = pcd.voxel_down_sample(voxel_size=0.002)
    path = OUTPUT_DIR / f"optimized_{label}.ply"
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"  {path} ({len(np.asarray(pcd.points)):,} points)")
    return pcd


# Fine steps comparison: old pivot vs new pivot
build_validation_cloud([0, 5, 10, 15, 20], "fine_OLD_pivot", PIVOT_X_INIT, PIVOT_Z_INIT)
build_validation_cloud([0, 5, 10, 15, 20], "fine_NEW_pivot", final_x, final_z)

# Full 72 frames with optimized pivot
build_validation_cloud(list(range(0, 360, 5)), "all72_NEW_pivot", final_x, final_z)

# Centroid drift with new pivot
print("\nCentroid drift with OPTIMIZED pivot:")
for angle_deg in range(0, 360, 15):
    astr = f"{angle_deg:03d}"
    depth_path = DATA_DIR / f"depth_{astr}.png"
    color_path = DATA_DIR / f"color_{astr}.png"
    if not depth_path.exists():
        continue
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64) / 1000.0
    valid = z > 0
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    pts_aligned = undo_rotation(pts, angle_deg, final_x, final_z)
    centroid = np.mean(pts_aligned, axis=0)
    print(f"  {angle_deg:3d}°  centroid: [{centroid[0]:+.4f}, {centroid[1]:+.4f}, {centroid[2]:+.4f}]")


print(f"""
{'='*60}
RESULTS
{'='*60}
Old pivot: X={PIVOT_X_INIT:.4f}, Z={PIVOT_Z_INIT:.4f}  cost={initial_cost:.6f}
New pivot: X={final_x:.6f}, Z={final_z:.6f}  cost={final_cost:.6f}

COMPARE IN MESHLAB:
  optimized_fine_OLD_pivot.ply  ← your current pivot (should look drifted)
  optimized_fine_NEW_pivot.ply  ← optimized pivot (should overlap better)
  optimized_all72_NEW_pivot.ply ← full reconstruction

If the new pivot looks good, update your scripts:
  PIVOT_X = {final_x:.6f}
  PIVOT_Z = {final_z:.6f}
{'='*60}
""")