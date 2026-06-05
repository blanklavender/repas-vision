"""
DIAGNOSTIC: Figure out exactly where point cloud alignment breaks.

Test 1: Cardinal angles (0, 90, 180, 270) — does the validated logic hold for 4 frames?
Test 2: Fine angles near 0 (0, 5, 10, 15, 20) — does small-step alignment work?
Test 3: Every 30 degrees (0, 30, 60, ..., 330) — 12 frames
Test 4: All 72 frames

Each test saves a PLY with distinct colors per frame so you can see
which frames are misaligned in MeshLab.

If Test 1 works but Test 2 fails → the turntable isn't stepping exactly 5°
If Test 1 fails at 180/270 → pivot accuracy degrades at large angles
If all tests fail → the rotation axis or pivot is wrong
"""

import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
import colorsys

DATA_DIR = Path("../../../captures/2026-03-21_flora_01_led/new_masked")
OUTPUT_DIR = Path("./diagnostics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FX, FY = 748.8987, 748.3514
CX, CY = 639.8700, 361.9517
PIVOT_Z = 0.5125
PIVOT_X = -0.007
PIVOT = np.array([PIVOT_X, 0.0, PIVOT_Z])


def depth_to_points_colors(depth_path, color_path):
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    color = cv2.imread(str(color_path))
    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64) / 1000.0
    valid = z > 0
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    colors = color_rgb[valid].astype(np.float64) / 255.0
    return points, colors


def undo_plant_rotation(points, angle_deg):
    pts = points - PIVOT
    theta = np.radians(-angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    Ry = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])
    pts_rotated = (Ry @ pts.T).T
    return pts_rotated + PIVOT


def get_color_for_angle(angle_deg, total=360):
    """Unique color per angle using HSV colormap."""
    hue = angle_deg / total
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return [r, g, b]


def build_cloud(angles, label):
    """Build aligned cloud from given angles, color-coded per frame."""
    all_points = []
    all_colors = []

    for angle_deg in angles:
        astr = f"{angle_deg:03d}"
        depth_path = DATA_DIR / f"depth_{astr}.png"
        color_path = DATA_DIR / f"color_{astr}.png"
        if not depth_path.exists():
            print(f"  MISSING: depth_{astr}.png")
            continue

        pts, cols = depth_to_points_colors(depth_path, color_path)
        pts_aligned = undo_plant_rotation(pts, angle_deg)

        # Color code by angle so you can see which frame is which
        color = get_color_for_angle(angle_deg)
        frame_colors = np.tile(color, (len(pts_aligned), 1))

        all_points.append(pts_aligned)
        all_colors.append(frame_colors)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.002)

    path = OUTPUT_DIR / f"diag_{label}.ply"
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"  Saved: {path}  ({len(np.asarray(pcd.points)):,} points)")
    return pcd


def build_cloud_real_colors(angles, label):
    """Same but with real RGB colors for visual inspection."""
    all_points = []
    all_colors = []

    for angle_deg in angles:
        astr = f"{angle_deg:03d}"
        depth_path = DATA_DIR / f"depth_{astr}.png"
        color_path = DATA_DIR / f"color_{astr}.png"
        if not depth_path.exists():
            continue

        pts, cols = depth_to_points_colors(depth_path, color_path)
        pts_aligned = undo_plant_rotation(pts, angle_deg)

        all_points.append(pts_aligned)
        all_colors.append(cols)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.002)

    path = OUTPUT_DIR / f"diag_{label}_realcolor.ply"
    o3d.io.write_point_cloud(str(path), pcd)
    print(f"  Saved: {path}  ({len(np.asarray(pcd.points)):,} points)")
    return pcd


# ============================================================
# Also check: is frame 0 alone a clean plant?
# ============================================================
print("=" * 60)
print("TEST 0: Single frame (frame 0 only) — baseline")
print("=" * 60)
build_cloud_real_colors([0], "test0_single_frame")
print("  → Should be a clean side-view of the plant")


# ============================================================
# Test 1: Cardinal angles
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: Cardinal angles (0, 90, 180, 270)")
print("  Red=0°, Green=90°, Cyan=180°, Magenta=270°")
print("=" * 60)
build_cloud([0, 90, 180, 270], "test1_cardinal")
build_cloud_real_colors([0, 90, 180, 270], "test1_cardinal")


# ============================================================
# Test 2: Fine steps near 0
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Fine steps (0, 5, 10, 15, 20)")
print("  If these don't overlap → turntable step size is wrong")
print("=" * 60)
build_cloud([0, 5, 10, 15, 20], "test2_fine_steps")
build_cloud_real_colors([0, 5, 10, 15, 20], "test2_fine_steps")


# ============================================================
# Test 3: Every 30 degrees
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Every 30° (12 frames)")
print("=" * 60)
build_cloud(list(range(0, 360, 30)), "test3_every30")
build_cloud_real_colors(list(range(0, 360, 30)), "test3_every30")


# ============================================================
# Test 4: All 72 frames
# ============================================================
print("\n" + "=" * 60)
print("TEST 4: All 72 frames")
print("=" * 60)
build_cloud(list(range(0, 360, 5)), "test4_all72")
build_cloud_real_colors(list(range(0, 360, 5)), "test4_all72")


# ============================================================
# BONUS: Check what happens WITHOUT undo_plant_rotation
# Just raw frame 0 + frame 5 overlaid — how much do they differ?
# ============================================================
print("\n" + "=" * 60)
print("TEST 5: Raw frames 0+5 WITHOUT rotation undo")
print("  → Shows how much the plant actually moved in 5°")
print("=" * 60)

pts0, cols0 = depth_to_points_colors(
    DATA_DIR / "depth_000.png", DATA_DIR / "color_000.png")
pts5, cols5 = depth_to_points_colors(
    DATA_DIR / "depth_005.png", DATA_DIR / "color_005.png")

pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(np.concatenate([pts0, pts5]))
raw_colors = np.concatenate([
    np.tile([1, 0, 0], (len(pts0), 1)),
    np.tile([0, 1, 0], (len(pts5), 1))
])
pcd_raw.colors = o3d.utility.Vector3dVector(raw_colors)
path = OUTPUT_DIR / "diag_test5_raw_0_vs_5.ply"
o3d.io.write_point_cloud(str(path), pcd_raw)
print(f"  Saved: {path}")
print("  Red=frame 0, Green=frame 5 (no rotation applied)")


# ============================================================
# BONUS: Print per-frame centroid after alignment
# If the centroid drifts steadily, the pivot is wrong
# ============================================================
print("\n" + "=" * 60)
print("CENTROID DRIFT CHECK")
print("  If centroids drift steadily, PIVOT is inaccurate")
print("=" * 60)

for angle_deg in range(0, 360, 15):
    astr = f"{angle_deg:03d}"
    depth_path = DATA_DIR / f"depth_{astr}.png"
    color_path = DATA_DIR / f"color_{astr}.png"
    if not depth_path.exists():
        continue
    pts, _ = depth_to_points_colors(depth_path, color_path)
    pts_aligned = undo_plant_rotation(pts, angle_deg)
    centroid = np.mean(pts_aligned, axis=0)
    print(f"  {angle_deg:3d}°  centroid: [{centroid[0]:+.4f}, {centroid[1]:+.4f}, {centroid[2]:+.4f}]")


print(f"""
{'='*60}
CHECK RESULTS IN MESHLAB:
{'='*60}
1. diag_test0_single_frame_realcolor.ply  — baseline, should be clean
2. diag_test1_cardinal.ply                — color-coded 0/90/180/270
3. diag_test1_cardinal_realcolor.ply      — same with real colors
4. diag_test2_fine_steps.ply              — 0/5/10/15/20 overlap test
5. diag_test3_every30.ply                 — 12 frames
6. diag_test4_all72.ply                   — all frames
7. diag_test5_raw_0_vs_5.ply             — raw alignment sanity check

ALSO: check the centroid drift printout above.

Report back:
  - Does test 1 (cardinal) look like one plant?
  - Does test 2 (fine steps) overlap cleanly?
  - Do centroids drift or stay stable?
{'='*60}
""") 