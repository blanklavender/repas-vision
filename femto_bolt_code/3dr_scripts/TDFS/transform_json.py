"""
STEP-BY-STEP point cloud + transforms generation.
Each step saves a debug PLY so you can verify in MeshLab.
Run this, check each debug file, tell me which step breaks.
"""

import json
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
import os

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("../../../captures/2026-04-02_flora_02_led/masked")
OUTPUT_DIR = Path("./transformed_TSDF")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FX, FY = 748.8987, 748.3514
CX, CY = 639.8700, 361.9517
W, H = 1280, 720

PIVOT_Z = 0.5638 # 0.549 
PIVOT_X = -0.0142 # -0.016
PIVOT = np.array([PIVOT_X, 0.0, PIVOT_Z])

ANGLES = list(range(0, 360, 5))
VOXEL_SIZE = 0.002


# ============================================================
# YOUR VALIDATED FUNCTIONS (unchanged)
# ============================================================
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


# ============================================================
# STEP 1: Build cloud in frame 0 camera space
#         EXACTLY like your validated 0+90 test, but all 72 frames
# ============================================================
print("=" * 60)
print("STEP 1: Build cloud using validated undo_plant_rotation")
print("        (frame 0 camera space, no other transforms)")
print("=" * 60)

all_points = []
all_colors = []

for angle_deg in ANGLES:
    astr = f"{angle_deg:03d}"
    depth_path = DATA_DIR / f"depth_{astr}.png"
    color_path = DATA_DIR / f"color_{astr}.png"
    if not depth_path.exists() or not color_path.exists():
        continue

    pts, cols = depth_to_points_colors(depth_path, color_path)
    pts_aligned = undo_plant_rotation(pts, angle_deg)

    all_points.append(pts_aligned)
    all_colors.append(cols)

all_points = np.concatenate(all_points, axis=0)
all_colors = np.concatenate(all_colors, axis=0)

pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(all_points)
pcd_raw.colors = o3d.utility.Vector3dVector(all_colors)
pcd_raw = pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)

debug1_path = OUTPUT_DIR / "debug_step1_camera_space.ply"
o3d.io.write_point_cloud(str(debug1_path), pcd_raw)

pts1 = np.asarray(pcd_raw.points)
print(f"Points: {len(pts1):,}")
print(f"Center: {np.mean(pts1, axis=0)}")
print(f"  → Plant should be near ({PIVOT_X:.3f}, ~0, {PIVOT_Z:.3f})")
print(f"Saved: {debug1_path}")
print(f"\n>>> OPEN THIS IN MESHLAB. Does it look like ONE plant?")
print(f">>> If NO — something is wrong with your input data or paths.")
print(f">>> If YES — continue to step 2.\n")


# ============================================================
# STEP 2: Center the plant at origin (subtract PIVOT)
#         Still in camera-space axes, just shifted
# ============================================================
print("=" * 60)
print("STEP 2: Center plant at origin (subtract PIVOT)")
print("=" * 60)

pts_centered = np.asarray(pcd_raw.points) - PIVOT
pcd_centered = o3d.geometry.PointCloud()
pcd_centered.points = o3d.utility.Vector3dVector(pts_centered)
pcd_centered.colors = pcd_raw.colors

debug2_path = OUTPUT_DIR / "debug_step2_centered.ply"
o3d.io.write_point_cloud(str(debug2_path), pcd_centered)

print(f"Center: {np.mean(pts_centered, axis=0)}")
print(f"  → Should be near (0, 0, 0)")
print(f"Saved: {debug2_path}")
print(f"\n>>> Should look identical to step 1, just shifted.\n")


# ============================================================
# STEP 3: Remap axes for nerfstudio (Z-up world)
#
# Camera space (OpenCV): X=right, Y=down, Z=forward
# Nerfstudio world:      Z=up
#
# Mapping:
#   cam X (right)   → world X (right)
#   cam Y (down)    → world -Z (so Z = up)
#   cam Z (forward) → world Y
# ============================================================
print("=" * 60)
print("STEP 3: Remap axes to nerfstudio world (Z-up)")
print("=" * 60)

CAM_TO_WORLD = np.array([
    [1,  0,  0],   # world X = cam X
    [0,  0,  1],   # world Y = cam Z
    [0, -1,  0],   # world Z = -cam Y
])

pts_world = (CAM_TO_WORLD @ pts_centered.T).T
pcd_world = o3d.geometry.PointCloud()
pcd_world.points = o3d.utility.Vector3dVector(pts_world)
pcd_world.colors = pcd_raw.colors

debug3_path = OUTPUT_DIR / "debug_step3_nerfstudio_world.ply"
o3d.io.write_point_cloud(str(debug3_path), pcd_world)

print(f"Center: {np.mean(pts_world, axis=0)}")
print(f"Saved: {debug3_path}")
print(f"\n>>> Should look like the same plant, just rotated so it stands upright")
print(f">>> with Z pointing up. If it's mangled, CAM_TO_WORLD is wrong.\n")

# Save final point cloud for nerfstudio
final_ply_path = OUTPUT_DIR / "points3D.ply"
o3d.io.write_point_cloud(str(final_ply_path), pcd_world)
print(f"Final point cloud: {final_ply_path}")


# ============================================================
# STEP 4: Camera poses
#
# Camera 0 in camera space: at origin = (0, 0, 0)
# Centered: (0,0,0) - PIVOT = (0.007, 0, -0.5125)
# In world: CAM_TO_WORLD @ centered = (0.007, -0.5125, 0)
#
# For frame N: plant rotated by angle_deg on turntable
# Equivalent: camera orbited by angle_deg around the plant
#
# The orbit is around camera-space Y axis (which becomes world Z after remap).
# So in world space, cameras orbit around Z axis in the XY plane. Good — that's
# what nerfstudio expects with Z-up.
#
# World-space orbit rotation (around Z):
#   Rz(angle) = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Camera poses in nerfstudio world space")
print("=" * 60)

# Camera 0 position in world
cam0_centered = np.array([0.0, 0.0, 0.0]) - PIVOT   # = (0.007, 0, -0.5125)
cam0_world = CAM_TO_WORLD @ cam0_centered             # = (0.007, -0.5125, 0)
print(f"Camera 0 world pos: {cam0_world}")

frames = []

for angle_deg in ANGLES:
    astr = f"{angle_deg:03d}"
    if not (DATA_DIR / f"color_{astr}.png").exists():
        continue

    # Orbit camera around world Z axis
    # (Z axis in world = Y axis in camera space = the vertical/gravity axis)
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

    camN_world = Rz @ cam0_world

    # Look-at orientation (OpenGL: -Z = forward, +Y = up)
    forward = -camN_world / np.linalg.norm(camN_world)  # toward origin
    z_cam = -forward    # OpenGL +Z = backward
    world_up = np.array([0.0, 0.0, 1.0])
    x_cam = np.cross(world_up, z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)

    c2w = np.eye(4)
    c2w[:3, 0] = x_cam
    c2w[:3, 1] = y_cam
    c2w[:3, 2] = z_cam
    c2w[:3, 3] = camN_world

    frame = {
        "file_path": f"images/color_{astr}.png",
        "transform_matrix": c2w.tolist(),
        "depth_file_path": f"depth/depth_{astr}.png",
    }
    mask_path = DATA_DIR / f"mask_{astr}.png"
    if mask_path.exists():
        frame["mask_path"] = f"masks/mask_{astr}.png"

    frames.append(frame)

# Sanity checks
T0 = np.array(frames[0]['transform_matrix'])
T18 = np.array(frames[18]['transform_matrix'])
pos0, pos18 = T0[:3, 3], T18[:3, 3]
dot = np.dot(pos0, pos18) / (np.linalg.norm(pos0) * np.linalg.norm(pos18))

print(f"\nFrame 0  pos:  {pos0}")
print(f"Frame 18 pos:  {pos18}")
print(f"Frame 0  look: {-T0[:3, 2]}")
print(f"Frame 0  up:   {T0[:3, 1]}")
print(f"Radius:        {np.linalg.norm(pos0):.4f} m")
print(f"Angle F0→F18:  {np.degrees(np.arccos(np.clip(dot, -1, 1))):.1f}°")


# ============================================================
# WRITE transforms.json
# ============================================================
transforms = {
    "fl_x": FX, "fl_y": FY,
    "cx": CX, "cy": CY,
    "w": W, "h": H,
    "camera_model": "OPENCV",
    "frames": frames,
}

transforms_path = OUTPUT_DIR / "transforms.json"
with open(transforms_path, "w") as f:
    json.dump(transforms, f, indent=2)
print(f"\ntransforms.json: {len(frames)} frames → {transforms_path}")


# ============================================================
# SYMLINK FILES
# ============================================================
for subdir in ["images", "depth", "masks"]:
    (OUTPUT_DIR / subdir).mkdir(exist_ok=True)

for angle_deg in ANGLES:
    astr = f"{angle_deg:03d}"
    for fname, subdir in [("color", "images"), ("depth", "depth"), ("mask", "masks")]:
        src = DATA_DIR / f"{fname}_{astr}.png"
        dst = OUTPUT_DIR / subdir / f"{fname}_{astr}.png"
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)


# ============================================================
# SUMMARY
# ============================================================
print(f"""
{'='*60}
DEBUG FILES (check each in MeshLab):
  1. {debug1_path}  ← validated camera space, MUST look correct
  2. {debug2_path}       ← same but centered at origin
  3. {debug3_path}   ← axis remapped for nerfstudio

NERFSTUDIO FILES:
  {final_ply_path}
  {transforms_path}

TRAIN:
  ns-train splatfacto \\
      --max-num-iterations 1000 \\
      nerfstudio-data \\
      --data {OUTPUT_DIR} \\
      --orientation-method none \\
      --center-method none \\
      --auto-scale-poses False

If rotation is flipped: change theta = np.radians(-angle_deg) in STEP 4.
{'='*60}
""")