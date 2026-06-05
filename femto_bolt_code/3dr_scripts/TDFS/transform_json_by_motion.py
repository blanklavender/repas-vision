"""
AprilTag-based frame-to-frame alignment.
No turntable model assumptions — just measures the actual
rigid body motion between consecutive frames using shared tags.
"""
import cv2
import numpy as np
from pupil_apriltags import Detector
from pathlib import Path
import open3d as o3d
import json

DATA_DIR = Path("../../../captures/2026-04-02_flora_02_led/masked")  # ← update
OUTPUT_DIR = Path("./tag_aligned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FX, FY = 748.8987, 748.3514
CX, CY = 639.8700, 361.9517
W, H = 1280, 720
ANGLES = list(range(0, 360, 5))
VOXEL_SIZE = 0.002

detector = Detector(families="tag36h11")


def get_tag_corners_3d(angle_deg):
    """Returns dict: tag_id -> (4, 3) array of 3D corner positions."""
    astr = f"{angle_deg:03d}"
    color = cv2.imread(str(DATA_DIR / f"color_{astr}.png"))
    depth = cv2.imread(str(DATA_DIR / f"depth_{astr}.png"), cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    tags = detector.detect(gray)
    result = {}

    for tag in tags:
        corners_3d = []
        valid = True
        for corner in tag.corners:
            u, v = int(round(corner[0])), int(round(corner[1]))
            if 0 <= u < W and 0 <= v < H:
                z_mm = depth[v, u]
                if z_mm > 0:
                    z = z_mm / 1000.0
                    x = (u - CX) * z / FX
                    y = (v - CY) * z / FY
                    corners_3d.append([x, y, z])
                else:
                    valid = False; break
            else:
                valid = False; break
        if valid and len(corners_3d) == 4:
            result[tag.tag_id] = np.array(corners_3d)

    return result


def rigid_transform_svd(A, B):
    """
    Find R, t such that B ≈ R @ A + t
    (transforms points from frame A's configuration to frame B's)
    Using SVD-based least squares.
    """
    assert A.shape == B.shape
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, d])  # ensure proper rotation
    R = Vt.T @ sign_matrix @ U.T
    t = centroid_B - R @ centroid_A
    
    # Residual
    transformed = (R @ A.T).T + t
    err = np.linalg.norm(transformed - B, axis=1).mean()
    return R, t, err


# ============================================================
# STEP 1: Detect tags in all frames
# ============================================================
print("Detecting AprilTags...")
all_tags = {}
for ang in ANGLES:
    tags = get_tag_corners_3d(ang)
    all_tags[ang] = tags
    tag_ids = list(tags.keys())
    print(f"  {ang:3d}°: tags {tag_ids}")

# ============================================================
# STEP 2: Chain frame-to-frame rigid transforms
#
# For each consecutive pair, find shared tags, compute R,t
# that maps frame N's tag points to frame 0's configuration.
#
# Key idea: frame0's tags are at some position. frame5's tags
# have moved (plant rotated 5°). Shared tags give us point
# correspondences → SVD gives exact rigid motion.
#
# We chain: T_0 = I, T_5 = T_{0→5}, T_10 = T_{0→5} @ T_{5→10}, ...
# But we want to UN-rotate, so we accumulate the INVERSE.
# ============================================================
print("\nComputing frame-to-frame transforms...")

# T_cumulative[ang] maps points FROM frame `ang` INTO frame 0's space
T_cumulative = {0: (np.eye(3), np.zeros(3))}

prev_ang = 0
failed_pairs = []

for i in range(1, len(ANGLES)):
    curr_ang = ANGLES[i]
    
    # Find shared tag IDs — try prev_ang first, then look back
    best_ref = None
    shared_ids = set(all_tags.get(prev_ang, {}).keys()) & set(all_tags[curr_ang].keys())
    
    if shared_ids and prev_ang in T_cumulative:
        best_ref = prev_ang
    else:
        # Look back through frames that ARE in T_cumulative
        for lookback in range(2, min(i+1, 20)):
            lb_ang = ANGLES[i - lookback]
            if lb_ang not in T_cumulative:
                continue
            shared_ids = set(all_tags[lb_ang].keys()) & set(all_tags[curr_ang].keys())
            if shared_ids:
                best_ref = lb_ang
                break
    
    if best_ref is None or not shared_ids:
        print(f"  {curr_ang:3d}°: NO shared tags with any aligned frame! Skipping.")
        failed_pairs.append(curr_ang)
        continue
    
    pts_prev = np.concatenate([all_tags[best_ref][tid] for tid in shared_ids])
    pts_curr = np.concatenate([all_tags[curr_ang][tid] for tid in shared_ids])
    
    R_rel, t_rel, err = rigid_transform_svd(pts_curr, pts_prev)
    
    R_prev, t_prev = T_cumulative[best_ref]
    R_cum = R_prev @ R_rel
    t_cum = R_prev @ t_rel + t_prev
    
    T_cumulative[curr_ang] = (R_cum, t_cum)
    print(f"  {curr_ang:3d}°: ref={best_ref:3d}°, {len(shared_ids)} shared tags, "
          f"{len(pts_curr)} points, residual={err:.5f}m")
    
    prev_ang = curr_ang  # only update on success

print(f"\nSuccessful: {len(T_cumulative)}/{len(ANGLES)} frames")
if failed_pairs:
    print(f"Failed: {failed_pairs}")


# ============================================================
# STEP 3: Build merged point cloud using measured transforms
# ============================================================
print("\nBuilding merged point cloud...")

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

all_points, all_colors = [], []

for ang in ANGLES:
    if ang not in T_cumulative:
        continue
    astr = f"{ang:03d}"
    pts, cols = depth_to_points_colors(
        DATA_DIR / f"depth_{astr}.png", DATA_DIR / f"color_{astr}.png")
    
    R, t = T_cumulative[ang]
    pts_aligned = (R @ pts.T).T + t
    
    all_points.append(pts_aligned)
    all_colors.append(cols)

all_points = np.concatenate(all_points)
all_colors = np.concatenate(all_colors)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd.colors = o3d.utility.Vector3dVector(all_colors)
pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

debug_path = OUTPUT_DIR / "debug_tag_aligned.ply"
o3d.io.write_point_cloud(str(debug_path), pcd)
print(f"Saved: {debug_path}")
print(f">>> Check this in MeshLab — should be ONE clean plant")


# ============================================================
# STEP 4: Generate nerfstudio poses from measured transforms
# ============================================================
print("\nGenerating nerfstudio poses...")

# Same axis remap as before
CAM_TO_WORLD = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
])

# Center: use frame 0 plant centroid
pts0, _ = depth_to_points_colors(
    DATA_DIR / "depth_000.png", DATA_DIR / "color_000.png")
plant_center = pts0.mean(axis=0)

frames = []
for ang in ANGLES:
    if ang not in T_cumulative:
        continue
    astr = f"{ang:03d}"
    
    R_align, t_align = T_cumulative[ang]
    # T_cumulative maps plant points from frame N → frame 0
    # Equivalent camera motion: camera N = inverse(T_cumulative) applied to camera 0
    # Camera 0 is at origin in camera space
    
    # Camera N position: where the camera "would be" if plant was static
    # cam_N = R_align^T @ (cam_0 - t_align)
    R_inv = R_align.T
    t_inv = -R_align.T @ t_align
    cam_pos_camspace = t_inv  # camera 0 is at origin
    
    # Center relative to plant
    cam_centered = cam_pos_camspace - plant_center
    cam_world = CAM_TO_WORLD @ cam_centered
    
    # Camera orientation in world space
    # The camera's rotation matrix in camera space: R_inv
    # Remap to world: CAM_TO_WORLD @ R_inv @ CAM_TO_WORLD^T
    R_cam_world = CAM_TO_WORLD @ R_inv @ CAM_TO_WORLD.T
    
    # Build c2w (OpenGL convention: -Z forward, +Y up)
    c2w = np.eye(4)
    c2w[:3, :3] = R_cam_world
    c2w[:3, 3] = cam_world
    
    frame = {
        "file_path": f"images/color_{astr}.png",
        "transform_matrix": c2w.tolist(),
        "depth_file_path": f"depth/depth_{astr}.png",
    }
    mask_path = DATA_DIR / f"mask_{astr}.png"
    if mask_path.exists():
        frame["mask_path"] = f"masks/mask_{astr}.png"
    frames.append(frame)

# Also save recentered point cloud
pts_world = (CAM_TO_WORLD @ (np.asarray(pcd.points) - plant_center).T).T
pcd_world = o3d.geometry.PointCloud()
pcd_world.points = o3d.utility.Vector3dVector(pts_world)
pcd_world.colors = pcd.colors
o3d.io.write_point_cloud(str(OUTPUT_DIR / "points3D.ply"), pcd_world)

transforms = {
    "fl_x": FX, "fl_y": FY,
    "cx": CX, "cy": CY,
    "w": W, "h": H,
    "camera_model": "OPENCV",
    "frames": frames,
}
with open(OUTPUT_DIR / "transforms.json", "w") as f:
    json.dump(transforms, f, indent=2)

print(f"transforms.json: {len(frames)} frames")
print(f"\n>>> First check debug_tag_aligned.ply in MeshLab")
print(f">>> If it's clean, train:")
print(f">>>   ns-train splatfacto --max-num-iterations 1000 \\")
print(f">>>       nerfstudio-data --data {OUTPUT_DIR} \\")
print(f">>>       --orientation-method none --center-method none --auto-scale-poses False")