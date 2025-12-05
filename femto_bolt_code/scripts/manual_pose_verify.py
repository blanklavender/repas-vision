#!/usr/bin/env python3
"""
april_tag_vs_manual_pose_comparison.py
Compare AprilTag-based CAD placement with manually provided pose matrix.
"""
from __future__ import annotations
import math
import json
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
from pupil_apriltags import Detector

# -------------------- CONFIG --------------------
COLOR_PATH = Path(r"../../captures/captures_colorworld_stable/capture_20251121_162722/color_20251121_162722.png")
PLY_PATH   = Path(r"../../captures/captures_colorworld_stable/capture_20251121_162722/final_crop/cropped_camframe_20251121_162722.ply")

INTRINSICS_JSON = Path(r"./calibration_parameters/factory_color_intrinsics_2025-09-08T143506.json")

# AprilTag settings
FAMILY     = "tag36h11"
TAG_ID     = -1
TAG_SIZE_M = 0.0303

# Visualization settings
AXES   = 0.05
SPHERE = 0.003
GRID   = 0.10
VOXEL  = 0.0

# Point cloud scaling 
# NOTE: RealSense point clouds are typically exported in METERS already
# Only change this if you know your point cloud is in different units
POINT_CLOUD_SCALE = 1.0  # 1.0 = already in meters, 0.001 = mm to meters

# CAD placement options
CAD_PLY                 = Path(r"../../cad_model/StructureOnly.PLY")
CAD_UNITS_TO_METERS     = 0.001
CAD_PRE_ROT_DEG_ZYX     = (0.0, 0.0, 0.0)

# Manual pose matrix (4x4 transformation matrix)
MANUAL_POSE_MATRIX = np.array([
    [9.995342493057250977e-01, 3.035889752209186554e-02, -3.081407397985458374e-03, 1.527985185384750366e-01],
    [-2.985795028507709503e-02, 9.938570261001586914e-01, 1.065677180886268616e-01, 3.250595331192016602e-01],
    [6.297726649791002274e-03, -1.064260751008987427e-01, 9.943006634712219238e-01, 7.968250513076782227e-01],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
], dtype=np.float64)

# # old WITH RESERVOIR AND LIGHTBOX Manual pose matrix (4x4 transformation matrix)
# MANUAL_POSE_MATRIX = np.array([
#     [9.975081682205200195e-01, 1.404522359371185303e-02, 6.913902610540390015e-02, 3.381231129169464111e-01],
#     [-1.310083549469709396e-02, 9.998147487640380859e-01, -1.409446820616722107e-02, 3.594959378242492676e-01],
#     [-6.932418793439865112e-02, 1.315351855009794235e-02, 9.975074529647827148e-01, 9.210860133171081543e-01],
#     [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
# ], dtype=np.float64)

# -------------------- Helper Functions --------------------
def load_color_intrinsics(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"Intrinsics JSON not found: {json_path}")
    data = json.loads(json_path.read_text())
    intr = data["color_intrinsics"] if (isinstance(data, dict) and "color_intrinsics" in data) else data
    for k in ("fx","fy","cx","cy"):
        if k not in intr:
            raise KeyError(f"Missing '{k}' in intrinsics JSON: {json_path}")
    fx = float(intr["fx"]); fy = float(intr["fy"])
    cx = float(intr["cx"]); cy = float(intr["cy"])
    iw = int(intr.get("width", 0)); ih = int(intr.get("height", 0))
    return fx, fy, cx, cy, iw, ih

def scale_intrinsics(fx, fy, cx, cy, src_w, src_h, dst_w, dst_h):
    if src_w <= 0 or src_h <= 0 or (src_w == dst_w and src_h == dst_h):
        return fx, fy, cx, cy
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    return fx*sx, fy*sy, cx*sx, cy*sy

def build_K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0,  fy, cy],
                     [0,   0,  1]], dtype=np.float64)

def detect_tag_center(img_bgr: np.ndarray, family: str, prefer_id: int = -1):
    det = Detector(families=family, nthreads=2, quad_decimate=1.0,
                   quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tags = det.detect(gray, estimate_tag_pose=False)
    if not tags:
        raise RuntimeError('No AprilTags detected')
    if prefer_id >= 0:
        for t in tags:
            tid = getattr(t, 'tag_id', getattr(t, 'id', -999))
            if tid == prefer_id:
                return (float(t.center[0]), float(t.center[1])), t
        raise RuntimeError(f'Requested tag id {prefer_id} not found')
    def area(tt):
        c = tt.corners.astype(np.float32)
        return float(cv2.contourArea(c))
    t = max(tags, key=area)
    return (float(t.center[0]), float(t.center[1])), t

def compute_reproj_error(obj_pts: np.ndarray, img_pts: np.ndarray,
                         rvec: np.ndarray, tvec: np.ndarray,
                         K: np.ndarray, dist: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - img_pts, axis=1)))

def solve_pnp_with_best_obj_order(img_corners_px: np.ndarray, K: np.ndarray,
                                  dist: np.ndarray, tag_size_m: float):
    half = float(tag_size_m) / 2.0
    TL = np.array([-half, -half, 0.0], dtype=np.float64)
    TR = np.array([+half, -half, 0.0], dtype=np.float64)
    BR = np.array([+half, +half, 0.0], dtype=np.float64)
    BL = np.array([-half, +half, 0.0], dtype=np.float64)

    candidates = [
        (["TL","TR","BR","BL"], [TL, TR, BR, BL]),
        (["TR","BR","BL","TL"], [TR, BR, BL, TL]),
        (["BR","BL","TL","TR"], [BR, BL, TL, TR]),
        (["BL","TL","TR","BR"], [BL, TL, TR, BR]),
        (["TR","TL","BL","BR"], [TR, TL, BL, BR]),
        (["TL","BL","BR","TR"], [TL, BL, BR, TR]),
        (["BL","BR","TR","TL"], [BL, BR, TR, TL]),
        (["BR","TR","TL","BL"], [BR, TR, TL, BL]),
    ]

    best = None
    best_score = np.inf
    best_err = None
    best_label = None

    for label, obj_list in candidates:
        obj_pts = np.array(obj_list, dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_corners_px, K, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            continue
        err = compute_reproj_error(obj_pts, img_corners_px, rvec, tvec, K, dist)
        z = float(tvec[2])
        score = err + (1000.0 if z <= 0 else 0.0)
        if score < best_score:
            best_score = score
            best = (obj_pts, rvec, tvec)
            best_err = err
            best_label = ",".join(label)

    if best is None:
        raise RuntimeError("solvePnP failed for all candidate corner orders.")
    obj_pts, rvec, tvec = best
    return obj_pts, rvec, tvec, best_err, best_label

def euler_zyx_to_R(z_deg: float, y_deg: float, x_deg: float) -> np.ndarray:
    z, y, x = [math.radians(a) for a in (z_deg, y_deg, x_deg)]
    cz, sz = math.cos(z), math.sin(z)
    cy, sy = math.cos(y), math.sin(y)
    cx, sx = math.cos(x), math.sin(x)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    return Rz @ Ry @ Rx

def load_cad_geometry(path: Path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh and (len(np.asarray(mesh.vertices)) > 0):
        mesh.compute_vertex_normals()
        return mesh
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd and (len(np.asarray(pcd.points)) > 0):
        return pcd
    raise RuntimeError(f"Failed to load CAD from {path}")

def prepare_cad_model(cad_path: Path):
    """
    Load and prepare CAD model with scaling and optional pre-rotation.
    Follows the same workflow as the working multi-tag script.
    """
    cad = load_cad_geometry(cad_path)
    
    # Scale from CAD units to meters (always at origin)
    S = float(CAD_UNITS_TO_METERS)
    if S != 1.0:
        cad.scale(S, center=(0, 0, 0))
        print(f"[CAD] Scaled by {S} (CAD units to meters)")
    
    # Optional pre-rotation (applied at origin before placement)
    if any(abs(a) > 1e-6 for a in CAD_PRE_ROT_DEG_ZYX):
        Rpre = euler_zyx_to_R(*CAD_PRE_ROT_DEG_ZYX)
        cad.rotate(Rpre, center=(0, 0, 0))
        print(f"[CAD] Applied pre-rotation ZYX: {CAD_PRE_ROT_DEG_ZYX}")
    
    return cad

def make_xy_grid(cell: float, n: int = 20, z: float = 0.0):
    extent = n * cell
    pts, lines, colors = [], [], []
    for y in np.linspace(-extent, extent, 2*n+1):
        pts += [[-extent, y, z], [extent, y, z]]
        lines.append([len(pts)-2, len(pts)-1]); colors.append([0.7,0.7,0.7])
    for x in np.linspace(-extent, extent, 2*n+1):
        pts += [[x, -extent, z], [x, extent, z]]
        lines.append([len(pts)-2, len(pts)-1]); colors.append([0.7,0.7,0.7])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts))
    ls.lines  = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return ls

def create_labeled_sphere(position: np.ndarray, color: list, radius: float):
    """Create a colored sphere at given position."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate(position)
    return sphere

# -------------------- Main --------------------
def main():
    # Load image
    if not COLOR_PATH.exists(): 
        raise FileNotFoundError(COLOR_PATH)
    if not INTRINSICS_JSON.exists(): 
        raise FileNotFoundError(INTRINSICS_JSON)
    
    img = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if img is None: 
        raise RuntimeError("Failed to read COLOR image")
    h, w = img.shape[:2]

    # Load intrinsics
    fx0, fy0, cx0, cy0, iw, ih = load_color_intrinsics(INTRINSICS_JSON)
    fx, fy, cx, cy = (scale_intrinsics(fx0, fy0, cx0, cy0, iw, ih, w, h)
                      if (iw and ih) else (fx0, fy0, cx0, cy0))
    print(f"[INTR] Using JSON intrinsics for {w}x{h}")
    print(f"       fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")

    # Detect AprilTag and compute pose
    (cx_pix, cy_pix), tag_center = detect_tag_center(img, FAMILY, TAG_ID)
    
    img_pts = tag_center.corners.astype(np.float64).reshape(4, 2)
    if img_pts.max() <= 1.5:
        img_pts[:, 0] *= w
        img_pts[:, 1] *= h
        print("[FIX] Corners normalized; scaled to pixels")

    K = build_K(fx, fy, cx, cy)
    dist = np.zeros((5,1), dtype=np.float64)

    obj_pts, rvec_pnp, tvec_pnp, err_px, order_label = solve_pnp_with_best_obj_order(
        img_corners_px=img_pts, K=K, dist=dist, tag_size_m=TAG_SIZE_M
    )
    R_pnp, _ = cv2.Rodrigues(rvec_pnp)
    t_pnp = tvec_pnp.reshape(3)

    # AprilTag transformation matrix
    T_apriltag = np.eye(4, dtype=float)
    T_apriltag[:3, :3] = R_pnp
    T_apriltag[:3,  3] = t_pnp
    
    print("\n" + "="*60)
    print("[APRILTAG POSE]")
    print("="*60)
    print(T_apriltag)
    print(f"Translation: {t_pnp}")
    print(f"Reprojection error: {err_px:.2f} px")
    
    print("\n" + "="*60)
    print("[MANUAL POSE]")
    print("="*60)
    print(MANUAL_POSE_MATRIX)
    print(f"Translation: {MANUAL_POSE_MATRIX[:3, 3]}")
    
    # Compute difference
    diff_translation = np.linalg.norm(T_apriltag[:3, 3] - MANUAL_POSE_MATRIX[:3, 3])
    print("\n" + "="*60)
    print("[COMPARISON]")
    print("="*60)
    print(f"Translation difference: {diff_translation*1000:.2f} mm")
    
    # --- Build scene ---
    geoms = []

    # Optional point cloud
    if PLY_PATH and PLY_PATH.exists():
        pcd = o3d.io.read_point_cloud(str(PLY_PATH))
        print(f"[PLY] Loaded point cloud with {len(pcd.points)} points")
        
        # Scale point cloud if needed (RealSense exports are typically already in meters)
        if POINT_CLOUD_SCALE != 1.0:
            pcd.scale(POINT_CLOUD_SCALE, center=(0, 0, 0))
            print(f"[PLY] Scaled point cloud by {POINT_CLOUD_SCALE}")
        
        # Optional voxel downsampling
        if VOXEL and VOXEL > 0: 
            pcd = pcd.voxel_down_sample(voxel_size=float(VOXEL))
            print(f"[PLY] Downsampled to {len(pcd.points)} points (voxel={VOXEL}m)")
        
        geoms.append(pcd)

    # Ground grid
    if GRID and GRID > 0:
        geoms.append(make_xy_grid(cell=float(GRID), n=20, z=0.0))

    # --- AprilTag visualization (RED) ---
    axes_apriltag = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(AXES))
    axes_apriltag.transform(T_apriltag)
    
    sphere_apriltag = create_labeled_sphere(t_pnp, [1.0, 0.2, 0.2], float(SPHERE))
    
    geoms += [axes_apriltag, sphere_apriltag]

    # --- Manual pose visualization (BLUE) ---
    axes_manual = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(AXES) * 1.2)
    axes_manual.transform(MANUAL_POSE_MATRIX)
    
    sphere_manual = create_labeled_sphere(
        MANUAL_POSE_MATRIX[:3, 3], 
        [0.2, 0.2, 1.0], 
        float(SPHERE) * 1.2
    )
    
    geoms += [axes_manual, sphere_manual]

    # --- Load and place CAD models ---
    if CAD_PLY and CAD_PLY.exists():
        print("\n" + "="*60)
        print("[CAD PLACEMENT]")
        print("="*60)
        
        # # CAD with AprilTag pose (semi-transparent RED tint)
        # cad_apriltag = prepare_cad_model(CAD_PLY)
        # cad_apriltag.transform(T_apriltag)
        # if isinstance(cad_apriltag, o3d.geometry.TriangleMesh):
        #     cad_apriltag.paint_uniform_color([1.0, 0.3, 0.3])  # Red tint
        # geoms.append(cad_apriltag)
        # print("[CAD] AprilTag placement: RED tint applied")
        
        # CAD with manual pose (semi-transparent BLUE tint)
        cad_manual = prepare_cad_model(CAD_PLY)
        cad_manual.transform(MANUAL_POSE_MATRIX)
        if isinstance(cad_manual, o3d.geometry.TriangleMesh):
            cad_manual.paint_uniform_color([0.3, 0.3, 1.0])  # Blue tint
        geoms.append(cad_manual)
        print("[CAD] Manual pose placement: BLUE tint applied")
    else:
        print("\n[WARNING] CAD_PLY not found; skipping CAD visualization")

    # Add a line connecting the two origins
    line_pts = np.array([t_pnp, MANUAL_POSE_MATRIX[:3, 3]])
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(line_pts)
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]])  # Yellow line
    geoms.append(line)

    print("\n[VISUALIZATION]")
    print("Legend:")
    print("  • RED sphere/axes/CAD = AprilTag pose")
    print("  • BLUE sphere/axes/CAD = Manual pose")
    print("  • YELLOW line = Connection between origins")
    print("\nLaunching Open3D viewer...")
    
    o3d.visualization.draw_geometries(
        geoms,
        window_name="AprilTag vs Manual Pose Comparison",
        width=1920,
        height=1080
    )

if __name__ == '__main__':
    main()