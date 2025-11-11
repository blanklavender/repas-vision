#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
import open3d as o3d
from pupil_apriltags import Detector

# =========================== CONFIG ===========================
# Inputs
COLOR_PATH = Path(r"./new_test_captures/capture_20251031_155407/color_20251031_155407.png")
PLY_PATH   = Path(r"./new_test_captures/capture_20251031_155407/point_cloud_20251031_155407.ply")
DEPTH_PATH = Path(r"./new_test_captures/capture_20251031_155407/aligned_depth_raw_20251031_155407.png")
INTRINSICS_JSON = Path(r"./calibration_parameters/factory_color_intrinsics_2025-09-08T143506.json")

# Outputs
EXPORT_PLY_PATH      = Path("./new_test_exports/point_cloud_20251031_155407/cropped_camframe.ply")
EXPORT_PLY_YUP_PATH  = Path("./new_test_exports/point_cloud_20251031_155407/cropped_yup.ply")
EXPORT_META_JSON     = Path("./new_test_exports/point_cloud_20251031_155407/cropped_meta.json")
WRITE_Y_UP_VERSION   = False
WRITE_ASCII          = False
COMPRESS_PLY         = False

# (create the folder if needed)
EXPORT_PLY_PATH.parent.mkdir(parents=True, exist_ok=True)

# AprilTag settings
FAMILY     = "tag36h11"
TAG_SIZE_M = 0.0293
TAG_IDS    = [9, 16]
ANCHOR_ID  = 9

# Depth estimation settings
CENTER_WIN = 5
DEPTH_SCALE = 0.001

# ===== Offsets in TAG LOCAL FRAME =====
# Tag local frame: +X right, +Y down, +Z forward (away from tag surface)
d1_front = np.array([0.08, 0.20, 0.00], dtype=float)
d1_back  = np.array([0.08, 0.20, 0.64], dtype=float)
d2_front = np.array([+0.08, +0.20, 0.00], dtype=float)
d2_back  = np.array([+0.08, +0.20, 0.64], dtype=float)

# Optional: manual translation in Tag 9's local frame (meters)
TAG9_LOCAL_TRANSLATION = np.array([0.0, 0.0, 0.0], dtype=float)

# Optional visualization tweaks
AXES_SIZE = 0.05
VOXEL     = 0.0

# ===================== Intrinsics helpers =====================
def load_color_intrinsics(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    data = json.loads(json_path.read_text())
    intr = data["color_intrinsics"] if (isinstance(data, dict) and "color_intrinsics" in data) else data
    fx = float(intr["fx"]); fy = float(intr["fy"])
    cx = float(intr["cx"]); cy = float(intr["cy"])
    iw = int(intr.get("width", 0)); ih = int(intr.get("height", 0))
    return fx, fy, cx, cy, iw, ih

def scale_intrinsics(fx, fy, cx, cy, src_w, src_h, dst_w, dst_h):
    if src_w <= 0 or src_h <= 0 or (src_w == dst_w and src_h == dst_h):
        return fx, fy, cx, cy
    sx = float(dst_w)/float(src_w); sy = float(dst_h)/float(src_h)
    return fx*sx, fy*sy, cx*sx, cy*sy

def build_K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0,  fy, cy],
                     [0,   0,  1]], dtype=np.float64)

# ===================== Depth helpers =====================
def median_depth(depth_img: np.ndarray, u: int, v: int, win: int = 5) -> float:
    """Extract median depth in a window around (u, v)."""
    h, w = depth_img.shape[:2]
    half = win // 2
    u0 = max(0, u - half)
    u1 = min(w, u + half + 1)
    v0 = max(0, v - half)
    v1 = min(h, v + half + 1)
    patch = depth_img[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))

# ===================== Detection / PnP ========================
@dataclass
class DetectedTag:
    id: int
    corners_px: np.ndarray
    area: float

def detect_all_tags(img_bgr: np.ndarray, family: str) -> list[DetectedTag]:
    det = Detector(families=family, nthreads=2, quad_decimate=1.0,
                   quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ts = det.detect(gray, estimate_tag_pose=False)
    out = []
    for t in ts:
        tid = getattr(t, 'tag_id', getattr(t, 'id', -999))
        c = t.corners.astype(np.float64).reshape(4,2)
        area = float(cv2.contourArea(c.astype(np.float32)))
        out.append(DetectedTag(id=int(tid), corners_px=c, area=area))
    return out

def compute_reproj_error(obj_pts: np.ndarray, img_pts: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - img_pts, axis=1)))

def solve_pnp_best_order(img_corners_px: np.ndarray, K: np.ndarray, dist: np.ndarray, tag_size_m: float):
    half = float(tag_size_m) / 2.0
    TL = np.array([-half, -half, 0.0]); TR = np.array([+half, -half, 0.0])
    BR = np.array([+half, +half, 0.0]);  BL = np.array([-half, +half, 0.0])
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
    best = None; best_score = np.inf; best_err = None; best_label = None
    for label, obj_list in candidates:
        obj_pts = np.array(obj_list, dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_corners_px, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok: continue
        err = compute_reproj_error(obj_pts, img_corners_px, rvec, tvec, K, dist)
        z = float(tvec[2]); score = err + (1000.0 if z <= 0 else 0.0)
        if score < best_score:
            best_score = score; best = (obj_pts, rvec, tvec); best_err = err; best_label = ",".join(label)
    if best is None: raise RuntimeError("solvePnP failed for all corner orderings.")
    obj_pts, rvec, tvec = best
    R, _ = cv2.Rodrigues(rvec); t = tvec.reshape(3)
    return R, t, best_err, best_label

# ===== NEW: Invert rotation in X and Y axes =====
def invert_rotation_xy(R: np.ndarray) -> np.ndarray:
    """
    Invert/flip the rotation matrix in X and Y axes.
    This applies a 180-degree rotation around the Z axis, effectively flipping X and Y.
    """
    # Create flip matrix: negate X and Y axes, keep Z
    flip_xy = np.array([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]
    ], dtype=np.float64)
    
    # Apply the flip transformation
    R_flipped = flip_xy @ R @ flip_xy.T
    
    return R_flipped

# ===== NEW: Transform point from tag frame to camera frame =====
def transform_point_tag_to_camera(point_tag_frame: np.ndarray, R_tag: np.ndarray, t_tag: np.ndarray) -> np.ndarray:
    """
    Transform a point from tag's local frame to camera frame.
    
    Args:
        point_tag_frame: 3D point in tag's coordinate system
        R_tag: Rotation matrix from tag frame to camera frame
        t_tag: Translation vector (tag center in camera frame)
    
    Returns:
        point_cam_frame: 3D point in camera coordinate system
    """
    return (R_tag @ point_tag_frame.reshape(3, 1)).reshape(3) + t_tag

def transform_point_tag_local_to_camera(local_pt: np.ndarray, R_tag: np.ndarray, t_tag: np.ndarray) -> np.ndarray:
    """Generic: tag-local -> camera."""
    return (R_tag @ local_pt.reshape(3, 1)).reshape(3) + t_tag

def transform_point_tag9(local_pt: np.ndarray, R9: np.ndarray, t9: np.ndarray) -> np.ndarray:
    """
    Tag 9: apply optional local translation along tag-9 axes, then map to camera.
    This ensures the translation happens in the tag-9 coordinate system (not camera).
    """
    local_pt = local_pt + TAG9_LOCAL_TRANSLATION
    return transform_point_tag_local_to_camera(local_pt, R9, t9)

# =============== Tag corner → camera coordinates ===============
def tag_corners_world_from_pose(R: np.ndarray, t: np.ndarray, tag_size_m: float) -> dict[str,np.ndarray]:
    """Return dict with 3D camera-frame positions of TL,TR,BR,BL,CTR."""
    half = float(tag_size_m) / 2.0
    TL = np.array([-half, -half, 0.0]); TR = np.array([+half, -half, 0.0])
    BR = np.array([+half, +half,  0.0]); BL = np.array([-half, +half,  0.0])
    CTR = np.array([0.0, 0.0, 0.0])
    def X(local): return (R @ local.reshape(3,1)).reshape(3) + t
    return {
        "TL": X(TL), "TR": X(TR), "BR": X(BR), "BL": X(BL), "CTR": X(CTR)
    }

def get_tag_corner_local(corner_name: str, tag_size_m: float) -> np.ndarray:
    """Get corner position in tag's local frame."""
    half = float(tag_size_m) / 2.0
    corners_local = {
        "TL": np.array([-half, -half, 0.0]),
        "TR": np.array([+half, -half, 0.0]),
        "BR": np.array([+half, +half, 0.0]),
        "BL": np.array([-half, +half, 0.0]),
        "CTR": np.array([0.0, 0.0, 0.0])
    }
    return corners_local[corner_name]

# =================== Open3D util / viz ========================
def axis_frame(T: np.ndarray | None = None, size: float = AXES_SIZE):
    f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(size))
    if T is not None: f.transform(T)
    return f

def create_sphere(center: np.ndarray, radius: float = 0.01, color=(1.0, 0.0, 0.0)):
    """Create a small sphere at a point for visualization."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere

def create_line(p1: np.ndarray, p2: np.ndarray, color=(0.0, 1.0, 0.0)):
    """Create a line between two points."""
    points = [p1, p2]
    lines = [[0, 1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color])
    return line_set

def create_aabb_wireframe(min_b: np.ndarray, max_b: np.ndarray, color=(1.0, 0.6, 0.0)):
    """Create wireframe for axis-aligned bounding box."""
    # 8 corners of the box
    corners = np.array([
        [min_b[0], min_b[1], min_b[2]],  # 0
        [max_b[0], min_b[1], min_b[2]],  # 1
        [max_b[0], max_b[1], min_b[2]],  # 2
        [min_b[0], max_b[1], min_b[2]],  # 3
        [min_b[0], min_b[1], max_b[2]],  # 4
        [max_b[0], min_b[1], max_b[2]],  # 5
        [max_b[0], max_b[1], max_b[2]],  # 6
        [min_b[0], max_b[1], max_b[2]],  # 7
    ])
    
    # 12 edges
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # front face
        [4, 5], [5, 6], [6, 7], [7, 4],  # back face
        [0, 4], [1, 5], [2, 6], [3, 7],  # connecting edges
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return line_set

def create_obb_from_points(points: np.ndarray, color=(0.0, 1.0, 1.0)):
    """Create oriented bounding box from a set of points."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    obb = pcd.get_oriented_bounding_box()
    obb.color = color
    return obb

# ============================== MAIN ==============================
def main():
    # Load inputs
    if not COLOR_PATH.exists(): raise FileNotFoundError(COLOR_PATH)
    if not PLY_PATH.exists():   raise FileNotFoundError(PLY_PATH)
    if not INTRINSICS_JSON.exists(): raise FileNotFoundError(INTRINSICS_JSON)

    img = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError("Failed to read COLOR image")
    h, w = img.shape[:2]

    # Load depth image
    Zc_img = None
    if DEPTH_PATH.exists():
        depth_raw = cv2.imread(str(DEPTH_PATH), cv2.IMREAD_ANYDEPTH)
        if depth_raw is not None:
            Zc_img = depth_raw.astype(np.float32) * DEPTH_SCALE
            print(f"[DEPTH] Loaded {DEPTH_PATH.name}, shape={Zc_img.shape}, scale={DEPTH_SCALE}")
        else:
            print(f"[WARN] Could not read depth image at {DEPTH_PATH}")
    else:
        print(f"[WARN] Depth image not found at {DEPTH_PATH}")

    fx0, fy0, cx0, cy0, iw, ih = load_color_intrinsics(INTRINSICS_JSON)
    fx, fy, cx, cy = (scale_intrinsics(fx0, fy0, cx0, cy0, iw, ih, w, h)
                      if (iw>0 and ih>0 and (iw!=w or ih!=h)) else (fx0, fy0, cx0, cy0))
    K = build_K(fx, fy, cx, cy)
    dist = np.zeros((5,1), dtype=np.float64)
    print(f"[INTR] {w}x{h} | fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    # Detect all tags
    detections = detect_all_tags(img, FAMILY)
    if not detections:
        raise RuntimeError("No AprilTags detected.")

    # Keep only tag 9 and tag 16
    chosen = [d for d in detections if d.id in TAG_IDS]
    if len(chosen) < 2:
        print(f"[WARN] Requested tags {TAG_IDS}, but visible: {[d.id for d in detections]}")

    # PnP for all chosen tags + depth-based tvec correction
    poses = {}  # id -> (R, t_corrected)
    
    print("\n" + "="*70)
    print("TAG POSE ESTIMATION")
    print("="*70)
    
    for det in chosen:
        img_pts = det.corners_px.copy()
        if img_pts.max() <= 1.5:  # normalized guard
            img_pts[:,0] *= w; img_pts[:,1] *= h
            print(f"[FIX] Tag {det.id}: normalized corners → pixels")
        
        # Get rvec and tvec from PnP
        R_tag, t_pnp, err_px, order = solve_pnp_best_order(img_pts, K, dist, TAG_SIZE_M)
        
        # Compute depth-based tvec
        t_final = t_pnp.copy()  # Default to PnP tvec
        P_depth = None
        
        if Zc_img is not None and float(t_pnp[2]) > 1e-6:
            # Project PnP tvec to pixel coordinates
            uv = (K @ t_pnp.reshape(3,1)).reshape(3)
            u, v = int(round(uv[0]/uv[2])), int(round(uv[1]/uv[2]))
            
            if 0 <= u < w and 0 <= v < h:
                Zc = median_depth(Zc_img, u, v, CENTER_WIN)
                if Zc > 0:
                    # Compute 3D position from depth
                    X = (u - cx) / fx * Zc
                    Y = (v - cy) / fy * Zc
                    P_depth = np.array([X, Y, Zc], dtype=float)
                    t_final = P_depth  # Use depth-based tvec
        
        print(f"\n[Tag {det.id}]")
        print(f"  Reproj error: {err_px:.2f}px")
        print(f"  Corner order: {order}")
        print(f"  R_tag (original):\n{R_tag}")
        
        # # ===== NEW: Apply XY inversion for Tag 9 =====
        # if det.id == 9:
        #     R_original = R_tag.copy()
        #     R_tag = invert_rotation_xy(R_tag)
        #     print(f"  R_tag (after XY inversion):\n{R_tag}")
        #     print(f"  [INFO] Applied XY axis inversion to Tag 9's rotation")
        
        poses[det.id] = (R_tag, t_final)
        
        print(f"  t_pnp:   {t_pnp}")
        if P_depth is not None:
            print(f"  t_depth: {t_final}  (✓ depth-corrected)")
        else:
            print(f"  t_final: {t_final}  (using PnP)")

    # ======== NEW APPROACH: Build bounding box in tag frame, then transform ========
    print("\n" + "="*70)
    print("BOUNDING BOX CONSTRUCTION (NEW APPROACH)")
    print("="*70)
    
    if 9 not in poses:
        raise RuntimeError("Tag id=9 not visible; required for bounding box.")
    if 16 not in poses:
        raise RuntimeError("Tag id=16 not visible; required for bounding box.")

    R9, t9 = poses[9]
    R16, t16 = poses[16]

    # Get corner positions in tag local frames
    TL_local_tag9 = get_tag_corner_local("BR", TAG_SIZE_M) # TEMP FIX FROM TL TO BR
    TR_local_tag16 = get_tag_corner_local("TR", TAG_SIZE_M)

    print(f"\nTag 9 (TL corner in local frame): {TL_local_tag9}")
    print(f"Tag 16 (TR corner in local frame): {TR_local_tag16}")

    # Show offsets in tag local frame
    print(f"\nOffsets (in tag local frame):")
    print(f"  d1_front (from Tag 9 TL): {d1_front}")
    print(f"  d1_back  (from Tag 9 TL): {d1_back}")
    print(f"  d2_front (from Tag 16 TR): {d2_front}")
    print(f"  d2_back  (from Tag 16 TR): {d2_back}")

    # STEP 1: Add offsets in tag local frame
    p1_tag9_frame = TL_local_tag9 + d1_front
    p2_tag9_frame = TL_local_tag9 + d1_back
    p3_tag16_frame = TR_local_tag16 + d2_front
    p4_tag16_frame = TR_local_tag16 + d2_back

    print(f"\nBounding Box Corners (in respective tag frames):")
    print(f"  p1_tag9_frame  = TL_local + d1_front = {p1_tag9_frame}")
    print(f"  p2_tag9_frame  = TL_local + d1_back  = {p2_tag9_frame}")
    print(f"  p3_tag16_frame = TR_local + d2_front = {p3_tag16_frame}")
    print(f"  p4_tag16_frame = TR_local + d2_back  = {p4_tag16_frame}")

    # STEP 2: Transform to camera frame (NO XY inversion on Tag 9)
    p1 = transform_point_tag9(p1_tag9_frame, R9, t9)
    p2 = transform_point_tag9(p2_tag9_frame, R9, t9)
    p3 = transform_point_tag_local_to_camera(p3_tag16_frame, R16, t16)
    p4 = transform_point_tag_local_to_camera(p4_tag16_frame, R16, t16)

    print(f"\nBounding Box Corners (transformed to camera frame):")
    print(f"  p1 = R9 @ p1_tag9_frame + t9   = {p1}")
    print(f"  p2 = R9 @ p2_tag9_frame + t9   = {p2}")
    print(f"  p3 = R16 @ p3_tag16_frame + t16 = {p3}")
    print(f"  p4 = R16 @ p4_tag16_frame + t16 = {p4}")

    # Also compute anchor points for visualization
    TL9_cam  = transform_point_tag9(TL_local_tag9, R9, t9)
    TR16_cam = transform_point_tag_local_to_camera(TR_local_tag16, R16, t16)
    
    print(f"\nAnchor Points (camera frame):")
    print(f"  TL9_cam  (Tag 9, TL corner): {TL9_cam}")
    print(f"  TR16_cam (Tag 16, TR corner): {TR16_cam}")

    # For masking, create axis-aligned bounding box
    all_pts = np.vstack([p1, p2, p3, p4])
    min_b = np.min(all_pts, axis=0)
    max_b = np.max(all_pts, axis=0)

    print(f"\nAxis-Aligned Bounding Box (AABB):")
    print(f"  min_b: {min_b}")
    print(f"  max_b: {max_b}")
    print(f"  Dimensions: X={max_b[0]-min_b[0]:.3f}m, Y={max_b[1]-min_b[1]:.3f}m, Z={max_b[2]-min_b[2]:.3f}m")

    # ======== Load point cloud and crop ========
    print("\n" + "="*70)
    print("POINT CLOUD PROCESSING")
    print("="*70)
    
    pcd = o3d.io.read_point_cloud(str(PLY_PATH))
    if VOXEL and VOXEL > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(VOXEL))

    P = np.asarray(pcd.points)
    if P.size == 0:
        raise RuntimeError("Point cloud has no points.")

    print(f"\nOriginal point cloud: {P.shape[0]} points")

    # Mask using axis-aligned bounding box
    mask = (
        (P[:,0] >= min_b[0]) & (P[:,0] <= max_b[0]) &
        (P[:,1] >= min_b[1]) & (P[:,1] <= max_b[1]) &
        (P[:,2] >= min_b[2]) & (P[:,2] <= max_b[2])
    )
    kept = np.count_nonzero(mask); total = P.shape[0]
    
    # Show statistics for each axis
    print(f"\nFiltering statistics:")
    mask_x = (P[:,0] >= min_b[0]) & (P[:,0] <= max_b[0])
    mask_y = (P[:,1] >= min_b[1]) & (P[:,1] <= max_b[1])
    mask_z = (P[:,2] >= min_b[2]) & (P[:,2] <= max_b[2])
    print(f"  X-axis: {np.count_nonzero(mask_x)}/{total} points in range [{min_b[0]:.3f}, {max_b[0]:.3f}]")
    print(f"  Y-axis: {np.count_nonzero(mask_y)}/{total} points in range [{min_b[1]:.3f}, {max_b[1]:.3f}]")
    print(f"  Z-axis: {np.count_nonzero(mask_z)}/{total} points in range [{min_b[2]:.3f}, {max_b[2]:.3f}]")
    print(f"  Combined: {kept}/{total} points ({100.0*kept/max(1,total):.1f}%)")

    pcd_fg = pcd.select_by_index(np.where(mask)[0])

    # ======== Create visualization geometries ========
    print("\n" + "="*70)
    print("CREATING VISUALIZATION")
    print("="*70)
    
    geoms = []

    # # 1. Tag coordinate frames (using modified R9 with XY inversion)
    # for tid in TAG_IDS:
    #     if tid in poses:
    #         R, t = poses[tid]
    #         T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    #         geoms.append(axis_frame(T, size=AXES_SIZE))

    # 2. Anchor points (TL9 and TR16)
    geoms.append(create_sphere(TL9_cam, radius=0.015, color=(1.0, 0.0, 1.0)))  # Magenta
    geoms.append(create_sphere(TR16_cam, radius=0.015, color=(1.0, 1.0, 0.0)))  # Yellow

    # 3. The 4 bounding box corners as colored spheres
    geoms.append(create_sphere(p1, radius=0.02, color=(1.0, 0.0, 0.0)))  # Red
    geoms.append(create_sphere(p2, radius=0.02, color=(0.0, 1.0, 0.0)))  # Green
    geoms.append(create_sphere(p3, radius=0.02, color=(0.0, 0.0, 1.0)))  # Blue
    geoms.append(create_sphere(p4, radius=0.02, color=(1.0, 1.0, 1.0)))  # White

    # 4. Lines connecting corners
    geoms.append(create_line(TL9_cam, p1, color=(1.0, 0.0, 1.0)))  # Anchor to p1
    geoms.append(create_line(TL9_cam, p2, color=(1.0, 0.0, 1.0)))  # Anchor to p2
    geoms.append(create_line(TR16_cam, p3, color=(1.0, 1.0, 0.0)))  # Anchor to p3
    geoms.append(create_line(TR16_cam, p4, color=(1.0, 1.0, 0.0)))  # Anchor to p4
    geoms.append(create_line(p1, p2, color=(0.5, 0.5, 0.5)))  # p1-p2
    geoms.append(create_line(p3, p4, color=(0.5, 0.5, 0.5)))  # p3-p4
    geoms.append(create_line(p1, p3, color=(0.5, 0.5, 0.5)))  # p1-p3
    geoms.append(create_line(p2, p4, color=(0.5, 0.5, 0.5)))  # p2-p4

    # 5. AABB wireframe (orange)
    aabb_wireframe = create_aabb_wireframe(min_b, max_b, color=(1.0, 0.6, 0.0))
    geoms.append(aabb_wireframe)

    # 6. OBB from the 4 corners (cyan)
    obb = create_obb_from_points(all_pts, color=(0.0, 1.0, 1.0))
    geoms.append(obb)

    # 7. Cropped point cloud
    geoms.append(pcd_fg)

    # 8. Original point cloud (downsampled and in gray for context)
    pcd_context = o3d.geometry.PointCloud(pcd)
    pcd_context.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(pcd_context)

    print("\nVisualization Legend:")
    print("  Magenta sphere  = TL9_cam (Tag 9, TL corner anchor)")
    print("  Yellow sphere   = TR16_cam (Tag 16, TR corner anchor)")
    print("  Red sphere      = p1 (front corner from Tag 9)")
    print("  Green sphere    = p2 (back corner from Tag 9)")
    print("  Blue sphere     = p3 (front corner from Tag 16)")
    print("  White sphere    = p4 (back corner from Tag 16)")
    print("  Orange wireframe = AABB (used for cropping)")
    print("  Cyan box         = OBB (oriented bounding box)")
    print("  Colored points   = Cropped point cloud")
    print("  Gray points      = Original point cloud (context)")
    print("\n  NOTE: Tag 9's coordinate frame shows XY-inverted orientation")
    print("  NOTE: Offsets applied in tag frame FIRST, then transformed to camera")

    # ======== Export ========
    ok_cam = o3d.io.write_point_cloud(
        str(EXPORT_PLY_PATH), pcd_fg, write_ascii=WRITE_ASCII, compressed=COMPRESS_PLY
    )
    if not ok_cam:
        raise RuntimeError(f"Failed to write {EXPORT_PLY_PATH}")

    if WRITE_Y_UP_VERSION:
        pcd_yup = o3d.geometry.PointCloud(pcd_fg)
        P_cam = np.asarray(pcd_yup.points)
        if P_cam.size:
            P_yup = P_cam.copy()
            P_yup[:, 1] *= -1.0
            pcd_yup.points = o3d.utility.Vector3dVector(P_yup)
            ok_yup = o3d.io.write_point_cloud(
                str(EXPORT_PLY_YUP_PATH), pcd_yup, write_ascii=WRITE_ASCII, compressed=COMPRESS_PLY
            )
            if not ok_yup:
                raise RuntimeError(f"Failed to write {EXPORT_PLY_YUP_PATH}")

    meta = {
        "source_color_image": str(COLOR_PATH),
        "source_depth_image": str(DEPTH_PATH),
        "source_ply": str(PLY_PATH),
        "export_ply_camera_frame": str(EXPORT_PLY_PATH),
        "export_ply_y_up": (str(EXPORT_PLY_YUP_PATH) if WRITE_Y_UP_VERSION else None),
        "frame_convention_camera": {
            "x": "right",
            "y": "down",
            "z": "forward (from camera to scene)",
            "units": "meters",
            "note": "OpenCV camera frame with depth-corrected tvec. Tag 9 has XY-inverted rotation. Offsets applied in tag frame first, then transformed to camera."
        },
        "offset_convention": {
            "note": "Offsets specified in tag local frame, added to corner positions, then entire point transformed to camera frame",
            "tag_local_frame": {
                "x": "right (along tag surface)",
                "y": "down (along tag surface)",
                "z": "forward (away from tag surface)"
            }
        },
        "transformation_order": {
            "step_1": "Get corner position in tag local frame (e.g., TL)",
            "step_2": "Add offset in tag local frame: point_tag = corner_local + offset",
            "step_3": "Transform to camera: point_cam = R_tag @ point_tag + t_tag"
        },
        "rotation_modification": {
            "tag_9": "XY axes inverted (180° rotation around Z axis)",
            "tag_16": "Original rotation from PnP"
        },
        "anchor_points_cam": {
            "TL9_cam": TL9_cam.tolist(),
            "TR16_cam": TR16_cam.tolist()
        },
        "bounding_box_corners_cam": {
            "p1": p1.tolist(),
            "p2": p2.tolist(),
            "p3": p3.tolist(),
            "p4": p4.tolist()
        },
        "intrinsics": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy), "width": int(w), "height": int(h)},
        "crop_aabb_min_xyz_m": [float(min_b[0]), float(min_b[1]), float(min_b[2])],
        "crop_aabb_max_xyz_m": [float(max_b[0]), float(max_b[1]), float(max_b[2])],
        "tag_size_m": float(TAG_SIZE_M),
        "used_tags": TAG_IDS,
        "depth_scale": float(DEPTH_SCALE),
    }
    EXPORT_META_JSON.write_text(json.dumps(meta, indent=2))
    
    print(f"\n[SAVE] Camera-frame PLY -> {EXPORT_PLY_PATH}")
    if WRITE_Y_UP_VERSION:
        print(f"[SAVE] Y-up PLY        -> {EXPORT_PLY_YUP_PATH}")
    print(f"[SAVE] Metadata JSON     -> {EXPORT_META_JSON}")

    print("\nOpening 3D viewer...")
    o3d.visualization.draw_geometries(geoms, window_name="Bounding Box Visualization (Tag-Frame-First Approach)",
                                     width=1920, height=1080)

if __name__ == "__main__":
    main()