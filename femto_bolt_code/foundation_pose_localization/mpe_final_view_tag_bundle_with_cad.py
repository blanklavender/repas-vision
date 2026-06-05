#!/usr/bin/env python3
from __future__ import annotations
import math
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
import open3d as o3d
from pupil_apriltags import Detector

# =========================================================
#                     CONFIG
# =========================================================
# Inputs (image, depth map aligned to color, optional scene point cloud)
COLOR_PATH = Path(r"./new_test_captures/capture_20251031_155222/color_20251031_155222.png")
DEPTH_NPY  = Path(r"./new_test_captures/capture_20251031_155222/aligned_depth_m_20251031_155222.npy")
PLY_PATH   = Path(r"./new_test_exports/point_cloud_20251031_155222/cropped_camframe.ply")

# Intrinsics JSON
INTRINSICS_JSON = Path(r"./calibration_parameters/factory_color_intrinsics_2025-09-08T143506.json")

# AprilTags
FAMILY       = "tag36h11"
TAG_IDS      = [9, 16]    # tags to use
TAG_SIZE_M   = 0.0293     # outer black square edge length (meters)
CENTER_WIN   = 5          # depth median window

# *** NEW: Define 3D positions of each tag in world coordinates ***
# This is the critical addition for multi-point embedding
# You need to measure/know where each tag is positioned relative to each other
TAG_3D_POSITIONS = {
    9: {
        'center': np.array([0.0, 0.0, 0.0], dtype=np.float64),  # Tag 9 at origin
        'rotation': np.eye(3, dtype=np.float64)  # No rotation
    },
    16: {
        'center': np.array([0.15, 0.0, 0.0], dtype=np.float64),  # Tag 16 is 15cm to the right
        'rotation': np.eye(3, dtype=np.float64)  # No rotation
    }
}

CAD_ANCHOR_ID= 16          # which tag to use for CAD placement

# Viz sizes
AXES   = 0.05
SPHERE = 0.003
GRID   = 0.10
VOXEL  = 0.0

# CAD placement options
CAD_PLY                 = Path(r"../../cad_model/StructureTotal-v2.PLY")
CAD_UNITS_TO_METERS     = 0.001
CAD_PRE_ROT_DEG_ZYX     = (0.0, -1, 0.0)
CAD_CENTER_ON_ORIGIN    = False
CAD_ORIGIN_OFFSET_LOCAL = (0.0, 0.0, 0.0)

# =========================================================
#               INTRINSICS / UTIL HELPERS
# =========================================================
def load_color_intrinsics(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    data = json.loads(json_path.read_text())
    intr = data["color_intrinsics"] if (isinstance(data, dict) and "color_intrinsics" in data) else data
    for k in ("fx", "fy", "cx", "cy"):
        if k not in intr:
            raise KeyError(f"Missing '{k}' in intrinsics JSON")
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

def median_depth(Z: np.ndarray, u: int, v: int, win: int) -> float:
    h, w = Z.shape
    r = max(1, win//2)
    u0,u1 = max(0,u-r), min(w,u+r+1)
    v0,v1 = max(0,v-r), min(h,v+r+1)
    patch = Z[v0:v1, u0:u1]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size else 0.0

# =========================================================
#               OPEN3D GEOM HELPERS
# =========================================================
def geom_centroid(g):
    """Get centroid of mesh or point cloud"""
    if isinstance(g, o3d.geometry.TriangleMesh):
        return g.get_center()
    else:
        return np.asarray(g.points).mean(axis=0)

def colored_axes_lines(size: float, colors_xyz=((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0))):
    radius = max(size * 0.02, 1e-4)
    cx = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size)
    cy = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size)
    cz = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size)
    cx.paint_uniform_color(colors_xyz[0]); cy.paint_uniform_color(colors_xyz[1]); cz.paint_uniform_color(colors_xyz[2])
    cx.rotate(cx.get_rotation_matrix_from_xyz((0.0, -np.pi/2.0, 0.0)), center=(0,0,0)); cx.translate((size/2.0, 0.0, 0.0))
    cy.rotate(cy.get_rotation_matrix_from_xyz((np.pi/2.0, 0.0, 0.0)), center=(0,0,0));  cy.translate((0.0, size/2.0, 0.0))
    cz.translate((0.0, 0.0, size/2.0))
    m = cx + cy + cz
    m.compute_vertex_normals()
    return m

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

def load_cad_geometry(path: Path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh and (len(np.asarray(mesh.vertices)) > 0):
        mesh.compute_vertex_normals()
        return mesh
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd and (len(np.asarray(pcd.points)) > 0):
        return pcd
    raise RuntimeError(f"Failed to load CAD from {path}")

def euler_zyx_to_R(z_deg: float, y_deg: float, x_deg: float) -> np.ndarray:
    z, y, x = [math.radians(a) for a in (z_deg, y_deg, x_deg)]
    cz, sz = math.cos(z), math.sin(z)
    cy, sy = math.cos(y), math.sin(y)
    cx, sx = math.cos(x), math.sin(x)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    return Rz @ Ry @ Rx

# =========================================================
#                DETECTION + MULTI-POINT SQPNP
# =========================================================
@dataclass
class DetectedTag:
    id: int
    corners_px: np.ndarray  # (4,2)
    area: float

def detect_all_tags(img_bgr: np.ndarray, family: str) -> list[DetectedTag]:
    det = Detector(families=family, nthreads=2, quad_decimate=1.0, quad_sigma=0.0,
                   refine_edges=1, decode_sharpening=0.25)
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

def solve_multi_tag_sqpnp(
    detections: list[DetectedTag],
    tag_3d_positions: dict,
    tag_size_m: float,
    K: np.ndarray,
    dist: np.ndarray,
    img_width: int,
    img_height: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Multi-point embedding: Solve PnP using all corners + centers from all tags with SQPnP.
    
    Args:
        detections: List of detected tags
        tag_3d_positions: Dict mapping tag_id -> {'center': np.array, 'rotation': np.array}
        tag_size_m: Tag size in meters
        K: Camera intrinsic matrix
        dist: Distortion coefficients
        img_width, img_height: Image dimensions for denormalization
    
    Returns:
        R: 3x3 rotation matrix (camera pose)
        t: 3D translation vector (camera pose)
        reproj_error: Mean reprojection error in pixels
    """
    half = float(tag_size_m) / 2.0
    
    # Define tag corners in tag-local coordinates
    # Standard convention when looking at tag face-on:
    tag_corners_local = np.array([
        [-half, -half, 0.0],  # Top-left
        [+half, -half, 0.0],  # Top-right
        [+half, +half, 0.0],  # Bottom-right
        [-half, +half, 0.0],  # Bottom-left
    ], dtype=np.float64)
    
    tag_center_local = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    all_obj_pts = []
    all_img_pts = []
    
    for det in detections:
        if det.id not in tag_3d_positions:
            print(f"[WARN] Tag {det.id} not in TAG_3D_POSITIONS, skipping")
            continue
            
        # Get the world position and orientation of this tag
        tag_world_center = tag_3d_positions[det.id]['center']
        tag_world_R = tag_3d_positions[det.id]['rotation']
        
        # Get the 4 corner pixels for this tag
        img_corners = det.corners_px.copy()
        
        # Denormalize if needed
        if img_corners.max() <= 1.5:
            img_corners[:, 0] *= img_width
            img_corners[:, 1] *= img_height
            print(f"[FIX] Tag {det.id}: corners looked normalized; scaled to pixels.")
        
        # pupil_apriltags returns corners in order: [BL, BR, TR, TL] (in image coords)
        # where "bottom" means higher y value (since y increases downward)
        # Reorder to match our [TL, TR, BR, BL] convention
        reordered_img_corners = np.array([
            img_corners[3],  # TL
            img_corners[2],  # TR
            img_corners[1],  # BR
            img_corners[0],  # BL
        ], dtype=np.float64)
        
        # Transform tag-local corner positions to world coordinates
        for i in range(4):
            # Corner in tag-local coords
            corner_local = tag_corners_local[i]
            # Transform to world coords: P_world = tag_world_R @ P_local + tag_world_center
            corner_world = tag_world_R @ corner_local + tag_world_center
            
            all_obj_pts.append(corner_world)
            all_img_pts.append(reordered_img_corners[i])
        
        # Add center point
        center_world = tag_world_R @ tag_center_local + tag_world_center
        img_center = img_corners.mean(axis=0)  # Average of 4 corners
        
        all_obj_pts.append(center_world)
        all_img_pts.append(img_center)
        
        print(f"[SQPNP] Added tag {det.id}: 4 corners + 1 center (5 points)")
    
    if len(all_obj_pts) < 3:
        raise RuntimeError(f"SQPnP requires at least 3 points, got {len(all_obj_pts)}")
    
    # Convert to numpy arrays
    all_obj_pts = np.array(all_obj_pts, dtype=np.float64)
    all_img_pts = np.array(all_img_pts, dtype=np.float64)
    
    print(f"[SQPNP] Solving with {len(all_obj_pts)} total points from {len(detections)} tags")
    
    # Solve using SQPnP
    success, rvec, tvec = cv2.solvePnP(
        all_obj_pts,
        all_img_pts,
        K,
        dist,
        flags=cv2.SOLVEPNP_SQPNP
    )
    
    if not success:
        raise RuntimeError("SQPnP failed to find solution")
    
    # Convert to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    
    # Compute reprojection error
    reproj_error = compute_reproj_error(all_obj_pts, all_img_pts, rvec, tvec, K, dist)
    
    print(f"[SQPNP] Success! Reprojection error: {reproj_error:.3f} pixels")
    
    return R, t, reproj_error

# =========================================================
#                        MAIN
# =========================================================
def main():
    # --- Load inputs ---
    if not COLOR_PATH.exists(): raise FileNotFoundError(COLOR_PATH)
    if not INTRINSICS_JSON.exists(): raise FileNotFoundError(INTRINSICS_JSON)
    img = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError("Failed to read COLOR image")
    h, w = img.shape[:2]

    Zc_img = None
    if DEPTH_NPY and DEPTH_NPY.exists():
        Zc_img = np.load(str(DEPTH_NPY))
        if Zc_img.shape != (h, w):
            raise RuntimeError(f"Depth size mismatch: COLOR {w}x{h} vs DEPTH {Zc_img.shape[1]}x{Zc_img.shape[0]}")

    # --- Intrinsics ---
    fx0, fy0, cx0, cy0, iw, ih = load_color_intrinsics(INTRINSICS_JSON)
    fx, fy, cx, cy = (scale_intrinsics(fx0, fy0, cx0, cy0, iw, ih, w, h)
                      if (iw>0 and ih>0 and (iw!=w or ih!=h)) else (fx0, fy0, cx0, cy0))
    K = build_K(fx, fy, cx, cy)
    dist = np.zeros((5,1), dtype=np.float64)

    # --- Detect all tags ---
    detections = detect_all_tags(img, FAMILY)
    if not detections:
        raise RuntimeError("No AprilTags detected.")

    # Filter to the set we need
    chosen = [d for d in detections if d.id in TAG_IDS]
    if len(chosen) == 0:
        raise RuntimeError(f"No requested tags {TAG_IDS} found. Detected: {[d.id for d in detections]}")

    print(f"\n[INFO] Detected {len(chosen)} tags: {[d.id for d in chosen]}")

    # --- Solve using multi-point SQPnP ---
    R_cam, t_cam, reproj_err = solve_multi_tag_sqpnp(
        chosen,
        TAG_3D_POSITIONS,
        TAG_SIZE_M,
        K,
        dist,
        w,
        h
    )
    
    print(f"\n[RESULT] Camera Pose (world->camera transform):")
    print(f"  R_cam:\n{R_cam}")
    print(f"  t_cam: {t_cam}")
    print(f"  Reproj error: {reproj_err:.3f} pixels")

    # --- Compute depth-validated positions ---
    anchor_P_depth = None
    tag_P_depths = {}
    
    for det in chosen:
        if det.id not in TAG_3D_POSITIONS:
            continue
            
        # Transform tag world position to camera frame
        tag_world_pos = TAG_3D_POSITIONS[det.id]['center']
        tag_cam_pos = R_cam @ tag_world_pos + t_cam
        
        # Project to image to get pixel location
        uv_h = K @ tag_cam_pos
        u, v = int(round(uv_h[0]/uv_h[2])), int(round(uv_h[1]/uv_h[2]))
        
        # Get depth measurement
        P_depth = None
        if Zc_img is not None and 0 <= u < w and 0 <= v < h:
            Zc = median_depth(Zc_img, u, v, CENTER_WIN)
            if Zc > 0:
                X = (u - cx) / fx * Zc
                Y = (v - cy) / fy * Zc
                P_depth = np.array([X, Y, Zc], dtype=float)
                print(f"[DEPTH] Tag {det.id}: PnP={tag_cam_pos}, Depth={P_depth}")
        
        tag_P_depths[det.id] = P_depth
        
        if det.id == CAD_ANCHOR_ID:
            anchor_P_depth = P_depth

    # --- Build Open3D scene ---
    geoms = []

    # Optional scene point cloud
    if PLY_PATH and PLY_PATH.exists():
        pcd = o3d.io.read_point_cloud(str(PLY_PATH))
        if VOXEL and VOXEL > 0:
            pcd = pcd.voxel_down_sample(voxel_size=float(VOXEL))
        geoms.append(pcd)

    # Show tag positions with axes
    for det in chosen:
        if det.id not in TAG_3D_POSITIONS or det.id not in tag_P_depths:
            continue
        
        P_depth = tag_P_depths[det.id]
        if P_depth is not None:
            # Show axes at depth position with camera-measured rotation
            world_axes = colored_axes_lines(float(AXES)*0.8)
            T = np.eye(4)
            T[:3, :3] = R_cam.T  # Inverse rotation (world frame)
            T[:3, 3] = P_depth
            world_axes.transform(T)
            geoms.append(world_axes)
            
            # Label sphere
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(SPHERE)*2)
            sph.compute_vertex_normals()
            sph.paint_uniform_color([1.0, 0.0, 0.0] if det.id == 9 else [0.0, 0.0, 1.0])
            sph.translate(P_depth)
            geoms.append(sph)

    if GRID and GRID > 0:
        geoms.append(make_xy_grid(cell=float(GRID), n=20, z=0.0))

    # --- Load & place CAD ---
    if CAD_PLY and CAD_PLY.exists() and anchor_P_depth is not None:
        cad = load_cad_geometry(CAD_PLY)
        
        # Get original centroid
        centroid_original = geom_centroid(cad)
        print(f"\n[CAD] Original centroid (CAD units): {centroid_original}")
        
        # Scale to meters
        S = float(CAD_UNITS_TO_METERS)
        cad.scale(S, center=centroid_original)
        
        centroid_meters = geom_centroid(cad)
        cad_origin_local = centroid_meters + (-centroid_original * S)
        
        # Apply camera rotation (inverse since we want world frame orientation)
        cad.rotate(R_cam.T, center=cad_origin_local)
        print(f"[CAD] Applied R_cam.T rotation")
        
        origin_world = cad_origin_local
        
        # Translate to anchor position
        translation = anchor_P_depth - origin_world
        cad.translate(translation)
        print(f"[CAD] Translated to anchor P_depth: {anchor_P_depth}")
        
        # Apply pre-rotation
        if any(abs(a) > 1e-6 for a in CAD_PRE_ROT_DEG_ZYX):
            Rpre = euler_zyx_to_R(*CAD_PRE_ROT_DEG_ZYX)
            cad.rotate(Rpre, center=anchor_P_depth)
            print(f"[CAD] Applied pre-rotation ZYX {CAD_PRE_ROT_DEG_ZYX}")
        
        geoms.append(cad)
        
        final_centroid = geom_centroid(cad)
        print(f"[DEBUG] Final CAD centroid: {final_centroid}")
        print(f"[DEBUG] Distance to anchor: {np.linalg.norm(final_centroid - anchor_P_depth):.4f}m")

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()