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

# Intrinsics JSON: either lean {"fx","fy","cx","cy","width","height"}
# or master {"color_intrinsics": {"fx","fy","cx","cy","width","height"}}
INTRINSICS_JSON = Path(r"./calibration_parameters/factory_color_intrinsics_2025-09-08T143506.json")

# AprilTags
FAMILY       = "tag36h11"
TAG_IDS      = [9, 16]    # tags to use for rotation averaging
CAD_ANCHOR_ID= 16          # which tag's translation to use for CAD placement
TAG_SIZE_M   = 0.0293     # outer black square edge length (meters)
CENTER_WIN   = 5          # depth median window for optional debug axes

# Viz sizes
AXES   = 0.05
SPHERE = 0.003
GRID   = 0.10
VOXEL  = 0.0   # meters; 0 = no downsampl

# CAD placement options
CAD_PLY                 = Path(r"../../cad_model/StructureTotal-v2.PLY")
CAD_UNITS_TO_METERS     = 0.001                 # 0.001 if CAD is authored in millimeters
CAD_PRE_ROT_DEG_ZYX     = (0.0, 0.0, 0.0)       # (Z, Y, X) degrees BEFORE placement
CAD_CENTER_ON_ORIGIN    = False                 # True if you want CAD recentered on its own bbox center
CAD_ORIGIN_OFFSET_LOCAL = (0.0, 0.0, 0.0)       # move this local point to tag origin before placement

# Export options
EXPORT_TRANSFORMED_CAD  = True                  # Set to True to export transformed CAD
OUTPUT_CAD_PLY          = Path(r"./new_test_exports/transformed_cad_output.ply")  # Output path for transformed CAD

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

def save_cad_geometry(geom, path: Path):
    """Save CAD geometry (mesh or point cloud) to PLY file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(geom, o3d.geometry.TriangleMesh):
        success = o3d.io.write_triangle_mesh(str(path), geom, write_ascii=False, compressed=True)
        geom_type = "mesh"
    elif isinstance(geom, o3d.geometry.PointCloud):
        success = o3d.io.write_point_cloud(str(path), geom, write_ascii=False, compressed=True)
        geom_type = "point cloud"
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")
    
    if success:
        print(f"[EXPORT] Successfully saved transformed CAD {geom_type} to: {path}")
        print(f"[EXPORT] File size: {path.stat().st_size / 1024:.2f} KB")
    else:
        print(f"[ERROR] Failed to save transformed CAD to: {path}")
    
    return success

def euler_zyx_to_R(z_deg: float, y_deg: float, x_deg: float) -> np.ndarray:
    z, y, x = [math.radians(a) for a in (z_deg, y_deg, x_deg)]
    cz, sz = math.cos(z), math.sin(z)
    cy, sy = math.cos(y), math.sin(y)
    cx, sx = math.cos(x), math.sin(x)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    return Rz @ Ry @ Rx  # Z * Y * X

# =========================================================
#                DETECTION + PnP + AVERAGING
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

def R_to_quat(R):
    rvec, _ = cv2.Rodrigues(R); theta = np.linalg.norm(rvec)
    if theta < 1e-12: return np.array([1,0,0,0], dtype=float)
    axis = (rvec/theta).reshape(3)
    qw = np.cos(theta/2.0); qv = axis*np.sin(theta/2.0)
    return np.array([qw, qv[0], qv[1], qv[2]], dtype=float)

def quat_to_R(q):
    q = q/np.linalg.norm(q)
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)

def average_rotations_quat(R_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-6, None); w /= w.sum()
    Q = np.stack([R_to_quat(R) for R in R_list], axis=0)
    # align hemisphere to avoid cancellation
    for i in range(1, len(Q)):
        if np.dot(Q[0], Q[i]) < 0: Q[i] = -Q[i]
    q_avg = (w[:,None]*Q).sum(axis=0); q_avg /= np.linalg.norm(q_avg)
    return quat_to_R(q_avg)

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

    # filter to the set we need for averaging
    chosen = [d for d in detections if d.id in TAG_IDS]
    if len(chosen) == 0:
        raise RuntimeError(f"No requested tags {TAG_IDS} found. Detected: {[d.id for d in detections]}")
    if len(chosen) == 1:
        print("[WARN] Only one of the requested tags is visible; rotation average will just use that one.")

    # --- PnP per chosen tag ---
    R_list, t_list, w_list, id_list = [], [], [], []
    P_depth_list = []  # Store P_depth for each tag
    anchor_t = None
    anchor_P_depth = None
    
    for det in chosen:
        img_pts = det.corners_px.copy()
        # Normalize fix if needed
        if img_pts.max() <= 1.5:
            img_pts[:,0] *= w; img_pts[:,1] *= h
            print(f"[FIX] Tag {det.id}: corners looked normalized; scaled to pixels.")

        R_tag, t_tag, err_px, order = solve_pnp_best_order(img_pts, K, dist, TAG_SIZE_M)
        weight = max(det.area, 1e-3) * (1.0 / max(err_px, 1e-3))
        
        # Compute P_depth for this tag
        P_depth_tag = None
        if Zc_img is not None and float(t_tag[2]) > 1e-6:
            uv = (K @ t_tag.reshape(3,1)).reshape(3)
            u, v = int(round(uv[0]/uv[2])), int(round(uv[1]/uv[2]))
            if 0 <= u < w and 0 <= v < h:
                Zc = median_depth(Zc_img, u, v, CENTER_WIN)
                if Zc > 0:
                    X = (u - cx) / fx * Zc
                    Y = (v - cy) / fy * Zc
                    P_depth_tag = np.array([X, Y, Zc], dtype=float)
        
        R_list.append(R_tag)
        t_list.append(t_tag)
        w_list.append(weight)
        id_list.append(det.id)
        P_depth_list.append(P_depth_tag)
        
        print(f"[PnP] id={det.id} reproj={err_px:.2f}px area={det.area:.0f} weight={weight:.1f} order={order}")
        print(f"      R_tag:\n{R_tag}")
        print(f"      t_tag: {t_tag}")
        if P_depth_tag is not None:
            print(f"      P_depth: {P_depth_tag}")
        
        if det.id == CAD_ANCHOR_ID:
            anchor_t = t_tag
            anchor_P_depth = P_depth_tag
    
    # TEMP_FIX FOR TAG 9 ORIENTATION ISSUE
    # After detecting tag 9, flip its coordinate frame
    for i, tid in enumerate(id_list):
        if tid == 9:
            # Apply 180° rotation around Z-axis to flip X and Y
            R_flip = np.array([[-1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1]], dtype=float)
            R_list[i] = R_list[i] @ R_flip
            print(f"[FIX] Applied 180° Z-rotation correction to tag 9")
            print(f"[FIX] Rvec of Tag 9:\n{R_list[i]}")

    # Compute averaged rotation across visible tags (weighted by area / error)
    if len(R_list) == 1:
        R_avg = R_list[0]
        print("[AVG] Only one tag detected, using its rotation")
    else:
        R_avg = average_rotations_quat(R_list, w_list)
        print(f"[AVG] Rotation averaged from {len(R_list)} tags:", id_list)
    
    print(f"[AVG] R_avg:\n{R_avg}")

    # Determine CAD translation anchor (prefer anchor id; fallback to highest weight)
    if anchor_t is None or anchor_P_depth is None:
        idx = int(np.argmax(np.asarray(w_list)))
        anchor_t = t_list[idx]
        anchor_P_depth = P_depth_list[idx]
        print(f"[WARN] Anchor tag id {CAD_ANCHOR_ID} not visible or no depth; using id={id_list[idx]} as CAD anchor.")

    # --- Build Open3D scene ---
    geoms = []

    # Optional scene point cloud
    if PLY_PATH and PLY_PATH.exists():
        pcd = o3d.io.read_point_cloud(str(PLY_PATH))
        if VOXEL and VOXEL > 0:
            pcd = pcd.voxel_down_sample(voxel_size=float(VOXEL))
        geoms.append(pcd)

    # Show per-tag markers with depth validation
    for i, (t_tag, tid) in enumerate(zip(t_list, id_list)):
        P_depth = P_depth_list[i]
        if P_depth is not None:
            # Show axes at each tag's depth position
            world_axes = colored_axes_lines(float(AXES)*0.8)
            # Apply the tag's rotation
            T = np.eye(4)
            T[:3, :3] = R_list[i]
            T[:3, 3] = P_depth
            world_axes.transform(T)
            geoms.append(world_axes)
            
            # Label sphere
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(SPHERE)*2)
            sph.compute_vertex_normals()
            sph.paint_uniform_color([1.0, 0.0, 0.0] if tid == 9 else [0.0, 0.0, 1.0])
            sph.translate(P_depth)
            geoms.append(sph)

    if GRID and GRID > 0:
        geoms.append(make_xy_grid(cell=float(GRID), n=20, z=0.0))

    # --- Load & place CAD ---
    transformed_cad = None  # Track the transformed CAD for export
    
    if CAD_PLY and CAD_PLY.exists() and anchor_P_depth is not None:
        cad = load_cad_geometry(CAD_PLY)
        
        # Step 1: Get original centroid (in CAD units, e.g., mm)
        centroid_original = geom_centroid(cad)
        print(f"[CAD] Original centroid (CAD units): {centroid_original}")
        
        # Step 2: Apply uniform scaling about centroid (keeps geometry proportions)
        S = float(CAD_UNITS_TO_METERS)  # e.g., 0.001 for mm->m
        cad.scale(S, center=centroid_original)
        print(f"[CAD] Scaled by {S}")
        
        # Step 3: Get new centroid (now in meters, numerically same as centroid_original)
        centroid_meters = geom_centroid(cad)
        print(f"[CAD] Centroid after scale (meters): {centroid_meters}")
        
        # Step 3.5: Calculate CAD origin position in local coordinates (before rotation)
        # The origin (0,0,0) in original CAD coordinates is now at:
        cad_origin_local = centroid_meters + (-centroid_original * S)
        print(f"[CAD] CAD origin in local coords (meters): {cad_origin_local}")
        
        # Step 4: Apply averaged rotation about CAD origin
        cad.rotate(R_avg, center=cad_origin_local)
        print(f"[CAD] Applied R_avg rotation about CAD origin")
        
        # Step 5: CAD origin position stays the same (we rotated about it)
        origin_world = cad_origin_local

        # # Step 4: Apply averaged rotation about CAD centroid (instead of origin)
        # cad.rotate(R_avg, center=centroid_meters)
        # print(f"[CAD] Applied R_avg rotation about CAD centroid")

        # # Step 5: After rotation about centroid, the CAD origin has moved
        # # We need to calculate where it ended up
        # # The origin transforms as: new_origin = centroid + R_avg @ (origin - centroid)
        # origin_world = centroid_meters + R_avg @ (cad_origin_local - centroid_meters)
        # print(f"[CAD] CAD origin after rotation: {origin_world}")
        
        # Step 6: Translate so CAD origin lands at anchor_P_depth (tag 16 position)
        translation = anchor_P_depth - origin_world
        cad.translate(translation)
        print(f"[CAD] Translated CAD origin to anchor P_depth: {anchor_P_depth}")
        print(f"[CAD] Translation vector: {translation}")
        
        # Step 7: Apply CAD_PRE_ROT_DEG_ZYX rotation about the CAD origin (now at anchor_P_depth)
        if any(abs(a) > 1e-6 for a in CAD_PRE_ROT_DEG_ZYX):
            Rpre = euler_zyx_to_R(*CAD_PRE_ROT_DEG_ZYX)
            cad.rotate(Rpre, center=anchor_P_depth)
            print(f"[CAD] Applied pre-rotation ZYX {CAD_PRE_ROT_DEG_ZYX} about CAD origin at P_depth")
        
        geoms.append(cad)
        transformed_cad = cad  # Save reference for export
        
        # --- Debug output ---
        final_centroid = geom_centroid(cad)     
        print(f"[DEBUG] Final CAD centroid (meters): {final_centroid}")
        print(f"[DEBUG] Anchor P_depth (meters): {anchor_P_depth}")
        print(f"[DEBUG] Distance centroid to anchor (m): {np.linalg.norm(final_centroid - anchor_P_depth):.4f}")
        
        # Verify CAD origin is at anchor_P_depth by checking a known point
        # The CAD origin (0,0,0) should now be at anchor_P_depth
        print(f"[DEBUG] CAD origin should be at: {anchor_P_depth}")
        
        # --- Export transformed CAD ---
        if EXPORT_TRANSFORMED_CAD and transformed_cad is not None:
            print("\n" + "="*60)
            print("EXPORTING TRANSFORMED CAD MODEL")
            print("="*60)
            save_cad_geometry(transformed_cad, OUTPUT_CAD_PLY)
            print("="*60 + "\n")
        
    else:
        if not (CAD_PLY and CAD_PLY.exists()):
            print("[INFO] CAD_PLY not set or not found; skipping CAD placement.")
        elif anchor_P_depth is None:
            print("[WARN] No valid anchor P_depth; skipping CAD placement.")

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()