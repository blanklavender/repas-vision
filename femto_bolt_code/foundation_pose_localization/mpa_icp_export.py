#!/usr/bin/env python3
from __future__ import annotations
import math, json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import cv2
import open3d as o3d
from pupil_apriltags import Detector

# =========================================================
#                     CONFIG
# =========================================================
COLOR_PATH = Path(r"./new_test_captures/capture_20251031_155407/color_20251031_155407.png")
DEPTH_NPY  = Path(r"./new_test_captures/capture_20251031_155407/aligned_depth_m_20251031_155407.npy")
PLY_PATH   = Path(r"./new_test_exports/point_cloud_20251031_155407/cropped_camframe.ply")
INTRINSICS_JSON = Path(r"./calibration_parameters/factory_color_intrinsics_2025-09-08T143506.json")

FAMILY       = "tag36h11"
TAG_IDS      = [9, 16]
CAD_ANCHOR_ID= 16
TAG_SIZE_M   = 0.0293
CENTER_WIN   = 5

AXES   = 0.05
SPHERE = 0.003
GRID   = 0.10
VOXEL  = 0.0

CAD_PLY                 = Path(r"../../cad_model/StructureTotal-v2.PLY")
CAD_UNITS_TO_METERS     = 0.001
CAD_ADJUSTMENT_ROT_ZYX  = (0.0, -1, 0.0)  # (Z,Y,X) degrees, applied about anchor after main rotation

ANCHOR_DOMINANCE = 0.95

USE_ICP_REFINEMENT = True
ICP_MAX_CORRESPONDENCE_DIST = 0.05
ICP_MAX_ITERATIONS = 100
ICP_RELATIVE_FITNESS = 1e-6
ICP_RELATIVE_RMSE = 1e-6
CAD_SAMPLE_POINTS = 50000
SCENE_VOXEL_SIZE = 0.005

# ---------- Export paths ----------
EXPORT_DIR = Path("./cad_export")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_CAD_PLY = EXPORT_DIR / "cad_transformed.ply"      # mesh or pcd depending on CAD input
EXPORT_SCENE_PLY = EXPORT_DIR / "scene_copy.ply"         # optional scene copy (if desired)
EXPORT_META_JSON = EXPORT_DIR / "cad_transform_meta.json"

# =========================================================
#               INTRINSICS / UTIL
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
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

def median_depth(Z: np.ndarray, u: int, v: int, win: int) -> float:
    h, w = Z.shape
    r = max(1, win//2)
    u0,u1 = max(0,u-r), min(w,u+r+1)
    v0,v1 = max(0,v-r), min(h,v+r+1)
    patch = Z[v0:v1, u0:u1]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size else 0.0

# ---------- transform helpers (to record 4x4) ----------
def T_identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)

def T_translate(t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T

def T_rotate_about_point(R: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Rotation R about point c (in world):  T = Trans(c) * [R] * Trans(-c)"""
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    return T_translate(c) @ T @ T_translate(-np.asarray(c))

def T_scale_about_point(s: float, c: np.ndarray) -> np.ndarray:
    """Uniform scale s about point c."""
    S = np.eye(4, dtype=np.float64)
    S[0,0] = S[1,1] = S[2,2] = float(s)
    return T_translate(c) @ S @ T_translate(-np.asarray(c))

# =========================================================
#               OPEN3D GEOM
# =========================================================
def geom_centroid(g):
    if isinstance(g, o3d.geometry.TriangleMesh):
        return g.get_center()
    else:
        pts = np.asarray(g.points)
        return pts.mean(axis=0) if len(pts) else np.zeros(3)

def colored_axes_lines(size: float, colors_xyz=((1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0))):
    radius = max(size * 0.02, 1e-4)
    cx = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size)
    cy = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size)
    cz = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size)
    cx.paint_uniform_color(colors_xyz[0]); cy.paint_uniform_color(colors_xyz[1]); cz.paint_uniform_color(colors_xyz[2])
    cx.rotate(cx.get_rotation_matrix_from_xyz((0.0, -np.pi/2.0, 0.0)), center=(0,0,0)); cx.translate((size/2.0, 0.0, 0.0))
    cy.rotate(cy.get_rotation_matrix_from_xyz((np.pi/2.0, 0.0, 0.0)), center=(0,0,0)); cy.translate((0.0, size/2.0, 0.0))
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
    return Rz @ Ry @ Rx  # Z * Y * X

def refine_with_icp(cad_geometry, scene_pcd, max_correspondence_dist, max_iterations):
    # Point-cloud versions for ICP
    if isinstance(cad_geometry, o3d.geometry.TriangleMesh):
        print(f"[ICP] Converting CAD mesh to point cloud ({CAD_SAMPLE_POINTS} points)")
        cad_pcd = cad_geometry.sample_points_uniformly(number_of_points=CAD_SAMPLE_POINTS)
    else:
        cad_pcd = cad_geometry

    scene_downsampled = scene_pcd.voxel_down_sample(voxel_size=SCENE_VOXEL_SIZE)
    print(f"[ICP] Scene points: {len(scene_pcd.points)} → {len(scene_downsampled.points)} (downsampled)")
    print(f"[ICP] CAD points: {len(cad_pcd.points)}")

    scene_downsampled.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    cad_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )

    print(f"[ICP] Running point-to-plane ICP (max_dist={max_correspondence_dist}m, max_iter={max_iterations})")

    reg = o3d.pipelines.registration.registration_icp(
        cad_pcd, scene_downsampled,
        max_correspondence_dist,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations,
            relative_fitness=ICP_RELATIVE_FITNESS,
            relative_rmse=ICP_RELATIVE_RMSE
        )
    )

    print(f"[ICP] Fitness: {reg.fitness:.4f}, Inlier RMSE: {reg.inlier_rmse:.6f} m")
    print(f"[ICP] ΔT:\n{reg.transformation}")

    # Quick readable summary
    Rcorr = reg.transformation[:3, :3]
    tcorr = reg.transformation[:3, 3]
    angle = np.degrees(np.arccos(np.clip((np.trace(Rcorr) - 1) / 2, -1, 1)))
    print(f"[ICP] Correction: rotation={angle:.2f}°, translation={np.linalg.norm(tcorr)*1000:.2f}mm")

    return reg.transformation, reg.fitness, reg.inlier_rmse

# =========================================================
#                DETECTION + PnP + AVERAGING
# =========================================================
@dataclass
class DetectedTag:
    id: int
    corners_px: np.ndarray
    area: float

def detect_all_tags(img_bgr: np.ndarray, family: str) -> List[DetectedTag]:
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

def compute_reproj_error(obj_pts, img_pts, rvec, tvec, K, dist) -> float:
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - img_pts, axis=1)))

def solve_pnp_best_order(img_corners_px: np.ndarray, K: np.ndarray, dist: np.ndarray, tag_size_m: float):
    half = float(tag_size_m) / 2.0
    TL = np.array([-half, -half, 0.0]); TR = np.array([+half, -half, 0.0])
    BR = np.array([+half, +half, 0.0]); BL = np.array([-half, +half, 0.0])
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

def average_rotations_quat(R_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-6, None); w /= w.sum()
    Q = np.stack([R_to_quat(R) for R in R_list], axis=0)
    for i in range(1, len(Q)):
        if np.dot(Q[0], Q[i]) < 0: Q[i] = -Q[i]
    q_avg = (w[:,None]*Q).sum(axis=0); q_avg /= np.linalg.norm(q_avg)
    return quat_to_R(q_avg)

# =========================================================
#                        MAIN
# =========================================================
def main():
    # ---------- Load & intrinsics ----------
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

    fx0, fy0, cx0, cy0, iw, ih = load_color_intrinsics(INTRINSICS_JSON)
    fx, fy, cx, cy = (scale_intrinsics(fx0, fy0, cx0, cy0, iw, ih, w, h)
                      if (iw>0 and ih>0 and (iw!=w or ih!=h)) else (fx0, fy0, cx0, cy0))
    K = build_K(fx, fy, cx, cy)
    dist = np.zeros((5,1), dtype=np.float64)

    # ---------- Detect tags ----------
    detections = detect_all_tags(img, FAMILY)
    if not detections:
        raise RuntimeError("No AprilTags detected.")
    chosen = [d for d in detections if d.id in TAG_IDS]
    if len(chosen) == 0:
        raise RuntimeError(f"No requested tags {TAG_IDS} found. Detected: {[d.id for d in detections]}")

    R_list, t_list, w_list, id_list = [], [], [], []
    P_depth_list = []
    anchor_t = None
    anchor_P_depth = None

    for det in chosen:
        img_pts = det.corners_px.copy()
        if img_pts.max() <= 1.5:
            img_pts[:,0] *= w; img_pts[:,1] *= h

        R_tag, t_tag, err_px, order = solve_pnp_best_order(img_pts, K, dist, TAG_SIZE_M)
        weight = max(det.area, 1e-3) * (1.0 / max(err_px, 1e-3))

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

        if det.id == CAD_ANCHOR_ID:
            anchor_t = t_tag
            anchor_P_depth = P_depth_tag

    # Optional flip for tag 9 (if needed in your setup)
    for i, tid in enumerate(id_list):
        if tid == 9:
            R_flip = np.array([[-1,0,0],[0,-1,0],[0,0,1]], dtype=float)
            R_list[i] = R_list[i] @ R_flip
            print("[FIX] Applied 180° Z-rotation to tag 9")

    if anchor_t is None or anchor_P_depth is None:
        idx = int(np.argmax(np.asarray(w_list)))
        anchor_t = t_list[idx]
        anchor_P_depth = P_depth_list[idx]
        print(f"[WARN] Anchor tag {CAD_ANCHOR_ID} not visible; using id={id_list[idx]}")

    # Rotation averaging (anchor-dominant)
    if len(R_list) == 1:
        R_avg = R_list[0]
        print("[AVG] Single tag detected")
        weights_final = [1.0]
    else:
        anchor_idx = id_list.index(CAD_ANCHOR_ID)
        weights_final = []
        for i in range(len(R_list)):
            if i == anchor_idx:
                weights_final.append(ANCHOR_DOMINANCE)
            else:
                weights_final.append((1.0 - ANCHOR_DOMINANCE) / (len(R_list) - 1))
        R_avg = average_rotations_quat(R_list, weights_final)
        print(f"[AVG] Anchor dominance = {ANCHOR_DOMINANCE:.0%}  weights = {['%.3f'%w for w in weights_final]}")

    # ---------- Scene PCD ----------
    geoms = []
    scene_pcd = None
    if PLY_PATH and PLY_PATH.exists():
        scene_pcd = o3d.io.read_point_cloud(str(PLY_PATH))
        if VOXEL and VOXEL > 0:
            scene_pcd = scene_pcd.voxel_down_sample(voxel_size=float(VOXEL))
        geoms.append(scene_pcd)

    # Per-tag viz axes
    for i, tid in enumerate(id_list):
        P_depth = P_depth_list[i]
        if P_depth is not None:
            world_axes = colored_axes_lines(float(AXES)*0.8)
            T = np.eye(4)
            T[:3,:3] = R_list[i]
            T[:3, 3] = P_depth
            world_axes.transform(T)
            geoms.append(world_axes)

            sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(SPHERE)*2)
            sph.compute_vertex_normals()
            sph.paint_uniform_color([1.0, 0.0, 0.0] if tid == 9 else [0.0, 0.0, 1.0])
            sph.translate(P_depth)
            geoms.append(sph)

    if GRID and GRID > 0:
        geoms.append(make_xy_grid(cell=float(GRID), n=20, z=0.0))

    # ---------- Load & place CAD + accumulate transform ----------
    T_cad_world = T_identity()     # will accumulate the exact transform we apply
    T_cad_world_no_icp = None      # snapshot before ICP

    if CAD_PLY and CAD_PLY.exists() and anchor_P_depth is not None:
        cad = load_cad_geometry(CAD_PLY)

        # 1) Uniform scale about origin
        s = float(CAD_UNITS_TO_METERS)
        if isinstance(cad, o3d.geometry.TriangleMesh):
            cad.scale(s, center=(0,0,0))
        else:
            cad.scale(s, center=(0,0,0))
        T_cad_world = T_scale_about_point(s, np.zeros(3)) @ T_cad_world
        print(f"[CAD] Scaled by {s}")

        # 2) Translate to anchor
        cad.translate(anchor_P_depth)
        T_cad_world = T_translate(anchor_P_depth) @ T_cad_world
        print(f"[CAD] Translated to anchor: {anchor_P_depth}")

        # 3) Rotate by averaged rotation about the anchor point
        cad.rotate(R_avg, center=anchor_P_depth)
        T_cad_world = T_rotate_about_point(R_avg, anchor_P_depth) @ T_cad_world
        print("[CAD] Applied anchor-weighted rotation")

        # 4) Small adjustment rotation (ZYX, degrees) about anchor if requested
        if any(abs(a) > 1e-6 for a in CAD_ADJUSTMENT_ROT_ZYX):
            Radj = euler_zyx_to_R(*CAD_ADJUSTMENT_ROT_ZYX)
            cad.rotate(Radj, center=anchor_P_depth)
            T_cad_world = T_rotate_about_point(Radj, anchor_P_depth) @ T_cad_world
            print(f"[CAD] Applied adjustment rotation {CAD_ADJUSTMENT_ROT_ZYX}")

        # Save snapshot before ICP
        T_cad_world_no_icp = T_cad_world.copy()

        # 5) ICP refinement (optional)
        if USE_ICP_REFINEMENT and scene_pcd is not None:
            print("\n" + "="*60)
            print("[ICP] Starting refinement…")
            print("="*60)
            icp_T, fitness, rmse = refine_with_icp(
                cad, scene_pcd,
                ICP_MAX_CORRESPONDENCE_DIST,
                ICP_MAX_ITERATIONS
            )
            cad.transform(icp_T)                # apply to geometry
            T_cad_world = icp_T @ T_cad_world   # accumulate
        elif USE_ICP_REFINEMENT and scene_pcd is None:
            print("[ICP] ⚠ Scene PCD missing; skipping ICP")

        geoms.append(cad)

        # ---------- Export ----------
        print("\n[SAVE] Exporting…")
        # Export transformed CAD (mesh or pcd)
        if isinstance(cad, o3d.geometry.TriangleMesh):
            ok = o3d.io.write_triangle_mesh(str(EXPORT_CAD_PLY), cad, write_vertex_normals=True)
        else:
            ok = o3d.io.write_point_cloud(str(EXPORT_CAD_PLY), cad)
        if not ok:
            raise RuntimeError(f"Failed to write {EXPORT_CAD_PLY}")
        print(f"  CAD → {EXPORT_CAD_PLY}")

        # Optionally dump a copy of the scene for convenience
        if scene_pcd is not None:
            o3d.io.write_point_cloud(str(EXPORT_SCENE_PLY), scene_pcd)
            print(f"  Scene → {EXPORT_SCENE_PLY}")

        # Export meta (intrinsics + transform matrices)
        meta = {
            "inputs": {
                "color_path": str(COLOR_PATH),
                "depth_npy": str(DEPTH_NPY),
                "scene_ply": str(PLY_PATH),
                "cad_source": str(CAD_PLY),
            },
            "intrinsics": {
                "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
                "width": int(w), "height": int(h)
            },
            "tags": {
                "used_ids": [int(i) for i in id_list],
                "anchor_id": int(CAD_ANCHOR_ID),
                "weights_final": [float(x) for x in weights_final],
            },
            "parameters": {
                "cad_units_to_meters": float(CAD_UNITS_TO_METERS),
                "adjustment_rot_zyx_deg": [float(a) for a in CAD_ADJUSTMENT_ROT_ZYX],
                "use_icp": bool(USE_ICP_REFINEMENT),
                "icp_max_corr_dist_m": float(ICP_MAX_CORRESPONDENCE_DIST),
                "icp_max_iters": int(ICP_MAX_ITERATIONS),
            },
            "transforms": {
                "T_cad_world_no_icp": T_cad_world_no_icp.tolist() if T_cad_world_no_icp is not None else None,
                "T_cad_world_final":  T_cad_world.tolist(),
            }
        }
        EXPORT_META_JSON.write_text(json.dumps(meta, indent=2))
        print(f"  Meta → {EXPORT_META_JSON}\n")

    else:
        if not (CAD_PLY and CAD_PLY.exists()):
            print("[INFO] CAD_PLY not found")
        elif anchor_P_depth is None:
            print("[WARN] No valid anchor depth")

    # ---------- Viz ----------
    o3d.visualization.draw_geometries(geoms, window_name="CAD placement + export")

if __name__ == "__main__":
    main()
