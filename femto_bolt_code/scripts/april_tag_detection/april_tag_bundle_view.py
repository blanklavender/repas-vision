#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
from pupil_apriltags import Detector
from dataclasses import dataclass

# ----------------- CONFIG -----------------
COLOR_PATH = Path(r"./captures_colorworld/capture_20250917_164436/color_20250917_164436.png")
DEPTH_NPY  = Path(r"./captures_colorworld/capture_20250917_164436/aligned_depth_m_20250917_164436.npy")
PLY_PATH   = Path(r"./captures_colorworld/capture_20250917_164436/point_cloud_20250917_164436.ply")
INTRINSICS_JSON = Path(r"./calibration_parameters/factory_color_intrinsics_2025-09-08T143506.json")

FAMILY     = "tag36h11"
TAG_IDS    = [9, 16]         # two tags to use
TAG_SIZE_M = 0.0293
CENTER_WIN = 5

AXES   = 0.05
SPHERE = 0.003
GRID   = 0.10
VOXEL  = 0.0

# --------------- I/O + intrinsics ---------------
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
    if src_w <= 0 or src_h <= 0:
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

# --------------- Open3D helpers ---------------
def colored_axes_lines(size: float, colors_xyz=((0.0,1.0,1.0),(0.0,1.0,1.0),(0.0,1.0,1.0))):
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

# --------------- PnP + detection ---------------
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

@dataclass
class DetectedTag:
    id: int
    corners_px: np.ndarray  # (4,2)
    area: float

def detect_all_tags(img_bgr: np.ndarray, family: str) -> list[DetectedTag]:
    det = Detector(families=family, nthreads=2, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ts = det.detect(gray, estimate_tag_pose=False)
    out = []
    for t in ts:
        tid = getattr(t, 'tag_id', getattr(t, 'id', -999))
        c = t.corners.astype(np.float64).reshape(4,2)
        area = float(cv2.contourArea(c.astype(np.float32)))
        out.append(DetectedTag(id=int(tid), corners_px=c, area=area))
    return out

# --------------- Rotation averaging (quaternion) ---------------
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
    # make quats same hemisphere to avoid cancel
    for i in range(1, len(Q)):
        if np.dot(Q[0], Q[i]) < 0: Q[i] = -Q[i]
    q_avg = (w[:,None]*Q).sum(axis=0); q_avg /= np.linalg.norm(q_avg)
    return quat_to_R(q_avg)

# --------------- Main ---------------
def main():
    # --- load files ---
    if not COLOR_PATH.exists(): raise FileNotFoundError(COLOR_PATH)
    if not DEPTH_NPY.exists():  raise FileNotFoundError(DEPTH_NPY)
    if not INTRINSICS_JSON.exists(): raise FileNotFoundError(INTRINSICS_JSON)

    img = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    Zc_img = np.load(str(DEPTH_NPY))
    if img is None: raise RuntimeError("Failed to read image")
    h, w = img.shape[:2]
    if Zc_img.shape != (h, w):
        raise RuntimeError(f"Size mismatch: COLOR {w}x{h} vs depth {Zc_img.shape[1]}x{Zc_img.shape[0]}")

    fx0, fy0, cx0, cy0, iw, ih = load_color_intrinsics(INTRINSICS_JSON)
    fx, fy, cx, cy = (scale_intrinsics(fx0, fy0, cx0, cy0, iw, ih, w, h)
                      if (iw>0 and ih>0 and (iw!=w or ih!=h)) else (fx0, fy0, cx0, cy0))
    K = build_K(fx, fy, cx, cy)
    dist = np.zeros((5,1), dtype=np.float64)

    # --- detect tags ---
    detections = detect_all_tags(img, FAMILY)
    chosen = [d for d in detections if d.id in TAG_IDS]
    if len(chosen) == 0:
        raise RuntimeError(f"No tags from {TAG_IDS} found. Detected: {[d.id for d in detections]}")
    if len(chosen) == 1:
        print("[WARN] Only one of the requested tags is visible. Using its rotation for both axes.")

    # --- PnP per tag ---
    R_list, t_list, w_list = [], [], []
    for det in chosen:
        img_pts = det.corners_px.copy()
        # if normalized, rescale
        if img_pts.max() <= 1.5:
            img_pts[:,0] *= w; img_pts[:,1] *= h
        R_tag, t_tag, err_px, _ = solve_pnp_best_order(img_pts, K, dist, TAG_SIZE_M)
        print(R_tag, t_tag)
        # weight = larger area and lower reproj error
        weight = max(det.area, 1e-3) * (1.0 / max(err_px, 1e-3))
        R_list.append(R_tag); t_list.append(t_tag); w_list.append(weight)
        print(f"[PnP] id={det.id} reproj={err_px:.2f}px area={det.area:.0f} weight={weight:.1f}")

    # --- average rotation across visible tags ---
    if len(R_list) == 1:
        R_avg = R_list[0]
    else:
        R_avg = average_rotations_quat(R_list, w_list)

    # --- build Open3D scene ---
    geoms = []
    if PLY_PATH and PLY_PATH.exists():
        pcd = o3d.io.read_point_cloud(str(PLY_PATH))
        if VOXEL and VOXEL > 0: pcd = pcd.voxel_down_sa-mple(voxel_size=float(VOXEL))
        geoms.append(pcd)

    # plot two axes at each tag's translation, both oriented by the same averaged R
    for t_tag in t_list:
        T = np.eye(4); T[:3,:3] = R_avg; T[:3,3] = t_tag
        axes = colored_axes_lines(float(AXES))
        axes.transform(T)
        geoms.append(axes)

        # optional little sphere at the tag position
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(SPHERE))
        sph.compute_vertex_normals(); sph.paint_uniform_color([0.8, 0.2, 1.0])
        sph.translate(t_tag)
        geoms.append(sph)

        # optional: show a world-aligned frame at the depth-validated point near this tag center
        # (project t_tag to pixels & sample median depth)
        Pc = t_tag
        if Pc[2] > 1e-6:
            uv = (K @ Pc.reshape(3,1)).reshape(3); u,v = int(round(uv[0]/uv[2])), int(round(uv[1]/uv[2]))
            if 0 <= u < w and 0 <= v < h:
                Zc = median_depth(Zc_img, u, v, CENTER_WIN)
                if Zc > 0:
                    X = (u - cx) / fx * Zc; Y = (v - cy) / fy * Zc
                    P_depth = np.array([X, Y, Zc], dtype=float)
                    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(AXES)*0.6)
                    world_axes.translate(P_depth)
                    geoms.append(world_axes)

    if GRID and GRID > 0:
        geoms.append(make_xy_grid(cell=float(GRID), n=20, z=0.0))

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
