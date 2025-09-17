# ------------------------------
# april_tag_pose_to_cad_open3d.py  (Detector-pose-only + CAD placement + origin debug)
# ------------------------------
#!/usr/bin/env python3
from __future__ import annotations
import math
import json
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
from pupil_apriltags import Detector

# -------------------- CONFIG --------------------
COLOR_PATH = Path(r"./hydroponic_system_captures/capture_20250903_114620/color_20250903_114620.png")
PLY_PATH   = Path(r"./hydroponic_system_captures/capture_20250903_114620/point_cloud_20250903_114620.ply")  # optional

# Intrinsics JSON: lean {"fx","fy","cx","cy","width","height"} or master {"color_intrinsics":{...}}
INTRINSICS_JSON = Path(r"./calibration_parameters/factory_color_intrinsics_2025-09-08T143506.json")

# AprilTag pose from detector
FAMILY     = "tag36h11"
TAG_ID     = -1          # -1 = largest by area
TAG_SIZE_M = 0.0303      # edge length of OUTER black square (meters)

# Viz sizes
AXES   = 0.05
SPHERE = 0.003
GRID   = 0.10
VOXEL  = 0.0

# CAD placement options
CAD_PLY                 = Path(r"../../cad_model/Structure2.PLY")
CAD_UNITS_TO_METERS     = 0.001         # 0.001 if CAD is in millimeters
CAD_PRE_ROT_DEG_ZYX     = (0.0, 0.0, 0.0)  # (Z, Y, X) degrees BEFORE placement
CAD_CENTER_ON_ORIGIN    = False         # keep False if you want CAD's local (0,0,0) on the tag
CAD_ORIGIN_OFFSET_LOCAL = (0.0, 0.0, 0.0)  # if CAD's intended anchor ≠ (0,0,0), give local offset (CAD units)

# -------------------- Intrinsics helpers --------------------
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

# -------------------- Detector pose (no PnP) --------------------
def detect_tag_pose_with_detector(img_bgr: np.ndarray,
                                  family: str,
                                  fx: float, fy: float, cx: float, cy: float,
                                  tag_size_m: float,
                                  prefer_id: int = -1):
    """
    Returns (R_ct (3x3), t_ct (3,), det_tag). Raises if this build doesn't expose pose fields.
    """
    det = Detector(families=family, nthreads=2, quad_decimate=1.0,
                   quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tags = det.detect(gray, estimate_tag_pose=True,
                      camera_params=[fx, fy, cx, cy],
                      tag_size=tag_size_m)
    if not tags:
        raise RuntimeError("No AprilTags detected (pose mode).")

    # pick requested id or largest by area
    if prefer_id >= 0:
        det_tag = next((t for t in tags
                        if getattr(t, 'tag_id', getattr(t, 'id', -999)) == prefer_id), None)
        if det_tag is None:
            raise RuntimeError(f"Requested tag id {prefer_id} not found.")
    else:
        def area(tt):
            c = tt.corners.astype(np.float32)
            return float(cv2.contourArea(c))
        det_tag = max(tags, key=area)

    # Extract pose; field names vary across builds
    R = getattr(det_tag, "pose_R", None)
    t = getattr(det_tag, "pose_t", None)
    if R is None or t is None:
        raise RuntimeError("Detector pose fields not found (pose_R/pose_t). "
                           "Upgrade pupil_apriltags or enable pose in your build.")
    R = np.array(R, dtype=float).reshape(3,3)
    t = np.array(t, dtype=float).reshape(3,)
    return R, t, det_tag

# -------------------- Detection / Pose --------------------
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
    # choose largest by area
    def area(tt):
        c = tt.corners.astype(np.float32)
        return float(cv2.contourArea(c))
    t = max(tags, key=area)
    return (float(t.center[0]), float(t.center[1])), t
    

# -------------------- Geometry helpers --------------------
def euler_zyx_to_R(z_deg: float, y_deg: float, x_deg: float) -> np.ndarray:
    z, y, x = [math.radians(a) for a in (z_deg, y_deg, x_deg)]
    cz, sz = math.cos(z), math.sin(z)
    cy, sy = math.cos(y), math.sin(y)
    cx, sx = math.cos(x), math.sin(x)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    return Rz @ Ry @ Rx  # Z * Y * X

def to_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T

def load_cad_geometry(path: Path):
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh and (len(np.asarray(mesh.vertices)) > 0):
        mesh.compute_vertex_normals()
        return mesh
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd and (len(np.asarray(pcd.points)) > 0):
        return pcd
    raise RuntimeError(f"Failed to load CAD from {path}")

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

def compute_reproj_error(obj_pts: np.ndarray, img_pts: np.ndarray,
                         rvec: np.ndarray, tvec: np.ndarray,
                         K: np.ndarray, dist: np.ndarray) -> float:
    """Mean L2 reprojection error in pixels."""
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(proj - img_pts, axis=1)))

def solve_pnp_with_best_obj_order(img_corners_px: np.ndarray, K: np.ndarray,
                                  dist: np.ndarray, tag_size_m: float):
    """
    Keep 'img_corners_px' as returned by detector (shape (4,2), pixels).
    Try several TL/TR/BR/BL object-point orderings; pick lowest reproj error,
    preferring positive Z.
    Returns: obj_pts(4x3), rvec(3x1), tvec(3x1), err_px(float), order_label(str)
    """
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
        score = err + (1000.0 if z <= 0 else 0.0)  # prefer positive-Z

        if score < best_score:
            best_score = score
            best = (obj_pts, rvec, tvec)
            best_err = err
            best_label = ",".join(label)

    if best is None:
        raise RuntimeError("solvePnP failed for all candidate corner orders.")

    obj_pts, rvec, tvec = best
    return obj_pts, rvec, tvec, best_err, best_label

# -------------------- Main --------------------
def main():
    # Load image
    if not COLOR_PATH.exists(): raise FileNotFoundError(COLOR_PATH)
    if not INTRINSICS_JSON.exists(): raise FileNotFoundError(INTRINSICS_JSON)
    img = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError("Failed to read COLOR image")
    h, w = img.shape[:2]

    # Intrinsics (JSON only), scaled to image size
    fx0, fy0, cx0, cy0, iw, ih = load_color_intrinsics(INTRINSICS_JSON)
    fx, fy, cx, cy = (scale_intrinsics(fx0, fy0, cx0, cy0, iw, ih, w, h)
                      if (iw and ih) else (fx0, fy0, cx0, cy0))
    print(f"[INTR] Using JSON intrinsics for {w}x{h} | "
          f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")

    # Detector pose (tag -> camera)
    # R_ct, t_ct, det_tag = detect_tag_pose_with_detector(
    #     img_bgr=img, family=FAMILY,
    #     fx=fx, fy=fy, cx=cx, cy=cy,
    #     tag_size_m=TAG_SIZE_M, prefer_id=TAG_ID
    # )
    (cx_pix, cy_pix), tag_center = detect_tag_center(img, FAMILY, TAG_ID)
    det_tag_for_pnp = tag_center

    # Ensure corners are in pixels (some pipelines output normalized [0..1])
    img_pts = det_tag_for_pnp.corners.astype(np.float64).reshape(4, 2)
    if img_pts.max() <= 1.5:  # likely normalized
        img_pts[:, 0] *= w
        img_pts[:, 1] *= h
        print("[FIX] PnP: corners looked normalized; scaled to pixels using image size.")

    K = build_K(fx, fy, cx, cy)
    dist = np.zeros((5,1), dtype=np.float64)  # set real distortion if you have it

    obj_pts, rvec_pnp, tvec_pnp, err_px, order_label = solve_pnp_with_best_obj_order(
        img_corners_px=img_pts, K=K, dist=dist, tag_size_m=TAG_SIZE_M
    )
    R_pnp, _ = cv2.Rodrigues(rvec_pnp)
    t_pnp = tvec_pnp.reshape(3)

    # T_cam_tag = to_4x4(R_ct, t_ct)
    T_cam_tag = np.eye(4, dtype=float)
    T_cam_tag[:3, :3] = R_pnp
    T_cam_tag[:3,  3] = t_pnp
    print(f"[POSE] solvepnp T_cam_tag: {T_cam_tag}")

    # --- Scene assembly ---
    geoms = []

    # Optional point cloud
    if PLY_PATH and PLY_PATH.exists():
        pcd = o3d.io.read_point_cloud(str(PLY_PATH))
        if VOXEL and VOXEL > 0: pcd = pcd.voxel_down_sample(voxel_size=float(VOXEL))
        geoms.append(pcd)

    # Tag axes + origin sphere (visual reference)
    axes_tag = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(AXES))
    axes_tag.transform(T_cam_tag)
    tag_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(SPHERE))
    tag_sphere.compute_vertex_normals(); tag_sphere.paint_uniform_color([1.0, 0.2, 0.2])
    # tag_sphere.translate(t_ct)
    tag_sphere.translate(t_pnp)
    geoms += [axes_tag, tag_sphere]

    # Optional ground grid
    if GRID and GRID > 0:
        geoms.append(make_xy_grid(cell=float(GRID), n=20, z=0.0))

    # --- Load & place CAD with its origin on the tag center ---
    if CAD_PLY and CAD_PLY.exists():
        cad = load_cad_geometry(CAD_PLY)

        # (1) Units → meters
        S = float(CAD_UNITS_TO_METERS)
        if S != 1.0:
            cad.scale(S, center=(0,0,0))

        # (2) Optional re-center (keep False to keep local origin as anchor)
        if CAD_CENTER_ON_ORIGIN:
            if isinstance(cad, o3d.geometry.TriangleMesh):
                cad.translate(-cad.get_center())
            else:
                cad.translate(-np.asarray(cad.points).mean(axis=0))

        # (3) Move chosen local anchor (in CAD native units) to origin
        ox, oy, oz = CAD_ORIGIN_OFFSET_LOCAL
        if (abs(ox) + abs(oy) + abs(oz)) > 0:
            offset_m = np.array([ox*S, oy*S, oz*S], dtype=float)
            cad.translate(-offset_m)

        # (4) Optional pre-rotation about CAD local origin (to match conventions)
        if any(abs(a) > 1e-6 for a in CAD_PRE_ROT_DEG_ZYX):
            Rpre = euler_zyx_to_R(*CAD_PRE_ROT_DEG_ZYX)
            cad.rotate(Rpre, center=(0,0,0))
        else:
            Rpre = np.eye(3)
        print(f"[INFO] CAD pre-rotation ZYX (deg): {Rpre}")

        # (5) Place CAD at the detector pose (tag origin → camera frame)
        cad.transform(T_cam_tag)
        geoms.append(cad)

        # --- DEBUG: visualize CAD local origin (after transforms) ---
        origin_local = np.zeros(3)
        if (abs(ox) + abs(oy) + abs(oz)) > 0:
            origin_local = -np.array([ox*S, oy*S, oz*S], dtype=float)
        origin_after_pre = (Rpre @ origin_local.reshape(3,1)).reshape(3)
        origin_world = (R_pnp @ origin_after_pre) + t_pnp

        # cad_origin_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(SPHERE)*1.2)
        # cad_origin_sphere.compute_vertex_normals()
        # cad_origin_sphere.paint_uniform_color([0.2, 0.9, 1.0])  # cyan
        # cad_origin_sphere.translate(origin_world)

        # cad_origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(AXES)*0.8)
        # cad_origin_axes.translate(origin_world)

        # geoms += [cad_origin_sphere, cad_origin_axes]

        # Numeric check (should be ~0 if anchor==local origin and no pre-rot/offset)
        # delta = t_pnp - origin_world
        # print("[DEBUG] CAD origin world:", origin_world)
        # print("[DEBUG] Tag center world:", t_pnp)
        # print("[DEBUG] Delta (tag - CADorigin) [m]:", delta,
        #       " |norm| [mm]:", np.linalg.norm(delta)*1000.0)
    else:
        print("[INFO] CAD_PLY not set or missing; skipping CAD placement.")

    o3d.visualization.draw_geometries(geoms)

if __name__ == '__main__':
    main()
