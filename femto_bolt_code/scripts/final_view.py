# ------------------------------
# april_tag_center_to_open3d.py  (detector pose + solvePnP pose + depth point)
# ------------------------------
#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
from pupil_apriltags import Detector

# -------------------- CONFIG --------------------
COLOR_PATH = Path(r"./hydroponic_system_captures/capture_20250903_114620/color_20250903_114620.png")
DEPTH_NPY  = Path(r"./hydroponic_system_captures/capture_20250903_114620/aligned_depth_m_20250903_114620.npy")
PLY_PATH   = Path(r"./hydroponic_system_captures/capture_20250903_114620/point_cloud_20250903_114620.ply")  # optional

# Intrinsics JSON 
INTRINSICS_JSON = Path(r"./calibration_parameters/factory_color_intrinsics_2025-09-08T143506.json")

FAMILY      = "tag36h11"
TAG_ID      = -1            # -1 = pick largest by area
TAG_SIZE_M  = 0.0303         # <-- set your printed tag edge length (meters)
WINDOW      = 5             # odd, e.g., 3/5/7

AXES        = 0.05          # meters (size of axes)
SPHERE      = 0.003         # meters
GRID        = 0.10          # meters per cell (0 to disable)
VOXEL       = 0.0           # meters (0 to disable)

# -------------------- Intrinsics I/O --------------------
def load_color_intrinsics(json_path: Path):
    """Load COLOR intrinsics from JSON. Supports lean or master bundle format."""
    if not json_path.exists():
        raise FileNotFoundError(f"Intrinsics JSON not found: {json_path}")
    data = json.loads(json_path.read_text())

    # Accept either top-level lean dict or master bundle with color_intrinsics
    intr = data["color_intrinsics"] if (isinstance(data, dict) and "color_intrinsics" in data) else data

    required = ("fx", "fy", "cx", "cy")
    for k in required:
        if k not in intr:
            raise KeyError(f"Missing '{k}' in intrinsics JSON: {json_path}")

    fx = float(intr["fx"]); fy = float(intr["fy"])
    cx = float(intr["cx"]); cy = float(intr["cy"])
    iw = int(intr.get("width", 0)); ih = int(intr.get("height", 0))
    return fx, fy, cx, cy, iw, ih

def scale_intrinsics(fx, fy, cx, cy, src_w, src_h, dst_w, dst_h):
    """Scale intrinsics from (src_w,src_h) to (dst_w,dst_h)."""
    if src_w <= 0 or src_h <= 0:
        return fx, fy, cx, cy
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    return fx * sx, fy * sy, cx * sx, cy * sy

def build_K(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0,  fy, cy],
                     [0,   0,  1]], dtype=np.float64)

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

def detect_tag_pose_with_detector(img_bgr: np.ndarray,
                                  family: str,
                                  fx: float, fy: float, cx: float, cy: float,
                                  tag_size_m: float,
                                  prefer_id: int = -1):
    """
    Runs the detector with estimate_tag_pose=True and returns (R_ct, t_ct, det_tag).
    If this build doesn't expose pose fields, returns (None, None, det_tag).
    """
    det = Detector(families=family, nthreads=2, quad_decimate=1.0,
                   quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    tags = det.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=[fx, fy, cx, cy],
        tag_size=tag_size_m
    )
    if not tags:
        raise RuntimeError('No AprilTags detected (pose mode)')

    # Pick by ID or largest area
    if prefer_id >= 0:
        for t in tags:
            tid = getattr(t, 'tag_id', getattr(t, 'id', -999))
            if tid == prefer_id:
                det_tag = t
                break
        else:
            raise RuntimeError(f"Requested tag id {prefer_id} not found (pose mode)")
    else:
        def area(tt):
            c = tt.corners.astype(np.float32)
            return float(cv2.contourArea(c))
        det_tag = max(tags, key=area)

    # Try to extract pose from detector (field names vary by version)
    R = getattr(det_tag, "pose_R", None)
    t = getattr(det_tag, "pose_t", None)
    if R is not None:
        R = np.array(R, dtype=float).reshape(3, 3)
    if t is not None:
        t = np.array(t, dtype=float).reshape(3,)
    return R, t, det_tag

# -------------------- Utilities --------------------
def median_depth(Z: np.ndarray, u: int, v: int, win: int) -> float:
    h, w = Z.shape
    r = max(1, win//2)
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    patch = Z[v0:v1, u0:u1]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0:
        return 0.0
    return float(np.median(patch))

def color_pixel_to_3d(cx: float, cy: float, Zc: float, fx: float, fy: float, cxo: float, cyo: float):
    X = (cx - cxo) / fx * Zc
    Y = (cy - cyo) / fy * Zc
    return np.array([X, Y, Zc], dtype=np.float64)

def make_xy_grid(cell: float, n: int = 20, z: float = 0.0):
    extent = n * cell
    pts, lines, colors = [], [], []
    for y in np.linspace(-extent, extent, 2 * n + 1):
        pts += [[-extent, y, z], [extent, y, z]]
        lines.append([len(pts)-2, len(pts)-1]); colors.append([0.7,0.7,0.7])
    for x in np.linspace(-extent, extent, 2 * n + 1):
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

def colored_axes_lines(size: float,
                       colors_xyz=((1, 0, 1), (1, 1, 0), (0, 1, 1))  # X,Y,Z colors
                       ):
    
    # Thickness (radius) as a fraction of axis length; tweak if you want thicker/thinner lines
    radius = max(size * 0.02, 1e-4)  # ~2% of length, clamp to a tiny minimum

    # Build three cylinders of height `size`, centered at origin, aligned to +Z by default
    cyl_x = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size, resolution=32)
    cyl_y = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size, resolution=32)
    cyl_z = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size, resolution=32)

    # Color them per-axis
    cyl_x.paint_uniform_color(colors_xyz[0])
    cyl_y.paint_uniform_color(colors_xyz[1])
    cyl_z.paint_uniform_color(colors_xyz[2])

    # Open3D cylinders are centered at origin and extend from -h/2..+h/2 along +Z.
    # Rotate to align with axes, then translate by half-length so they start at the origin.
    # +X: rotate Z->X (yaw -90°), then move +X by size/2
    cyl_x.rotate(cyl_x.get_rotation_matrix_from_xyz((0.0, -np.pi/2.0, 0.0)), center=(0, 0, 0))
    cyl_x.translate((size/2.0, 0.0, 0.0))

    # +Y: rotate Z->Y (pitch +90°), then move +Y by size/2
    cyl_y.rotate(cyl_y.get_rotation_matrix_from_xyz((np.pi/2.0, 0.0, 0.0)), center=(0, 0, 0))
    cyl_y.translate((0.0, size/2.0, 0.0))

    # +Z: already aligned; just move +Z by size/2
    cyl_z.translate((0.0, 0.0, size/2.0))

    # Merge into one mesh so you can transform once and add once
    axes_mesh = cyl_x + cyl_y + cyl_z
    axes_mesh.compute_vertex_normals()
    return axes_mesh



# -------------------- Main --------------------
def main():
    # --- Load files ---
    if not COLOR_PATH.exists(): raise FileNotFoundError(COLOR_PATH)
    if not DEPTH_NPY.exists():  raise FileNotFoundError(DEPTH_NPY)
    if not INTRINSICS_JSON.exists(): raise FileNotFoundError(INTRINSICS_JSON)

    img = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    Zc_img = np.load(str(DEPTH_NPY))  # meters, COLOR grid
    if img is None: raise RuntimeError('Failed to read COLOR image')
    if Zc_img.ndim != 2: raise RuntimeError('Aligned depth .npy must be 2D (meters)')

    h, w = img.shape[:2]
    if Zc_img.shape != (h, w):
        raise RuntimeError(f"Size mismatch: COLOR {w}x{h} vs aligned depth {Zc_img.shape[1]}x{Zc_img.shape[0]}")

    # --- Load & (if needed) scale intrinsics to image size ---
    fx0, fy0, cx0, cy0, iw, ih = load_color_intrinsics(INTRINSICS_JSON)
    fx, fy, cx, cy = (scale_intrinsics(fx0, fy0, cx0, cy0, iw, ih, w, h)
                      if (iw > 0 and ih > 0 and (iw != w or ih != h))
                      else (fx0, fy0, cx0, cy0))
    if iw and ih and (iw != w or ih != h):
        print(f"[INTR] JSON {iw}x{ih} → scaled to {w}x{h} | fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    else:
        print(f"[INTR] Using JSON intrinsics for {w}x{h} | fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")

    K = build_K(fx, fy, cx, cy)
    dist = np.zeros((5,1), dtype=np.float64)  # set real distortion if you have it

    # --- Detect AprilTag center (for depth-based 3D point) ---
    (cx_pix, cy_pix), tag_center = detect_tag_center(img, FAMILY, TAG_ID)
    u = int(round(cx_pix)); v = int(round(cy_pix))

    Zc = median_depth(Zc_img, u, v, WINDOW)
    if Zc <= 0:
        raise RuntimeError('No valid depth around tag center. Increase WINDOW or check alignment.')

    P_depth = color_pixel_to_3d(cx_pix, cy_pix, Zc, fx, fy, cx, cy)
    print('[POINT] Depth-based tag center 3D (camera frame) [m]: '
          f'[{P_depth[0]:.4f}, {P_depth[1]:.4f}, {P_depth[2]:.4f}]')

    # --- Detector pose (6-DoF) ---
    R_det, t_det, det_tag_pose = detect_tag_pose_with_detector(
        img_bgr=img, family=FAMILY,
        fx=fx, fy=fy, cx=cx, cy=cy,
        tag_size_m=TAG_SIZE_M,
        prefer_id=TAG_ID
    )
    have_det_pose = (R_det is not None) and (t_det is not None)
    if have_det_pose:
        T_cam_tag_det = np.eye(4, dtype=float)
        T_cam_tag_det[:3, :3] = R_det
        T_cam_tag_det[:3,  3] = t_det

        np.set_printoptions(precision=6, suppress=True)
        print("[POSE-DET] Detector pose (tag -> camera):")
        print("          R =\n", R_det)
        print("          t =", t_det)
        print("          ||t - P_depth|| [mm] =", np.linalg.norm(t_det - P_depth)*1000.0)
    else:
        print("[POSE-DET] Detector build did not expose pose fields; skipping tag-aligned axes (detector).")

    # --- solvePnP pose (6-DoF) using corners + intrinsics + TAG_SIZE_M ---
    # Use the same tag picked in pose mode if available; otherwise use the one from center-only pass.
    # --- solvePnP pose (6-DoF) using corners + intrinsics + TAG_SIZE_M (robust order) ---
    det_tag_for_pnp = det_tag_pose if have_det_pose else tag_center

    # Ensure corners are in pixels (some pipelines output normalized [0..1])
    img_pts = det_tag_for_pnp.corners.astype(np.float64).reshape(4, 2)
    if img_pts.max() <= 1.5:  # likely normalized
        img_pts[:, 0] *= w
        img_pts[:, 1] *= h
        print("[FIX] PnP: corners looked normalized; scaled to pixels using image size.")

    obj_pts, rvec_pnp, tvec_pnp, err_px, order_label = solve_pnp_with_best_obj_order(
        img_corners_px=img_pts, K=K, dist=dist, tag_size_m=TAG_SIZE_M
    )
    R_pnp, _ = cv2.Rodrigues(rvec_pnp)
    t_pnp = tvec_pnp.reshape(3)

    print("[POSE-PNP] OpenCV solvePnP pose (tag -> camera) [best-order]:")
    print("          order:", order_label)
    print("          reproj err [px]:", err_px)
    print("          R =\n", R_pnp)
    print("          t =", t_pnp)
    print("          ||t - P_depth|| [mm] =", np.linalg.norm(t_pnp - P_depth)*1000.0)
    if have_det_pose:
        print("          ||t_pnp - t_det|| [mm] =", np.linalg.norm(t_pnp - t_det)*1000.0)

    T_cam_tag_pnp = np.eye(4, dtype=float)
    T_cam_tag_pnp[:3, :3] = R_pnp
    T_cam_tag_pnp[:3,  3] = t_pnp

    print("[POSE-PNP] OpenCV solvePnP pose (tag -> camera):")
    print("          R =\n", R_pnp)
    print("          t =", t_pnp)
    print("          ||t - P_depth|| [mm] =", np.linalg.norm(t_pnp - P_depth)*1000.0)
    if have_det_pose:
        print("          ||t_pnp - t_det|| [mm] =", np.linalg.norm(t_pnp - t_det)*1000.0)

    # --- Build Open3D scene (retain your existing display) ---
    geoms = []
    if PLY_PATH and PLY_PATH.exists():
        pcd = o3d.io.read_point_cloud(str(PLY_PATH))
        if VOXEL and VOXEL > 0:
            pcd = pcd.voxel_down_sample(voxel_size=float(VOXEL))
        geoms.append(pcd)

    # 1) Depth-based center (RED) + world-aligned axes (translated only)
    sphere_depth = o3d.geometry.TriangleMesh.create_sphere(radius=float(SPHERE))
    sphere_depth.compute_vertex_normals(); sphere_depth.paint_uniform_color([1.0, 0.2, 0.2])  # red
    sphere_depth.translate(P_depth)

    axes_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(AXES))
    axes_world.translate(P_depth)  # translation only (no rotation)

    geoms += [sphere_depth, axes_world]

    # 2) Detector pose (MAGENTA sphere) + oriented axes (if available)
    if have_det_pose:
        axes_det = colored_axes_lines(float(AXES), ((1.0, 0.0, 1.0),   # X magenta
                                                (1.0, 0.0, 1.0),   
                                                (1.0, 0.0, 1.0)))  
        axes_det.transform(T_cam_tag_det)

        sphere_det = o3d.geometry.TriangleMesh.create_sphere(radius=float(SPHERE)*1.0)
        sphere_det.compute_vertex_normals(); sphere_det.paint_uniform_color([1.0, 0.0, 1.0])  # magenta
        sphere_det.translate(t_det)

        geoms += [axes_det, sphere_det]

    # 3) solvePnP pose (CYAN sphere) + oriented axes
    axes_pnp = colored_axes_lines(float(AXES), ((0.0, 1.0, 1.0),   # X magenta
                                                (0.0, 1.0, 1.0),   
                                                (0.0, 1.0, 1.0)))  
    axes_pnp.transform(T_cam_tag_pnp)

    sphere_pnp = o3d.geometry.TriangleMesh.create_sphere(radius=float(SPHERE)*0.9)
    sphere_pnp.compute_vertex_normals(); sphere_pnp.paint_uniform_color([0.1, 0.8, 1.0])  # cyan
    sphere_pnp.translate(t_pnp)

    geoms += [axes_pnp, sphere_pnp]

    if GRID and GRID > 0:
        geoms.append(make_xy_grid(cell=float(GRID), n=20, z=0.0))

    o3d.visualization.draw_geometries(geoms)

if __name__ == '__main__':
    main()
