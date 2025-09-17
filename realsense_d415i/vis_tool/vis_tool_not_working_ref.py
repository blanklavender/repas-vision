import json
import math
import cv2
import numpy as np
import open3d as o3d
from pupil_apriltags import Detector

# ======================
# Paths & config
# ======================
PLY_PATH   = r"../02_ply_generation/ply_generation_scripts/not_aligned_outputs/pose 1/out_20250808_142429.ply"
IMAGE_PATH = r"../02_ply_generation/ply_generation_scripts/not_aligned_outputs/pose 1/rgb_20250808_142429.png"

# Intrinsics JSON for the COLOR stream (matches IMAGE_PATH)
INTRINSICS_JSON = r"../00_april_tag_detection_callibration/color_intrinsics_1280_720_copy.json"

# Tag + detection
TAG_SIZE = 0.0303  # meters (outer black square)
MARGIN_THRESH = 10.0

# Viewer scale (UI only; does not affect geometry)
VIEW_SCALE = 1.0

# ======================
# CAD model configuration
# ======================
CAD_MODEL_PATH = r"../../cad_model/Structure2.PLY"  # ← set this
CAD_UNITS = "mm"  # one of {"m","cm","mm"}; converts to meters
CAD_UNIFORM_SCALE = 1.0
# If your CAD's local axes differ from the tag's axes, apply a pre-rotation (XYZ order, degrees)
CAD_PRE_ROT_EULER_DEG = (180.0, 0.0, 0.0)
# CAD local origin offset (before rotation), in CAD units
CAD_ORIGIN_OFFSET_LOCAL = np.array([0.0, 0.0, 0.0], dtype=np.float64)
COLOR_CAD_IF_UNCOLORED = True
CAD_SOLID_COLOR = (0.85, 0.85, 0.2)  # RGB in [0,1]

# ======================
# Helpers
# ======================
def load_intrinsics(json_path):
    with open(json_path, "r") as f:
        d = json.load(f)
    K = np.array([
        [d["fx"], 0.0, d["ppx"]],
        [0.0, d["fy"], d["ppy"]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist = np.array(d.get("coeffs", [0,0,0,0,0]), dtype=np.float64)
    return K, dist, (int(d["width"]), int(d["height"]))

def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def detect_with_params(gray_img, quad_decimate, quad_sigma):
    detector = Detector(
        families='tag36h11',
        nthreads=4,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    return detector.detect(gray_img)

def detect_best_tag(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    variants = [gray, enhance_contrast(gray)]
    best = None

    for g in variants:
        tags = detect_with_params(g, quad_decimate=1.0, quad_sigma=0.0)
        good = [t for t in tags if t.decision_margin >= MARGIN_THRESH]

        if not good:
            tags = detect_with_params(g, quad_decimate=0.5, quad_sigma=1.0)
            good = [t for t in tags if t.decision_margin >= MARGIN_THRESH]

        if good:
            cand = max(good, key=lambda t: t.decision_margin)
            if (best is None) or (cand.decision_margin > best.decision_margin):
                best = cand
            break
        elif tags:
            cand = max(tags, key=lambda t: t.decision_margin)
            if (best is None) or (cand.decision_margin > best.decision_margin):
                best = cand

    if best is None:
        raise RuntimeError("No AprilTags detected in the image.")
    return best

def compute_reproj_error(obj_pts, img_pts, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    return np.mean(np.linalg.norm(proj - img_pts.reshape(-1, 2), axis=1))

def solve_pnp_with_best_obj_order(corners, K, dist, tag_size):
    """
    Try multiple TL/TR/BR/BL orderings; pick lowest reprojection error, prefer positive Z.
    Returns: obj_pts, rvec, tvec, err_px, order_label
    """
    half = tag_size / 2.0
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

    for label, obj in candidates:
        obj_pts = np.array(obj, dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        err = compute_reproj_error(obj_pts, corners, rvec, tvec, K, dist)
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

# ---- Z estimation (project PCD to image to sample a window) ----
def project_points_cv(P_cv, K, dist):
    if P_cv.shape[0] == 0:
        return np.empty((0,2)), np.zeros((0,), dtype=bool)
    rvec = np.zeros((3,1), dtype=np.float64)
    tvec = np.zeros((3,1), dtype=np.float64)
    pts = P_cv.astype(np.float64).reshape(-1,1,3)
    uv, _ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    uv = uv.reshape(-1, 2)
    valid = P_cv[:,2] > 0
    return uv, valid

def estimate_z_from_pcd_around_pixel(pcd, K, dist, u0, v0, S, win_px=15, max_pts=50000):
    P = np.asarray(pcd.points)
    if P.shape[0] == 0:
        return None
    if P.shape[0] > max_pts:
        idx = np.random.choice(P.shape[0], max_pts, replace=False)
        P = P[idx]

    # Open3D → OpenCV coords: p_cv = S @ p_o3d
    P_cv = (S @ P.T).T
    uv, valid = project_points_cv(P_cv, K, dist)
    if uv.shape[0] == 0:
        return None

    u, v = uv[:,0], uv[:,1]
    Z = P_cv[:,2]
    sel = valid & (np.abs(u - u0) <= win_px) & (np.abs(v - v0) <= win_px)
    if not np.any(sel):
        sel = valid & (np.abs(u - u0) <= 2*win_px) & (np.abs(v - v0) <= 2*win_px)
    if not np.any(sel):
        return None
    return float(np.median(Z[sel]))

# ---- CAD helpers ----
def euler_xyz_deg_to_R(euler_deg):
    rx, ry, rz = [math.radians(v) for v in euler_deg]
    Rx = np.array([[1,0,0],[0, math.cos(rx),-math.sin(rx)],[0, math.sin(rx), math.cos(rx)]], dtype=np.float64)
    Ry = np.array([[ math.cos(ry),0, math.sin(ry)],[0,1,0],[-math.sin(ry),0, math.cos(ry)]], dtype=np.float64)
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz), math.cos(rz),0],[0,0,1]], dtype=np.float64)
    return Rz @ Ry @ Rx  # XYZ intrinsic

def make_se3(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T

def units_to_meters_factor(units):
    u = units.lower().strip()
    if u == "m":  return 1.0
    if u == "cm": return 0.01
    if u == "mm": return 0.001
    raise ValueError(f"Unknown CAD units: {units}")

def load_and_prepare_cad(path, units="m", uniform_scale=1.0, pre_rot_euler_deg=(0,0,0),
                         origin_offset_local=np.zeros(3), colorize=True, solid_color=(0.8,0.8,0.2)):
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise FileNotFoundError(f"Failed to read CAD mesh or mesh is empty: {path}")
    mesh.compute_vertex_normals()

    # Convert units → meters and apply optional uniform scale
    s_units = units_to_meters_factor(units)
    total_scale = float(s_units) * float(uniform_scale)
    if total_scale != 1.0:
        mesh.scale(total_scale, center=np.zeros(3))

    # CAD-local offset (before rotation)
    if np.linalg.norm(origin_offset_local) > 0:
        mesh.translate(origin_offset_local * s_units * uniform_scale)

    # Optional CAD-local pre-rotation
    R_pre = euler_xyz_deg_to_R(pre_rot_euler_deg)
    if not np.allclose(R_pre, np.eye(3), atol=1e-12):
        mesh.rotate(R_pre, center=(0,0,0))

    # Color if needed
    if colorize and (not mesh.has_vertex_colors()) and (not mesh.has_triangle_normals()):
        mesh.paint_uniform_color(solid_color)

    return mesh

# ======================
# Main
# ======================
def main():
    # Load image (color frame)
    bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    # Load COLOR intrinsics
    K, dist, _ = load_intrinsics(INTRINSICS_JSON)

    # Detect AprilTag on color image
    tag = detect_best_tag(bgr)
    corners = tag.corners.astype(np.float64)

    print("Corners (px):")
    for i, (x, y) in enumerate(corners):
        print(f"  {i}: ({x:.2f}, {y:.2f})")

    # Solve pose in OpenCV COLOR camera frame
    obj_pts, rvec, tvec, err_px, order_label = solve_pnp_with_best_obj_order(corners, K, dist, TAG_SIZE)
    R_cv, _ = cv2.Rodrigues(rvec)
    t_cv = tvec.reshape(3)

    # Project tag origin → pixel (for Z sampling window)
    origin3D = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    proj_center, _ = cv2.projectPoints(origin3D, rvec, tvec, K, dist)
    u_pred, v_pred = proj_center.reshape(2)

    # Load point cloud (already aligned to COLOR frame)
    pcd = o3d.io.read_point_cloud(PLY_PATH)

    # Basis change: OpenCV → Open3D
    S = np.diag([1.0, -1.0, -1.0])

    # Read median depth near the predicted pixel from the PCD
    z_pcd = estimate_z_from_pcd_around_pixel(pcd, K, dist, u_pred, v_pred, S, win_px=15)

    # Tag pose in Open3D basis
    R_o3d = S @ R_cv @ S
    t_o3d = S @ t_cv

    print(f"Predicted tag center (image px): ({u_pred:.2f}, {v_pred:.2f})")
    print(f"Reprojection error (mean L2, px): {err_px:.3f}")
    print(f"Chosen 3D corner order: {order_label} (ensures positive Z)")
    print(f"Tag position BEFORE scale (Open3D, m): [{t_o3d[0]:.6f}, {t_o3d[1]:.6f}, {t_o3d[2]:.6f}]")

    # Optional Z-only scale to match depth from PCD
    if z_pcd is not None and t_cv[2] > 0:
        s = z_pcd / float(t_cv[2])
        print(f"Depth at tag center from PCD (m): {z_pcd:.4f}")
        print(f"PnP tvec.z (m): {float(t_cv[2]):.4f}")
        print(f"Computed scale s = z_pcd / t_pnp,z: {s:.6f}")
        # Z-only scaling
        t_cv[2] *= s
        t_o3d = S @ t_cv
        print(f"Tag position AFTER Z-only scale (Open3D, m): [{t_o3d[0]:.6f}, {t_o3d[1]:.6f}, {t_o3d[2]:.6f}]")
    else:
        print("Warning: could not estimate z_pcd (empty selection) or invalid tvec.z. "
              "Verify the PLY is color-aligned and the sampling window covers points.")

    # === Build tag pose SE(3) in Open3D world ===
    T_tag_o3d = make_se3(R_o3d, t_o3d)

    # === Load & prepare CAD mesh, then place it at the tag ===
    cad_mesh = None
    if CAD_MODEL_PATH and len(CAD_MODEL_PATH.strip()) > 0:
        cad_mesh = load_and_prepare_cad(
            CAD_MODEL_PATH,
            units=CAD_UNITS,
            uniform_scale=CAD_UNIFORM_SCALE,
            pre_rot_euler_deg=CAD_PRE_ROT_EULER_DEG,
            origin_offset_local=CAD_ORIGIN_OFFSET_LOCAL,
            colorize=COLOR_CAD_IF_UNCOLORED,
            solid_color=CAD_SOLID_COLOR,
        )
        cad_mesh.transform(T_tag_o3d)

    # Visualization sizing
    img_h, img_w = bgr.shape[:2]
    win_w = int(img_w * VIEW_SCALE)
    win_h = int(img_h * VIEW_SCALE)

    # Axis size heuristics
    extents = []
    if pcd.has_points() and len(pcd.points):
        extents.append(pcd.get_axis_aligned_bounding_box().get_extent())
    if cad_mesh is not None and not cad_mesh.is_empty():
        extents.append(cad_mesh.get_axis_aligned_bounding_box().get_extent())
    extent_vec = np.max(np.vstack(extents), axis=0) if extents else np.array([1.0, 1.0, 1.0], dtype=np.float64)
    extent = float(np.linalg.norm(extent_vec))

    def axis_size(base, min_sz=0.02, max_sz=0.10):
        return float(np.clip(base, min_sz, max_sz))

    size_cam = axis_size(extent * 0.010)
    size_tag = axis_size(extent * 0.015)

    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size_cam, origin=[0, 0, 0])
    tag_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size_tag, origin=t_o3d.tolist())
    tag_frame.rotate(R_o3d, center=t_o3d.tolist())

    geoms = [cam_frame, tag_frame]
    if pcd is not None and pcd.has_points():
        geoms.insert(0, pcd)
    if cad_mesh is not None:
        geoms.append(cad_mesh)

    print("\n=== Scene Summary ===")
    print(f"PCD points: {len(pcd.points) if pcd is not None else 0}")
    if cad_mesh is not None:
        v = np.asarray(cad_mesh.vertices)
        print(f"CAD verts: {len(v)} | Units: {CAD_UNITS} | Scale: {CAD_UNIFORM_SCALE}")
    print(f"Axis sizes → camera: {size_cam:.4f} m, tag: {size_tag:.4f} m\n")

    o3d.visualization.draw_geometries(
        geoms,
        width=win_w,
        height=win_h
    )

if __name__ == "__main__":
    main()
