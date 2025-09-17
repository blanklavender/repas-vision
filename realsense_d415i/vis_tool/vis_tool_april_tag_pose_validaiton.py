import json
import cv2
import numpy as np
import open3d as o3d
from pupil_apriltags import Detector

# ======================
# Paths & config
# ======================

# POSE 1
# PLY_PATH   = r"../02_ply_generation/ply_generation_scripts/not_aligned_outputs/pose 1/out_20250808_142429.ply"
# IMAGE_PATH = r"../02_ply_generation/ply_generation_scripts/not_aligned_outputs/pose 1/rgb_20250808_142429.png"

# POSE 1 ALIGNED
PLY_PATH   = r"../02_ply_generation/ply_generation_scripts/aligned_outputs/pose 1/pc_20250808_142303.ply"
IMAGE_PATH = r"../02_ply_generation/ply_generation_scripts/aligned_outputs/pose 1/rgb_20250808_142303.png"

# PLYs from the capture code are in the DEPTH frame → set this False to force transform
PLY_IN_COLOR_FRAME = True

# JSONs
EXTRINSICS_JSON  = r"../00_april_tag_detection_callibration\depth_to_color_extrinsics.json"
INTRINSICS_JSON  = r"../00_april_tag_detection_callibration/color_intrinsics_1280_720_copy.json"

# Tag + detection
TAG_SIZE = 0.0303  # meters (outer black square)
MARGIN_THRESH = 10.0

# Viewer scale (UI only; does not affect geometry)
VIEW_SCALE = 1.0


# ======================
# Helpers
# ======================

def load_intrinsics(json_path):
    with open(json_path, "r") as f:
        d = json.load(f)
    K = np.array([
        [d["fx"], 0.0,      d["ppx"]],
        [0.0,     d["fy"],  d["ppy"]],
        [0.0,     0.0,      1.0    ]
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

def read_depth_to_color_extrinsics(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    R_dc = np.array(data["R_dc"], dtype=np.float64).reshape(3,3)
    t_dc = np.array(data["t_dc"], dtype=np.float64).reshape(3,)
    return R_dc, t_dc

def solve_pnp_with_best_obj_order(corners, K, dist, tag_size):
    """
    Keep 'corners' as returned by the detector.
    Try several TL/TR/BR/BL orderings; pick the one with lowest reproj error,
    preferring positive Z.
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

# ---- New helpers for Z estimation ----

def project_points_cv(P_cv, K, dist):
    """
    Project 3D points already in camera (OpenCV) coords with intrinsics/distortion.
    Returns (u,v) and a valid mask (Z>0).
    """
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
    """
    pcd: Open3D point cloud already in the COLOR frame (Open3D coords).
    Convert to OpenCV camera coords with S and project using K/dist.
    Returns median Z (m) of points whose (u,v) fall within a window around (u0,v0).
    """
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


# ======================
# Main
# ======================

def main():
    # Load image
    bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    # Load intrinsics
    K, dist, _ = load_intrinsics(INTRINSICS_JSON)

    # Detect AprilTag
    tag = detect_best_tag(bgr)
    corners = tag.corners.astype(np.float64)

    print("Corners (px):")
    for i, (x, y) in enumerate(corners):
        print(f"  {i}: ({x:.2f}, {y:.2f})")

    # Solve pose
    obj_pts, rvec, tvec, err_px, order_label = solve_pnp_with_best_obj_order(corners, K, dist, TAG_SIZE)
    R_cv, _ = cv2.Rodrigues(rvec)
    t_cv = tvec.reshape(3)  # OpenCV camera coords

    # Project tag origin → pixel
    origin3D = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    proj_center, _ = cv2.projectPoints(origin3D, rvec, tvec, K, dist)
    u_pred, v_pred = proj_center.reshape(2)

    # Load point cloud
    pcd = o3d.io.read_point_cloud(PLY_PATH)

    # OpenCV -> Open3D conversion
    S = np.diag([1.0, -1.0, -1.0])

    # Depth->Color extrinsics (OpenCV coords) → Open3D
    R_dc, t_dc = read_depth_to_color_extrinsics(EXTRINSICS_JSON)
    R_dc_o3d = S @ R_dc @ S
    t_dc_o3d = S @ t_dc
    T_dc_o3d = np.eye(4)
    T_dc_o3d[:3, :3] = R_dc_o3d
    T_dc_o3d[:3,  3] = t_dc_o3d

    # STEP 1: Transform PLY → color frame (no double-transform)
    if not PLY_IN_COLOR_FRAME:
        pcd.transform(T_dc_o3d)

    # STEP 2–3: (u_pred, v_pred) already computed; read median depth z_pcd near that pixel
    z_pcd = estimate_z_from_pcd_around_pixel(pcd, K, dist, u_pred, v_pred, S, win_px=15)

    # Tag pose in Open3D space (before scaling)
    R_o3d = S @ R_cv @ S
    t_o3d = S @ t_cv

    print(f"Predicted tag center (image px): ({u_pred:.2f}, {v_pred:.2f})")
    print(f"Reprojection error (mean L2, px): {err_px:.3f}")
    print(f"Chosen 3D corner order: {order_label}  (fix: matched 2D→3D order; ensures positive Z)")
    print(f"Tag position BEFORE scale (Open3D, m): [{t_o3d[0]:.6f}, {t_o3d[1]:.6f}, {t_o3d[2]:.6f}]")

    # STEP 4–5: Compute s and apply ONLY to Z (leave X,Y unchanged)
    if z_pcd is not None and t_cv[2] > 0:
        s = z_pcd / float(t_cv[2])
        print(f"Depth at tag center from PCD (m): {z_pcd:.4f}")
        print(f"PnP tvec.z (m):                  {float(t_cv[2]):.4f}")
        print(f"Computed scale s = z_pcd / t_pnp,z: {s:.6f}")

        # Z-only scaling (requested)
        t_cv[2] *= s

        # Recompute Open3D translation after Z-only scaling
        t_o3d = S @ t_cv
        print(f"Tag position AFTER  Z-only scale (Open3D, m): [{t_o3d[0]:.6f}, {t_o3d[1]:.6f}, {t_o3d[2]:.6f}]")
    else:
        print("Warning: could not estimate z_pcd (empty selection) or invalid tvec.z. "
              "Check PLY_IN_COLOR_FRAME/extrinsics and pixel window.")

    # Visualization
    img_h, img_w = bgr.shape[:2]
    win_w = int(img_w * VIEW_SCALE)
    win_h = int(img_h * VIEW_SCALE)

    bbox = pcd.get_axis_aligned_bounding_box()
    extent_vec = bbox.get_extent() if len(pcd.points) else np.array([1.0, 1.0, 1.0])
    extent = float(np.linalg.norm(extent_vec))

    def axis_size(base, min_sz=0.02, max_sz=0.08):
        return float(np.clip(base, min_sz, max_sz))

    size_cam = axis_size(extent * 0.010)
    size_tag = axis_size(extent * 0.015)

    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size_cam, origin=[0, 0, 0])
    tag_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size_tag, origin=t_o3d.tolist())
    tag_frame.rotate(R_o3d, center=t_o3d.tolist())

    o3d.visualization.draw_geometries(
        [pcd, cam_frame, tag_frame],
        width=win_w, height=win_h
    )

if __name__ == "__main__":
    main()
