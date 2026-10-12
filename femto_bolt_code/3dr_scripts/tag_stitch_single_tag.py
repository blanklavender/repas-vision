#!/usr/bin/env python3
"""
tag_stitch.py  --  bare-bones single-tag stitching.

Idea
----
1. Follow ONE tag (TAG_ID) through the shots in ANGLES.
2. In every shot, lift its 4 corners to 3D (camera frame of that shot).
3. Register each shot's 4 corners onto the reference shot's 4 corners.
   No turntable angle / axis is used -- the transform comes only from the
   4 points in space.  The detector returns corners in a fixed order, so the
   correspondences are known and point-to-point ICP collapses to a single
   closed-form Kabsch step (that is what ICP converges to).
4. Apply that transform to each shot's point cloud and show them in a viewer
   where every shot can be toggled on/off.

FIT_MODE = "rigid"        rotation + translation (the corners overlap)
           "translation"  translation only (corner centroids overlap, no rotation)

Filenames:  color_003_ang030.png  depth_003_ang030.png  cloud_003_ang030.ply
"""

import configparser
import glob
import os
import re

import cv2
import numpy as np
import open3d as o3d
from pupil_apriltags import Detector

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
INPUT_DIR = r"C:\Users\quartz\02_projects\repas-vision\captures\femto_bolt\flora_unit\yellow_pot_09-30_deg_10_speed_5"
COLOR_DIR = os.path.join(INPUT_DIR, "")
DEPTH_DIR = os.path.join(INPUT_DIR, "")
CLOUD_DIR = os.path.join(INPUT_DIR, "")

INTRINSICS_FILE = (r"C:\Users\quartz\02_projects\repas-vision\femto_bolt_code\scripts\calibration_parameters\CameraParam_Femto_BoltCL8956200ZM_Color1280x720_Depth640x576.ini")

TAG_ID = 16
TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.024
ANGLES = list(range(0, 80, 10))     # 0..70 deg
REF_ANGLE = 0

FIT_MODE = "rigid"                  # "rigid" | "translation"

DEPTH_SCALE_M = 0.001
DEPTH_MIN_M, DEPTH_MAX_M = 0.15, 1.50
PLANE_ERODE_PX = 2                  # stay off the black/white tag border
PLANE_INLIER_M = 0.0015

VIEW_VOXEL_M = 0.0                  # display downsample, raise if the viewer drags
CORNER_SPHERE_M = 0.0015
POINT_SIZE = 2

PALETTE = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.70, 0.20], [0.15, 0.35, 0.95], [1.00, 0.75, 0.00],
    [0.85, 0.15, 0.85], [0.00, 0.80, 0.80], [1.00, 0.45, 0.00], [0.55, 0.85, 0.55],
])
NAME_RE = re.compile(r"_(\d+)_ang(\d+)$")


# ---------------------------------------------------------------------------
# files + intrinsics
# ---------------------------------------------------------------------------
def find_by_angle(dirpath, pattern):
    out = {}
    for p in glob.glob(os.path.join(dirpath, pattern)):
        m = NAME_RE.search(os.path.splitext(os.path.basename(p))[0])
        if m:
            out[int(m.group(2))] = p
    return out


def load_intrinsics(path, w, h):
    cfg = configparser.ConfigParser()
    with open(path) as f:
        cfg.read_string(f.read())
    i, d = cfg["ColorIntrinsic"], cfg["ColorDistortion"]
    sx, sy = w / int(i["width"]), h / int(i["height"])
    K = np.array([[float(i["fx"]) * sx, 0.0, float(i["cx"]) * sx],
                  [0.0, float(i["fy"]) * sy, float(i["cy"]) * sy],
                  [0.0, 0.0, 1.0]])
    dist = np.array([float(d[k]) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")])
    return K, dist


def undistort(uv, K, dist):
    """pixels (N,2) -> normalised rays (N,3) with z = 1."""
    xy = cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    return np.column_stack([xy, np.ones(len(xy))])


# ---------------------------------------------------------------------------
# tag corners in 3D: fit a plane to the depth inside the tag, cast corner rays onto it
# ---------------------------------------------------------------------------
def fit_plane(pts, iters=150, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((pts - s[0]) @ n) < PLANE_INLIER_M
        if best is None or inl.sum() > best.sum():
            best = inl
    P = pts[best]
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][-1]
    return n, -n @ c


def tag_corners_3d(color, depth, K, dist, detector):
    dets = [d for d in detector.detect(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
            if d.tag_id == TAG_ID]
    if not dets:
        return None, None, "tag not detected"
    cpx = np.asarray(dets[0].corners, np.float64)

    mask = np.zeros(depth.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(cpx).astype(np.int32), 1)
    mask = cv2.erode(mask, np.ones((2 * PLANE_ERODE_PX + 1,) * 2, np.uint8))
    v, u = np.nonzero(mask)
    z = depth[v, u].astype(np.float64) * DEPTH_SCALE_M
    ok = (z > DEPTH_MIN_M) & (z < DEPTH_MAX_M)
    if ok.sum() < 30:
        return None, cpx, f"only {ok.sum()} depth px on tag"
    pts = undistort(np.column_stack([u[ok], v[ok]]), K, dist) * z[ok, None]

    n, d = fit_plane(pts)
    rays = undistort(cpx, K, dist)
    return rays * (-d / (rays @ n))[:, None], cpx, None


# ---------------------------------------------------------------------------
# 4-point registration
# ---------------------------------------------------------------------------
def fit_transform(src, dst):
    """src, dst: (4,3) corresponding corners.  Returns 4x4 T with T*src ~= dst."""
    ps, pd = src.mean(0), dst.mean(0)
    R = np.eye(3)
    if FIT_MODE == "rigid":
        U, _, Vt = np.linalg.svd((src - ps).T @ (dst - pd))
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, pd - R @ ps
    return T


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------------
# viewer
# ---------------------------------------------------------------------------
def corner_markers(C, color):
    m = o3d.geometry.TriangleMesh()
    for p in C:
        s = o3d.geometry.TriangleMesh.create_sphere(CORNER_SPHERE_M)
        s.translate(p)
        m += s
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(C),
                              o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]]))
    m.paint_uniform_color(color)
    ls.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m, ls


def viewer(shots, center):
    angles = [s["angle"] for s in shots]
    items = []
    for s in shots:
        a, col = s["angle"], s["color"]
        items.append(dict(name=f"cloud {a:03d}", geometry=s["cloud"], group="clouds",
                          is_visible=True))
        m, ls = corner_markers(s["C_world"], col)
        items.append(dict(name=f"corners {a:03d}", geometry=m, group="corners", is_visible=True))
        items.append(dict(name=f"outline {a:03d}", geometry=ls, group="corners", is_visible=True))

    state = {"i": 0}

    def show(vis, sel, clouds=True):
        for a in angles:
            vis.show_geometry(f"cloud {a:03d}", clouds and a in sel)
            vis.show_geometry(f"corners {a:03d}", a in sel)
            vis.show_geometry(f"outline {a:03d}", a in sel)

    def step(vis, d):
        state["i"] = (state["i"] + d) % len(angles)
        a = angles[state["i"]]
        show(vis, {a})
        print(f"[VIEW] solo {a:03d}")

    def ref_plus(vis):
        state["i"] = (state["i"] + 1) % len(angles)
        sel = {REF_ANGLE, angles[state["i"]]}
        show(vis, sel)
        print(f"[VIEW] {sorted(sel)}")

    actions = [("All", lambda v: show(v, set(angles))),
               ("Corners only", lambda v: show(v, set(angles), clouds=False)),
               ("Ref + next", ref_plus),
               ("Solo next", lambda v: step(v, +1)),
               ("Solo prev", lambda v: step(v, -1))]

    print("[VIEW] tick individual shots in the scene list, or use the action buttons")
    o3d.visualization.draw(items, title=f"tag {TAG_ID} stitch", show_ui=True,
                           point_size=POINT_SIZE, lookat=center.astype(np.float32),
                           eye=np.zeros(3, np.float32), up=np.array([0, -1, 0], np.float32),
                           show_skybox=False, actions=actions)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    colors = find_by_angle(COLOR_DIR, "color_*.png")
    depths = find_by_angle(DEPTH_DIR, "depth_*.png")
    clouds = find_by_angle(CLOUD_DIR, "cloud_*.ply")

    if REF_ANGLE not in colors:
        raise SystemExit(f"no color_*_ang{REF_ANGLE:03d}.png in {COLOR_DIR} "
                         f"({len(colors)} colour files found)")
    probe = cv2.imread(colors[REF_ANGLE])
    h, w = probe.shape[:2]
    K, dist = load_intrinsics(INTRINSICS_FILE, w, h)
    detector = Detector(families=TAG_FAMILY, quad_decimate=1.0, refine_edges=True)

    # ---- 1+2: corners per shot ------------------------------------------
    shots = []
    for a in ANGLES:
        if a not in colors or a not in depths:
            print(f"[SKIP] {a:03d}: missing colour/depth")
            continue
        color = cv2.imread(colors[a])
        depth = cv2.imread(depths[a], cv2.IMREAD_UNCHANGED)
        if depth.shape[:2] != (h, w):
            raise SystemExit(f"depth {depth.shape[:2]} not aligned to colour {(h, w)}")
        C, cpx, why = tag_corners_3d(color, depth, K, dist, detector)
        if C is None:
            print(f"[SKIP] {a:03d}: {why}")
            continue
        sides = np.linalg.norm(C - np.roll(C, -1, 0), axis=1) * 1000
        print(f"[TAG ] {a:03d}  sides {np.round(sides, 1)} mm  "
              f"(expected {TAG_SIZE_M * 1000:.1f})")
        shots.append(dict(angle=a, C=C))

    ref = next((s for s in shots if s["angle"] == REF_ANGLE), None)
    if ref is None:
        raise SystemExit(f"tag {TAG_ID} not usable in reference shot {REF_ANGLE:03d}")

    # ---- 3: register each shot's 4 corners onto the reference's --------
    print(f"\n  angle   fit_rot_deg   corner_rms_mm   (FIT_MODE = {FIT_MODE})")
    for i, s in enumerate(shots):
        s["T"] = fit_transform(s["C"], ref["C"])
        s["C_world"] = apply(s["T"], s["C"])
        rms = np.sqrt(np.mean(np.sum((s["C_world"] - ref["C"]) ** 2, 1))) * 1000
        rot = np.degrees(np.linalg.norm(cv2.Rodrigues(s["T"][:3, :3])[0]))
        s["color"] = PALETTE[i % len(PALETTE)]
        print(f"   {s['angle']:03d}   {rot:9.2f}     {rms:9.3f}")

    # ---- 4: move the clouds ---------------------------------------------
    for s in shots:
        pcd = o3d.io.read_point_cloud(clouds[s["angle"]])
        if VIEW_VOXEL_M > 0:
            pcd = pcd.voxel_down_sample(VIEW_VOXEL_M)
        s["cloud"] = pcd.transform(s["T"])
        print(f"[PCD ] {s['angle']:03d}  {len(pcd.points):,} pts")

    viewer(shots, ref["C"].mean(0))


if __name__ == "__main__":
    main()
