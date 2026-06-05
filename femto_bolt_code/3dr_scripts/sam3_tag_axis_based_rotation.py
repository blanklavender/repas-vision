#!/usr/bin/env python3
"""
turntable_align.py  --  turntable axis fit + point-cloud alignment, one script.

Axis estimation, in order of increasing trust
---------------------------------------------
1. SEED    : mean of the AprilTag face "up" vectors.
2. BASELINE: two independent ring circles, axis = bottom centre -> top centre.
             Kept only as a comparison -- the ~4.5 cm baseline makes it fragile.
3. JOINT   : one concentric model over every tag observation.  Shared axis and
             shared centre; per-tag radius and per-tag height.
4. CLOUDS  : 4-dof refinement (2 tilt + 2 lateral) against the point clouds.

Filenames:  color_012_ang120.png  depth_005_ang050.png  cloud_000_ang000.ply
"""


import configparser
import glob
import json
import os
import re

import cv2
import numpy as np
import open3d as o3d
from pupil_apriltags import Detector

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
INPUT_DIR = "../../flora_turntable/captures/yellow_pot_08-25_deg_10_speed_5"

COLOR_DIR = os.path.join(INPUT_DIR, "rgb")
DEPTH_DIR = os.path.join(INPUT_DIR, "depth")
CLOUD_DIR = os.path.join(INPUT_DIR, "pointcloud")

OUT_SUBDIR = "aligned"
DEBUG_SUBDIR = "debug"

INTRINSICS_FILE = ("../scripts/calibration_parameters/"
                   "CameraParam_Femto_BoltCL8956200ZM_Color1280x720_Depth640x576.ini")
INTRINSIC_SECTION = "ColorIntrinsic"
DISTORTION_SECTION = "ColorDistortion"
UNDISTORT_RAYS = True

COLOR_PREFIX = "color_"
DEPTH_PREFIX = "depth_"
CLOUD_PREFIX = "cloud_"
COLOR_EXT = ".png"
CLOUD_EXT = ".ply"
DEPTH_EXTS = (".npy", ".png")

# ---------------------------------------------------------------------------
# angles
# ---------------------------------------------------------------------------
ANGLE_INCREMENT_DEG = 10.0
FORCE_INCREMENT_ANGLE = False
ANGLE_SIGN = "auto"
REF_ANGLE_DEG = 0

# ---------------------------------------------------------------------------
# depth / tags
# ---------------------------------------------------------------------------
DEPTH_SCALE_M = 0.001
DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 1.50

TAG_FAMILY = "tag36h11"
TAG_SIZE_M = 0.028
DEPTH_WIN_PX = 5
MAX_TILT_DEG = 25.0

TOP_RING_IDS = (12, 14, 16, 18, 20, 22)
BOTTOM_RING_IDS = (13, 15, 17, 19, 21, 23)

# ---------------------------------------------------------------------------
# joint concentric fit
# ---------------------------------------------------------------------------
RADIUS_MODE = "per_tag"    # "per_tag" | "per_ring" | "shared"
HEIGHT_WEIGHT = 1.0
RADIUS_WEIGHT = 1.0
JOINT_PASSES = 3
JOINT_ITERS = 600

BASELINE_COMPARE = True
TAG_AXIS_SOURCE = "joint"  # "joint" | "baseline" | "seed"

# ---------------------------------------------------------------------------
# cloud-based axis refinement (4 dof: 2 tilt + 2 lateral)
# ---------------------------------------------------------------------------
REFINE_AXIS_ON_CLOUDS = True
REFINE_PAIR_STEPS_DEG = (30, 90)
REFINE_MAX_PTS = 2000
REFINE_ITERS = 250
REFINE_TILT_STEP_DEG = 0.30
REFINE_SHIFT_STEP_MM = 1.5

# ---------------------------------------------------------------------------
# depth sanity check + bias diagnostic
# ---------------------------------------------------------------------------
DEBUG_TAG_ID = 15
DEBUG_Z_OFFSETS_M = (0.001, 0.045)
OBLIQUITY_BINS_DEG = (0, 15, 30, 45, 60, 90)

# ---------------------------------------------------------------------------
# point clouds
# ---------------------------------------------------------------------------
VOXEL_M = 0.0
LOAD_ANGLES = None
MERGED_VOXEL_M = 0.0
POINT_SIZE = 3
AXIS_HALF_LEN_M = 0.15
FRAME_AXIS_SIZE_M = 0.02
SAVE_MERGED = True
SHOW_VIEWER = True

RESIDUAL_MODE = "both"

# ---------------------------------------------------------------------------
# viewer: one cloud per axis approach, and per-angle selection
# ---------------------------------------------------------------------------
VIEW_APPROACHES = ("seed", "baseline", "joint", "refined")
VIEW_DEFAULT_APPROACH = "refined"
VIEW_ANGLES = None          # e.g. (0, 90, 180, 270); None = every loaded angle
VIEW_PER_ANGLE = True       # one toggleable item per angle, per approach
VIEW_TRUE_COLOR = True      # keep the captured RGB (False = per-frame palette)
VIEW_VOXEL_M = 0.0          # display-only downsample; raise if the viewer drags
VIEW_STEP_DEG = 90          # spacing used by the "Every N deg" button
VIEW_SHOW_FRAMES = False    # per-shot coordinate triads

WRITE_DEBUG_IMAGES = True
AXIS_DRAW_HALF_LEN_M = 0.12
AXIS_SAMPLES = 40
DRAW_THICK = 2

PALETTE = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.70, 0.20], [0.15, 0.35, 0.95], [1.00, 0.75, 0.00],
    [0.85, 0.15, 0.85], [0.00, 0.80, 0.80], [1.00, 0.45, 0.00], [0.55, 0.85, 0.55],
])

NAME_RE = re.compile(r"^(?P<idx>\d+)(?:_ang(?P<ang>\d+))?$")


# ===========================================================================
# small numeric helpers
# ===========================================================================
def unit(v):
    return np.asarray(v, dtype=np.float64) / np.linalg.norm(v)


def basis_from_axis(u):
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(u @ ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = unit(np.cross(u, ref))
    e2 = unit(np.cross(u, e1))
    return e1, e2


def axis_angle_matrix(axis, theta):
    a = unit(axis)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotation_about(P, u_hat, deg):
    R = axis_angle_matrix(u_hat, np.deg2rad(deg))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = P - R @ P
    return T


def nelder_mead(fn, x0, step, iters=400, tol=1e-10):
    x0 = np.asarray(x0, dtype=np.float64)
    pts = [x0.copy()]
    for i in range(len(x0)):
        p = x0.copy()
        p[i] += step[i]
        pts.append(p)
    f = [fn(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(f)
        pts = [pts[i] for i in order]
        f = [f[i] for i in order]
        if abs(f[-1] - f[0]) <= tol * (abs(f[0]) + tol):
            break
        cen = np.mean(pts[:-1], axis=0)
        xr = cen + (cen - pts[-1])
        fr = fn(xr)
        if fr < f[0]:
            xe = cen + 2.0 * (cen - pts[-1])
            fe = fn(xe)
            pts[-1], f[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < f[-2]:
            pts[-1], f[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen)
            fc = fn(xc)
            if fc < f[-1]:
                pts[-1], f[-1] = xc, fc
            else:
                for i in range(1, len(pts)):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0])
                    f[i] = fn(pts[i])
    order = np.argsort(f)
    return pts[order[0]], f[order[0]]


# ===========================================================================
# intrinsics / projection
# ===========================================================================
def load_intrinsics(path):
    cfg = configparser.ConfigParser()
    with open(path) as f:
        cfg.read_string(f.read())
    i, d = cfg[INTRINSIC_SECTION], cfg[DISTORTION_SECTION]
    K = np.array([[float(i["fx"]), 0.0, float(i["cx"])],
                  [0.0, float(i["fy"]), float(i["cy"])],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([float(d["k1"]), float(d["k2"]), float(d["p1"]), float(d["p2"]),
                     float(d["k3"]), float(d["k4"]), float(d["k5"]), float(d["k6"])],
                    dtype=np.float64)
    return K, dist, int(i["width"]), int(i["height"])


def scale_K(K, src_wh, dst_wh):
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    out = K.copy()
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def ray_map(Ks, dist, w, h):
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    if UNDISTORT_RAYS:
        pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float64)
        norm = cv2.undistortPoints(pts, Ks, dist).reshape(h, w, 2)
        return norm[..., 0].astype(np.float32), norm[..., 1].astype(np.float32)
    return (((xs - Ks[0, 2]) / Ks[0, 0]).astype(np.float32),
            ((ys - Ks[1, 2]) / Ks[1, 1]).astype(np.float32))


def project(pts3d, Ks, dist):
    p = np.asarray(pts3d, dtype=np.float64).reshape(-1, 1, 3)
    uv, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), Ks, dist)
    return uv.reshape(-1, 2)


def median_depth(depth, u, v, win):
    h, w = depth.shape[:2]
    patch = depth[max(0, v - win):min(h, v + win + 1), max(0, u - win):min(w, u + win + 1)]
    vals = patch[patch > 0].astype(np.float32)
    return float(np.median(vals)) * DEPTH_SCALE_M if vals.size else 0.0


def unproject(xn, yn, u, v, z):
    return np.array([xn[v, u] * z, yn[v, u] * z, z], dtype=np.float64)


def kabsch(P, Q):
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


# ===========================================================================
# tag detection
# ===========================================================================
def detect_tags(color_bgr, depth, xn, yn, detector, keep_ids):
    s = TAG_SIZE_M / 2.0
    canon = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]])
    up_cam = np.array([0.0, -1.0, 0.0])
    out = []
    for det in detector.detect(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)):
        if keep_ids is not None and det.tag_id not in keep_ids:
            continue
        meas, ok = [], True
        for (u, v) in det.corners:
            u, v = int(round(u)), int(round(v))
            z = median_depth(depth, u, v, DEPTH_WIN_PX)
            if not (DEPTH_MIN_M < z < DEPTH_MAX_M):
                ok = False
                break
            meas.append(unproject(xn, yn, u, v, z))
        if not ok:
            continue
        meas = np.asarray(meas)
        R, t = kabsch(canon, meas)
        fit_rms = float(np.sqrt(np.mean(np.sum((canon @ R.T + t - meas) ** 2, axis=1))))
        cu, cv_ = int(round(det.center[0])), int(round(det.center[1]))
        zc = median_depth(depth, cu, cv_, DEPTH_WIN_PX)
        if not (DEPTH_MIN_M < zc < DEPTH_MAX_M):
            continue
        ax0, ax1 = R[:, 0], R[:, 1]
        u_hat = ax1 if abs(ax1 @ up_cam) >= abs(ax0 @ up_cam) else ax0
        if u_hat @ up_cam < 0:
            u_hat = -u_hat
        u_hat = unit(u_hat)
        centre = unproject(xn, yn, cu, cv_, zc)
        normal = unit(R[:, 2])
        if normal @ centre > 0:
            normal = -normal
        obliq = float(np.degrees(np.arccos(np.clip(abs(normal @ unit(centre)), -1, 1))))
        tilt = float(np.degrees(np.arccos(np.clip(u_hat @ up_cam, -1, 1))))
        out.append(dict(id=int(det.tag_id), center=centre, u=u_hat, n=normal,
                        obliq=obliq, tilt=tilt, rms=fit_rms,
                        corners=np.asarray(det.corners)))
    return out


# ===========================================================================
# axis estimator 2: two independent ring circles
# ===========================================================================
def fit_shared_circle(pts2d, ids):
    uid = sorted(set(ids))
    idx = {k: i for i, k in enumerate(uid)}
    n = len(pts2d)
    A = np.zeros((n, 2 + len(uid)))
    b = np.zeros(n)
    for i, (p, k) in enumerate(zip(pts2d, ids)):
        A[i, 0], A[i, 1] = 2.0 * p[0], 2.0 * p[1]
        A[i, 2 + idx[k]] = 1.0
        b[i] = p @ p
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    radii = {k: float(np.sqrt(max(sol[2 + idx[k]] + c @ c, 0.0))) for k in uid}
    resid = A @ sol - b
    r_mean = float(np.mean(list(radii.values()))) or 1.0
    geo_rms = float(np.sqrt(np.mean(resid ** 2))) / (2.0 * r_mean)
    return c, radii, geo_rms


def fit_ring(tags, u, e1, e2, label):
    pts2d = np.array([[t["center"] @ e1, t["center"] @ e2] for t in tags])
    ids = [t["id"] for t in tags]
    if len(pts2d) < 2 + len(set(ids)) + 1:
        raise SystemExit(f"{label} ring: too few observations")
    c2, radii, rms = fit_shared_circle(pts2d, ids)
    height = float(np.mean([t["center"] @ u for t in tags]))
    return dict(c2=c2, C3=c2[0] * e1 + c2[1] * e2 + height * u, radii=radii,
                rms=rms, height=height, n=len(pts2d), ids=sorted(set(ids)))


def solve_two_ring_axis(top_tags, bot_tags, u_seed, iters=15, tol=1e-4):
    u = unit(u_seed)
    for _ in range(iters):
        e1, e2 = basis_from_axis(u)
        top = fit_ring(top_tags, u, e1, e2, "top")
        bot = fit_ring(bot_tags, u, e1, e2, "bottom")
        v = top["C3"] - bot["C3"]
        sep = float(np.linalg.norm(v))
        if sep < 1e-4:
            raise SystemExit("ring centres coincide")
        u_new = v / sep
        delta = float(np.degrees(np.arccos(np.clip(u @ u_new, -1.0, 1.0))))
        u = u_new
        if delta < tol:
            break
    e1, e2 = basis_from_axis(u)
    return (u, fit_ring(top_tags, u, e1, e2, "top"),
            fit_ring(bot_tags, u, e1, e2, "bottom"))


# ===========================================================================
# axis estimator 3: joint concentric fit
# ===========================================================================
def make_groups(tags):
    ids = np.array([t["id"] for t in tags])
    ring = np.array([0 if t["ring"] == "top" else 1 for t in tags])
    h_groups = [np.nonzero(ids == k)[0] for k in sorted(set(ids.tolist()))]
    if RADIUS_MODE == "per_tag":
        r_groups = h_groups
    elif RADIUS_MODE == "per_ring":
        r_groups = [g for g in (np.nonzero(ring == 0)[0], np.nonzero(ring == 1)[0])
                    if len(g)]
    else:
        r_groups = [np.arange(len(tags))]
    return h_groups, r_groups


def axis_terms(u, P, X):
    d = X - P
    h = d @ u
    r = np.linalg.norm(d - np.outer(h, u), axis=1)
    return h, r


def axis_cost(u, P, X, h_groups, r_groups):
    h, r = axis_terms(u, P, X)
    res = [HEIGHT_WEIGHT * (h[g] - h[g].mean()) for g in h_groups]
    res += [RADIUS_WEIGHT * (r[g] - r[g].mean()) for g in r_groups]
    e = np.concatenate(res)
    return float(np.sqrt(np.mean(e ** 2)))


def solve_joint_axis(tags, u0, P0):
    X = np.stack([t["center"] for t in tags])
    h_groups, r_groups = make_groups(tags)
    u, P = unit(u0), np.array(P0, dtype=np.float64)
    print(f"[JOINT] radius mode '{RADIUS_MODE}'  "
          f"{len(X)} observations, {len(h_groups)} height groups, "
          f"{len(r_groups)} radius group(s)")
    print(f"[JOINT] start cost = {axis_cost(u, P, X, h_groups, r_groups)*1000:.3f} mm")
    for p in range(JOINT_PASSES):
        e1, e2 = basis_from_axis(u)

        def f(x):
            uu = unit(u + np.deg2rad(x[0]) * e1 + np.deg2rad(x[1]) * e2)
            PP = P + x[2] * 1e-3 * e1 + x[3] * 1e-3 * e2
            return axis_cost(uu, PP, X, h_groups, r_groups)

        x, c = nelder_mead(f, [0.0, 0.0, 0.0, 0.0], [0.25, 0.25, 1.0, 1.0],
                           iters=JOINT_ITERS)
        u = unit(u + np.deg2rad(x[0]) * e1 + np.deg2rad(x[1]) * e2)
        P = P + x[2] * 1e-3 * e1 + x[3] * 1e-3 * e2
        print(f"[JOINT] pass {p}: cost = {c*1000:.3f} mm   "
              f"tilt step ({x[0]:+.3f}, {x[1]:+.3f}) deg   "
              f"shift ({x[2]:+.2f}, {x[3]:+.2f}) mm")

    h, r = axis_terms(u, P, X)
    h_res = np.concatenate([h[g] - h[g].mean() for g in h_groups])
    r_res = np.concatenate([r[g] - r[g].mean() for g in r_groups])
    print(f"[JOINT] height scatter RMS = {np.sqrt(np.mean(h_res**2))*1000:.3f} mm "
          f"<-- what fixes the tilt")
    print(f"[JOINT] radius scatter RMS = {np.sqrt(np.mean(r_res**2))*1000:.3f} mm "
          f"<-- what fixes the lateral position")
    return u, P, h, r


# ===========================================================================
# depth-bias diagnostic
# ===========================================================================
def bias_report(tags, h, r):
    ids = np.array([t["id"] for t in tags])
    obl = np.array([t["obliq"] for t in tags])
    dr = np.zeros_like(r)
    dh = np.zeros_like(h)
    for k in sorted(set(ids.tolist())):
        g = np.nonzero(ids == k)[0]
        dr[g] = r[g] - r[g].mean()
        dh[g] = h[g] - h[g].mean()

    print("\n[BIAS] per-tag geometry (radius should be identical on one cylinder)")
    print("       id   n   radius_mm   spread_mm   height_mm   obliquity_deg")
    for k in sorted(set(ids.tolist())):
        g = np.nonzero(ids == k)[0]
        ring = "top" if tags[g[0]]["ring"] == "top" else "bot"
        print(f"   {ring} {k:3d} {len(g):3d}   {r[g].mean()*1000:8.2f}   "
              f"{r[g].std()*1000:9.2f}   {h[g].mean()*1000:9.2f}   "
              f"{obl[g].min():5.1f}-{obl[g].max():5.1f}")

    print("\n[BIAS] radius deviation vs viewing obliquity")
    print("       obliquity_deg     n   mean_dradius_mm   mean_dheight_mm")
    edges = OBLIQUITY_BINS_DEG
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (obl >= lo) & (obl < hi)
        if m.sum() < 3:
            continue
        print(f"       {lo:3d} - {hi:3d}     {m.sum():4d}   "
              f"{dr[m].mean()*1000:+13.3f}   {dh[m].mean()*1000:+13.3f}")
    if obl.max() - obl.min() > 5:
        c = np.corrcoef(obl, dr)[0, 1]
        print(f"       correlation(obliquity, radius deviation) = {c:+.3f}")


# ===========================================================================
# cloud-based 4-dof axis refinement
# ===========================================================================
def refine_axis_on_clouds(samples, angles, u0, P0, undo_sign):
    pairs = []
    aset = set(angles)
    for step in REFINE_PAIR_STEPS_DEG:
        for a in angles:
            b = (a + step) % 360
            if b in aset:
                pairs.append((a, b))
    if not pairs:
        print("[REFINE] no usable frame pairs -- skipped")
        return u0, P0, None, None
    print(f"[REFINE] 4-dof axis fit on {len(pairs)} cloud pairs "
          f"(steps {REFINE_PAIR_STEPS_DEG} deg, <= {REFINE_MAX_PTS} pts/frame)")

    fixed = {a: o3d.geometry.PointCloud(o3d.utility.Vector3dVector(samples[a]))
             for a in angles}

    def cost(u, P):
        tot = []
        for a, b in pairs:
            T = rotation_about(P, u, undo_sign * (b - a))
            X = samples[b] @ T[:3, :3].T + T[:3, 3]
            moved = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
            tot.append(np.median(np.asarray(
                moved.compute_point_cloud_distance(fixed[a]))))
        return float(np.mean(tot))

    u, P = unit(u0), np.array(P0, dtype=np.float64)
    c0 = cost(u, P)
    print(f"[REFINE] start cost = {c0*1000:.3f} mm")
    e1, e2 = basis_from_axis(u)

    def f(x):
        uu = unit(u + np.deg2rad(x[0]) * e1 + np.deg2rad(x[1]) * e2)
        PP = P + x[2] * 1e-3 * e1 + x[3] * 1e-3 * e2
        return cost(uu, PP)

    x, c = nelder_mead(f, [0.0, 0.0, 0.0, 0.0],
                       [REFINE_TILT_STEP_DEG, REFINE_TILT_STEP_DEG,
                        REFINE_SHIFT_STEP_MM, REFINE_SHIFT_STEP_MM],
                       iters=REFINE_ITERS)
    u_new = unit(u + np.deg2rad(x[0]) * e1 + np.deg2rad(x[1]) * e2)
    P_new = P + x[2] * 1e-3 * e1 + x[3] * 1e-3 * e2
    moved_deg = float(np.degrees(np.arccos(np.clip(u @ u_new, -1, 1))))
    print(f"[REFINE] final cost = {c*1000:.3f} mm  "
          f"(was {c0*1000:.3f} mm, {100*(1-c/max(c0,1e-9)):.1f}% better)")
    print(f"[REFINE] axis moved {moved_deg:.3f} deg, "
          f"line shifted {np.hypot(x[2], x[3]):.2f} mm")
    return u_new, P_new, c0, c


# ===========================================================================
# filenames
# ===========================================================================
def parse_suffix(stem, prefix):
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix):]
    m = NAME_RE.match(suffix)
    if not m:
        return None
    idx = int(m.group("idx"))
    ang = m.group("ang")
    angle = (int(round(idx * ANGLE_INCREMENT_DEG))
             if (ang is None or FORCE_INCREMENT_ANGLE) else int(ang))
    return idx, angle % 360, suffix


def scan(dirpath, prefix, exts):
    found = {}
    for ext in exts:
        for path in sorted(glob.glob(os.path.join(dirpath, f"{prefix}*{ext}"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            got = parse_suffix(stem, prefix)
            if got is None:
                continue
            idx, angle, suffix = got
            found.setdefault(idx, (angle, suffix, path))
    return found


def polyline(img, uv, color):
    pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, color, DRAW_THICK, cv2.LINE_AA)


# ===========================================================================
# viewer -- one cloud set per axis approach, per-angle toggles
# ===========================================================================
def comparison_viewer(variants, angles, statics, frames=None):
    """variants: {approach_name: {angle: point_cloud}}, already transformed."""
    names = list(variants)
    default = next((n for n in names if n.endswith(VIEW_DEFAULT_APPROACH)), names[-1])
    state = {"approach": default, "i": 0}

    items = []
    for name in names:
        merged = o3d.geometry.PointCloud()
        for a in angles:
            merged += variants[name][a]
        if VIEW_VOXEL_M > 0:
            merged = merged.voxel_down_sample(VIEW_VOXEL_M)
        items.append(dict(name=f"{name} :: MERGED", geometry=merged, group=name,
                          is_visible=(name == default)))
        if VIEW_PER_ANGLE:
            for a in angles:
                items.append(dict(name=f"{name} :: ang{a:03d}",
                                  geometry=variants[name][a], group=name,
                                  is_visible=False))
    if frames and VIEW_SHOW_FRAMES:
        for a in angles:
            items.append(dict(name=f"frame {a:03d}", geometry=frames[a],
                              group="shot frames", is_visible=False))
    for nm, g in statics:
        items.append(dict(name=nm, geometry=g, group="geometry", is_visible=True))

    def paint(vis, sel_angles, merged_on):
        approach = state["approach"]
        for name in names:
            vis.show_geometry(f"{name} :: MERGED", merged_on and name == approach)
            if VIEW_PER_ANGLE:
                for a in angles:
                    vis.show_geometry(f"{name} :: ang{a:03d}",
                                      (name == approach) and (a in sel_angles))

    def merged_only(vis):
        paint(vis, set(), True)
        print(f"[VIEW] {state['approach']}: merged")

    def all_angles(vis):
        paint(vis, set(angles), False)
        print(f"[VIEW] {state['approach']}: all {len(angles)} angles")

    def none(vis):
        paint(vis, set(), False)

    def every_n(vis):
        sel = {a for a in angles if a % VIEW_STEP_DEG == 0}
        paint(vis, sel, False)
        print(f"[VIEW] {state['approach']}: {sorted(sel)}")

    def ref_next(vis):
        state["i"] = (state["i"] + 1) % len(angles)
        sel = {angles[state["i"]]}
        if REF_ANGLE_DEG in angles:
            sel.add(REF_ANGLE_DEG)
        paint(vis, sel, False)
        print(f"[VIEW] {state['approach']}: {sorted(sel)}")

    def solo_next(vis):
        state["i"] = (state["i"] + 1) % len(angles)
        paint(vis, {angles[state["i"]]}, False)
        print(f"[VIEW] {state['approach']}: {angles[state['i']]:03d}")

    def solo_prev(vis):
        state["i"] = (state["i"] - 1) % len(angles)
        paint(vis, {angles[state["i"]]}, False)
        print(f"[VIEW] {state['approach']}: {angles[state['i']]:03d}")

    def next_approach(vis):
        state["approach"] = names[(names.index(state["approach"]) + 1) % len(names)]
        paint(vis, set(), True)
        print(f"[VIEW] approach -> {state['approach']}")

    print(f"[VIEW] approaches: {names}   (active: {default})")
    print(f"[VIEW] angles: {angles}")
    print("[VIEW] buttons act on the ACTIVE approach; the side panel also has a "
          "checkbox per angle")
    o3d.visualization.draw(
        items, title="turntable alignment", show_ui=True, point_size=POINT_SIZE,
        actions=[("Next approach", next_approach), ("Merged", merged_only),
                 ("All angles", all_angles), ("None", none),
                 (f"Every {VIEW_STEP_DEG} deg", every_n),
                 ("Ref + next", ref_next), ("Solo next", solo_next),
                 ("Solo prev", solo_prev)])


# ===========================================================================
# main
# ===========================================================================
def main():
    out_dir = os.path.join(INPUT_DIR, OUT_SUBDIR)
    dbg_dir = os.path.join(out_dir, DEBUG_SUBDIR)
    os.makedirs(dbg_dir, exist_ok=True)

    K, dist, src_w, src_h = load_intrinsics(INTRINSICS_FILE)
    colors = scan(COLOR_DIR, COLOR_PREFIX, (COLOR_EXT,))
    depths = scan(DEPTH_DIR, DEPTH_PREFIX, DEPTH_EXTS)
    if not colors:
        raise SystemExit(f"no {COLOR_PREFIX}*{COLOR_EXT} in {COLOR_DIR}")

    frames = [(idx,) + colors[idx] for idx in sorted(colors)]
    print(f"[LOAD] {len(frames)} colour frames | increment {ANGLE_INCREMENT_DEG:g} deg "
          f"({'forced' if FORCE_INCREMENT_ANGLE else 'ang-field wins'})")

    probe = cv2.imread(frames[0][3], cv2.IMREAD_COLOR)
    if probe is None:
        raise SystemExit(f"cannot read {frames[0][3]}")
    h_img, w_img = probe.shape[:2]
    Ks = scale_K(K, (src_w, src_h), (w_img, h_img))
    xn, yn = ray_map(Ks, dist, w_img, h_img)
    detector = Detector(families=TAG_FAMILY)
    ring_ids = set(TOP_RING_IDS) | set(BOTTOM_RING_IDS)

    obs = []
    for idx, angle, suffix, cpath in frames:
        if idx not in depths:
            print(f"[SKIP] no depth for index {idx:03d}")
            continue
        dpath = depths[idx][2]
        color = cv2.imread(cpath, cv2.IMREAD_COLOR)
        depth = (np.load(dpath) if dpath.endswith(".npy")
                 else cv2.imread(dpath, cv2.IMREAD_UNCHANGED))
        if color is None or depth is None:
            print(f"[SKIP] unreadable pair {suffix}")
            continue
        tags = [t for t in detect_tags(color, depth, xn, yn, detector, ring_ids)
                if t["tilt"] < MAX_TILT_DEG]
        for t in tags:
            t["ring"] = "top" if t["id"] in TOP_RING_IDS else "bottom"
            t["angle"] = angle
        obs.append(dict(idx=idx, angle=angle, suffix=suffix, path=cpath, tags=tags))
        print(f"[TAGS] {angle:03d}  "
              f"top={sorted({t['id'] for t in tags if t['ring']=='top'})}  "
              f"bottom={sorted({t['id'] for t in tags if t['ring']=='bottom'})}")

    all_tags = [t for o in obs for t in o["tags"]]
    top_tags = [t for t in all_tags if t["ring"] == "top"]
    bot_tags = [t for t in all_tags if t["ring"] == "bottom"]
    print(f"\n[TAGS] {len(all_tags)} usable observations "
          f"({len(top_tags)} top, {len(bot_tags)} bottom)")
    if len(top_tags) < 4 or len(bot_tags) < 4:
        raise SystemExit("not enough tag observations on one of the rings")

    up_cam = np.array([0.0, -1.0, 0.0])

    def tilt_deg(v):
        return float(np.degrees(np.arccos(np.clip(v @ up_cam, -1, 1))))

    # ---- 1. seed --------------------------------------------------------
    u_seed = unit(np.stack([t["u"] for t in all_tags]).mean(0))
    print(f"\n[SEED] tag-face normals  u={u_seed.round(4)}  "
          f"tilt {tilt_deg(u_seed):.3f} deg")

    # ---- 2. baseline ----------------------------------------------------
    u_base = P_base = None
    if BASELINE_COMPARE:
        u_base, top, bot = solve_two_ring_axis(top_tags, bot_tags, u_seed)
        P_base = bot["C3"]
        sep = float(np.linalg.norm(top["C3"] - bot["C3"]))
        print(f"[BASE] two independent rings, {sep*100:.2f} cm baseline")
        print(f"[BASE] top radii (cm): " +
              ", ".join("id%d=%.2f" % (k, v * 100) for k, v in sorted(top["radii"].items())))
        print(f"[BASE] bot radii (cm): " +
              ", ".join("id%d=%.2f" % (k, v * 100) for k, v in sorted(bot["radii"].items())))
        print(f"[BASE] circle scatter: top {top['rms']*1000:.2f} mm, "
              f"bottom {bot['rms']*1000:.2f} mm")
        print(f"[BASE] u={u_base.round(4)}  tilt {tilt_deg(u_base):.3f} deg  "
              f"P={P_base.round(4)}")

    # ---- 3. joint concentric fit ----------------------------------------
    print("\n[JOINT] concentric fit: shared axis + shared centre")
    u0 = u_base if u_base is not None else u_seed
    P0 = P_base if P_base is not None else np.stack(
        [t["center"] for t in bot_tags]).mean(0)
    u_hat, P, h_all, r_all = solve_joint_axis(all_tags, u0, P0)

    bot_idx = [i for i, t in enumerate(all_tags) if t["ring"] == "bottom"]
    P = P + float(np.mean(h_all[bot_idx])) * u_hat

    print(f"[JOINT] u={u_hat.round(4)}  tilt {tilt_deg(u_hat):.3f} deg")

    bias_report(all_tags, h_all, r_all)

    cands = [("seed", u_seed), ("joint", u_hat)]
    print("\n[SPREAD] disagreement between the tag-side axis estimates")
    worst = 0.0
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            d = float(np.degrees(np.arccos(np.clip(cands[i][1] @ cands[j][1], -1, 1))))
            worst = max(worst, d)
            print(f"   {cands[i][0]:8s} vs {cands[j][0]:8s} : {d:6.3f} deg")
    if u_base is not None:
        db = float(np.degrees(np.arccos(np.clip(u_base @ u_hat, -1, 1))))
        print(f"   baseline vs joint    : {db:6.3f} deg   (informational only)")
    print(f"[SPREAD] -> ~{worst:.2f} deg of tag-derived axis uncertainty")

    u_tag, P_tag = u_hat, P
    if TAG_AXIS_SOURCE == "baseline" and u_base is not None:
        u_tag, P_tag = u_base, P_base
    elif TAG_AXIS_SOURCE == "seed":
        u_tag = u_seed

    print("\n" + "=" * 70)
    print(f"  TURNTABLE AXIS UNIT VECTOR (tags, '{TAG_AXIS_SOURCE}', +Y of table)")
    print(f"    u = [{u_tag[0]: .6f}, {u_tag[1]: .6f}, {u_tag[2]: .6f}]")
    print(f"    tilt from camera-up = {tilt_deg(u_tag):.3f} deg")
    print("  AXIS POINT P (bottom-ring height, camera frame, metres)")
    print(f"    P = [{P_tag[0]: .6f}, {P_tag[1]: .6f}, {P_tag[2]: .6f}]")
    print("=" * 70)

    # ---- depth sanity check ---------------------------------------------
    ref_obs = next((o for o in obs if o["angle"] == REF_ANGLE_DEG), None)
    dbg_tag = (next((t for t in ref_obs["tags"] if t["id"] == DEBUG_TAG_ID), None)
               if ref_obs else None)
    z_meas = None
    if dbg_tag is None:
        print(f"\n[ZCHK] tag {DEBUG_TAG_ID} not visible at {REF_ANGLE_DEG:03d}")
    else:
        z_meas = float(dbg_tag["center"][2])
        print(f"\n[ZCHK] tag id{DEBUG_TAG_ID} @ angle {REF_ANGLE_DEG:03d}")
        print(f"[ZCHK]   camera z (tag face)  = {z_meas:.4f} m")
        running = z_meas
        for off in DEBUG_Z_OFFSETS_M:
            running += off
            print(f"[ZCHK]   + {off*100:.2f} cm            -> {running:.4f} m")
        total = sum(DEBUG_Z_OFFSETS_M)
        print(f"[ZCHK]   manual  (z + {total*100:.2f} cm) = {z_meas + total:.4f} m")
        print(f"[ZCHK]   geometric axis z     = {P_tag[2]:.4f} m")
        print(f"[ZCHK]   manual - geometric   = {(z_meas + total - P_tag[2])*1000:+.1f} mm")

    # ---- rotation sign ---------------------------------------------------
    e1, e2 = basis_from_axis(u_tag)
    c2 = np.array([P_tag @ e1, P_tag @ e2])

    def circ_angle(t):
        q = np.array([t["center"] @ e1, t["center"] @ e2]) - c2
        return float(np.arctan2(q[1], q[0]))

    def cmean(a):
        return float(np.angle(np.mean(np.exp(1j * np.asarray(a)))))

    theta = {o["angle"]: [(t["id"], circ_angle(t)) for t in o["tags"]]
             for o in obs if o["tags"]}

    if ANGLE_SIGN in (1, -1):
        undo_sign = int(ANGLE_SIGN)
        s_meas = -undo_sign
        print(f"\n[SIGN] forced by config: undo sign = {undo_sign:+d}")
    else:
        report = {}
        for s_try in (1, -1):
            beta, resid = {}, []
            for k in {i for v in theta.values() for i, _ in v}:
                beta[k] = cmean([th - s_try * np.deg2rad(a)
                                 for a, v in theta.items() for i, th in v if i == k])
            for a, v in theta.items():
                for i, th in v:
                    resid.append(np.angle(np.exp(1j * (th - beta[i]
                                                       - s_try * np.deg2rad(a)))))
            report[s_try] = float(np.degrees(np.sqrt(np.mean(np.square(resid)))))
            print(f"\n[SIGN] angle_sign={s_try:+d}  residual RMS={report[s_try]:7.2f} deg")
        s_meas = min(report, key=report.get)
        undo_sign = -s_meas
        print(f"[SIGN] measured sign = {s_meas:+d} -> undo sign = {undo_sign:+d}")

    beta = {}
    for k in {i for v in theta.values() for i, _ in v}:
        beta[k] = cmean([th - s_meas * np.deg2rad(a)
                         for a, v in theta.items() for i, th in v if i == k])
    phi = {a: cmean([th - beta[i] for i, th in v]) for a, v in theta.items()}
    ref_phi = phi.get(REF_ANGLE_DEG, 0.0)
    slip = {a: float(np.degrees(np.angle(np.exp(
        1j * (phi[a] - ref_phi - s_meas * np.deg2rad(a)))))) for a in sorted(phi)}
    if slip:
        print(f"[ANGL] apparent slip: mean|.|={np.mean(np.abs(list(slip.values()))):.2f} "
              f"deg  max|.|={np.max(np.abs(list(slip.values()))):.2f} deg")
        print(f"[ANGL] closure at {max(slip):03d} deg = {slip[max(slip)]:+.2f} deg")

    # ---- debug overlays --------------------------------------------------
    if WRITE_DEBUG_IMAGES:
        axis_pts = np.stack([P_tag + u_tag * t for t in
                             np.linspace(-AXIS_DRAW_HALF_LEN_M,
                                         AXIS_DRAW_HALF_LEN_M, AXIS_SAMPLES)])
        axis_uv = project(axis_pts, Ks, dist)
        for o in obs:
            vis = cv2.imread(o["path"], cv2.IMREAD_COLOR)
            if vis is None:
                continue
            polyline(vis, axis_uv, (0, 0, 255))
            for t in o["tags"]:
                col = (0, 255, 0) if t["ring"] == "top" else (0, 255, 255)
                cv2.polylines(vis, [np.round(t["corners"]).astype(np.int32).reshape(-1, 1, 2)],
                              True, col, 2, cv2.LINE_AA)
                cu, cv_ = np.round(project([t["center"]], Ks, dist)[0]).astype(int)
                cv2.putText(vis, f"id{t['id']} ob{t['obliq']:.0f}",
                            (int(cu) + 6, int(cv_) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, col, 1, cv2.LINE_AA)
            cv2.putText(vis, f"angle {o['angle']:03d}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(dbg_dir, f"axis_{o['angle']:03d}_{o['suffix']}.png"),
                        vis)
        print(f"\n[SAVE] {dbg_dir}/axis_*.png")

    # =======================================================================
    # point clouds
    # =======================================================================
    plys = scan(CLOUD_DIR, CLOUD_PREFIX, (CLOUD_EXT,))
    if not plys:
        print(f"\n[PCD ] no {CLOUD_PREFIX}*{CLOUD_EXT} in {CLOUD_DIR} -- stopping")
        return

    raw = {}
    for idx in sorted(plys):
        angle, suffix, path = plys[idx]
        if LOAD_ANGLES is not None and angle not in LOAD_ANGLES:
            continue
        pcd = o3d.io.read_point_cloud(path)
        if VOXEL_M > 0:
            pcd = pcd.voxel_down_sample(VOXEL_M)
        raw[angle] = (suffix, pcd)
    if not raw:
        raise SystemExit("no clouds selected (check LOAD_ANGLES)")
    angles = sorted(raw)
    res = f"voxel {VOXEL_M*1000:.1f} mm" if VOXEL_M > 0 else "full resolution"
    print(f"\n[PCD ] {len(angles)} clouds loaded ({res})")

    u_used, P_used = u_tag, P_tag
    refine_before = refine_after = None
    if REFINE_AXIS_ON_CLOUDS:
        rng = np.random.default_rng(0)
        samples = {}
        for a in angles:
            X = np.asarray(raw[a][1].points)
            if len(X) > REFINE_MAX_PTS:
                X = X[rng.choice(len(X), REFINE_MAX_PTS, replace=False)]
            samples[a] = np.ascontiguousarray(X)
        print()
        u_used, P_used, refine_before, refine_after = refine_axis_on_clouds(
            samples, angles, u_tag, P_tag, undo_sign)
        print("\n" + "=" * 70)
        print("  REFINED AXIS (used for the transforms below)")
        print(f"    u = [{u_used[0]: .6f}, {u_used[1]: .6f}, {u_used[2]: .6f}]")
        print(f"    P = [{P_used[0]: .6f}, {P_used[1]: .6f}, {P_used[2]: .6f}]")
        print("=" * 70)

    clouds, shot_frames, transforms = {}, {}, {}
    for a in angles:
        suffix, pcd = raw[a]
        deg = undo_sign * a
        T = rotation_about(P_used, u_used, deg)
        pc = o3d.geometry.PointCloud(pcd)
        pc.transform(T)
        clouds[a] = pc
        transforms[f"{a:03d}"] = dict(suffix=suffix, undo_deg=float(deg),
                                      source="filename", T=T.tolist())
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=FRAME_AXIS_SIZE_M)
        cf.transform(T)
        shot_frames[a] = cf
        print(f"[XFRM] {a:03d}  undo={deg:+8.2f} deg  pts={len(pc.points):,}")

    if RESIDUAL_MODE in ("neighbour", "both") and len(angles) > 1:
        print("[RESID] each frame vs its neighbour (high overlap -- the honest metric)")
        vals = []
        for i, a in enumerate(angles):
            b = angles[(i + 1) % len(angles)]
            d = np.asarray(clouds[a].compute_point_cloud_distance(clouds[b]))
            vals.append(np.median(d))
        print(f"   median over {len(vals)} pairs = {np.median(vals)*1000:.2f} mm  "
              f"worst = {np.max(vals)*1000:.2f} mm")

    if RESIDUAL_MODE in ("ref", "both") and REF_ANGLE_DEG in clouds:
        target = clouds[REF_ANGLE_DEG]
        print("[RESID] each frame vs the reference frame "
              "(inflated past ~120 deg by loss of surface overlap)")
        meds = []
        for a in angles:
            if a == REF_ANGLE_DEG:
                continue
            d = np.asarray(clouds[a].compute_point_cloud_distance(target))
            meds.append(np.median(d))
            print(f"   {a:03d}  median={np.median(d)*1000:6.2f} mm  "
                  f"p90={np.percentile(d, 90)*1000:6.2f} mm")
        if meds:
            print(f"[RESID] overall median = {np.median(meds)*1000:.2f} mm")

    merged = o3d.geometry.PointCloud()
    for a in angles:
        merged += clouds[a]
    if MERGED_VOXEL_M > 0:
        merged = merged.voxel_down_sample(MERGED_VOXEL_M)
    print(f"[MERGE] {len(merged.points):,} pts")

    a0 = P_used - u_used * AXIS_HALF_LEN_M
    a1 = P_used + u_used * AXIS_HALF_LEN_M
    axis_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.stack([a0, a1])),
        lines=o3d.utility.Vector2iVector([[0, 1]]))
    axis_ls.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])
    world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    world.translate(P_used)
    statics = [("rotation_axis", axis_ls), ("axis_origin_P", world)]

    if SAVE_MERGED:
        o3d.io.write_point_cloud(os.path.join(out_dir, "merged.ply"), merged)
        with open(os.path.join(out_dir, "axis_refined.json"), "w") as f:
            json.dump(dict(
                axis_dir=u_used.tolist(), axis_point=P_used.tolist(),
                axis_dir_tags=u_tag.tolist(), axis_point_tags=P_tag.tolist(),
                axis_dir_seed=u_seed.tolist(),
                axis_dir_baseline=(u_base.tolist() if u_base is not None else None),
                radius_mode=RADIUS_MODE, undo_sign=int(undo_sign),
                refine_on_clouds=bool(REFINE_AXIS_ON_CLOUDS),
                refine_cost_before_m=refine_before, refine_cost_after_m=refine_after,
                debug_tag_id=DEBUG_TAG_ID, debug_tag_z_m=z_meas,
                slip_deg=slip, frames=transforms), f, indent=2)
        print(f"[SAVE] {out_dir}/merged.ply")
        print(f"[SAVE] {out_dir}/axis_refined.json")

    # =======================================================================
    # viewer: every axis approach, true colour, per-angle selection
    # =======================================================================
    if SHOW_VIEWER:
        axes = {}
        if "seed" in VIEW_APPROACHES:
            axes["1_seed"] = (u_seed, P)
        if "baseline" in VIEW_APPROACHES and u_base is not None:
            axes["2_baseline"] = (u_base, P_base)
        if "joint" in VIEW_APPROACHES:
            axes["3_joint"] = (u_hat, P)
        if "refined" in VIEW_APPROACHES:
            axes["4_refined"] = (u_used, P_used)
        if not axes:
            axes["4_refined"] = (u_used, P_used)

        view_angles = [a for a in angles
                       if VIEW_ANGLES is None or a in set(VIEW_ANGLES)]
        if not view_angles:
            view_angles = angles
        print(f"\n[VIEW] building {len(axes)} axis variant(s) x "
              f"{len(view_angles)} angle(s)  "
              f"({'true colour' if VIEW_TRUE_COLOR else 'per-frame palette'})")

        variants = {}
        for name in sorted(axes):
            au, aP = axes[name]
            per_angle = {}
            for i, a in enumerate(view_angles):
                pc = o3d.geometry.PointCloud(raw[a][1])
                pc.transform(rotation_about(aP, au, undo_sign * a))
                if not VIEW_TRUE_COLOR or not pc.has_colors():
                    pc.paint_uniform_color(PALETTE[i % len(PALETTE)])
                if VIEW_VOXEL_M > 0:
                    pc = pc.voxel_down_sample(VIEW_VOXEL_M)
                per_angle[a] = pc
            variants[name] = per_angle

        comparison_viewer(variants, view_angles, statics, shot_frames)


if __name__ == "__main__":
    main()