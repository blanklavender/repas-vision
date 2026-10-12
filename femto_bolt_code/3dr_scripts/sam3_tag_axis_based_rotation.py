#!/usr/bin/env python3
"""
turntable_align.py  --  turntable axis fit + point-cloud alignment, one script.

Single pipeline
---------------
AXIS   : two independent ring circles.  The mean of the AprilTag face "up"
         vectors is the seed that starts the iteration; the axis is then the
         bottom-ring centre -> top-ring centre direction, re-solved until the
         direction stops moving.  P is the bottom-ring centre.

ANGLES : the true stepper-motor angles from manifest.csv (column `true_deg`)
         are what drive every rotation, sign estimate and residual.  The
         commanded / filename angle is kept only as a label.

Filenames:  color_012_ang120.png  depth_005_ang050.png  cloud_000_ang000.ply
"""


import configparser
import csv
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
INPUT_DIR = r"C:\Users\quartz\02_projects\repas-vision\captures\femto_bolt\flora_unit\yellow_pot_08-25_deg_10_speed_5"

COLOR_DIR = os.path.join(INPUT_DIR, "rgb")
DEPTH_DIR = os.path.join(INPUT_DIR, "depth")
CLOUD_DIR = os.path.join(INPUT_DIR, "pointcloud")

OUT_SUBDIR = "aligned"
DEBUG_SUBDIR = "debug"

MANIFEST_FILE = os.path.join(INPUT_DIR, "manifest.csv")
MANIFEST_INDEX_COL = "index"
MANIFEST_TRUE_COL = "true_deg"
MANIFEST_COLOR_COL = "color"
REQUIRE_MANIFEST = True     # False = silently fall back to filename angles

INTRINSICS_FILE = (r"C:\Users\quartz\02_projects\repas-vision\femto_bolt_code\scripts\calibration_parameters\CameraParam_Femto_BoltCL8956200ZM_Color1280x720_Depth640x576.ini")
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
ANGLE_INCREMENT_DEG = 10.0      # label fallback only
FORCE_INCREMENT_ANGLE = False
ANGLE_SIGN = "auto"
REF_ANGLE_DEG = 0               # label of the reference shot

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

RING_ITERS = 15
RING_TOL_DEG = 1e-4

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
# viewer
# ---------------------------------------------------------------------------
VIEW_APPROACH_NAME = "ring"
VIEW_ANGLES = None          # e.g. (0, 90, 180, 270); None = every loaded angle
VIEW_PER_ANGLE = True       # one toggleable item per angle
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
# axis: two independent ring circles, seeded by the tag-face normals
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


def solve_two_ring_axis(top_tags, bot_tags, u_seed, iters=RING_ITERS, tol=RING_TOL_DEG):
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


def axis_terms(u, P, X):
    d = X - P
    h = d @ u
    r = np.linalg.norm(d - np.outer(h, u), axis=1)
    return h, r


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
# filenames + manifest
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


def load_manifest(path):
    """index -> true stepper angle (deg).  Also keys the colour file name."""
    by_index, by_color = {}, {}
    if not os.path.isfile(path):
        if REQUIRE_MANIFEST:
            raise SystemExit(f"manifest not found: {path}")
        print(f"[MANI] no manifest at {path} -- falling back to filename angles")
        return by_index, by_color
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if MANIFEST_TRUE_COL not in row or row[MANIFEST_TRUE_COL] in (None, ""):
                continue
            true_deg = float(row[MANIFEST_TRUE_COL]) % 360.0
            if row.get(MANIFEST_INDEX_COL) not in (None, ""):
                by_index[int(row[MANIFEST_INDEX_COL])] = true_deg
            name = row.get(MANIFEST_COLOR_COL)
            if name:
                by_color[os.path.splitext(os.path.basename(name))[0]] = true_deg
    print(f"[MANI] {len(by_index)} true angles from {os.path.basename(path)} "
          f"(column '{MANIFEST_TRUE_COL}')")
    return by_index, by_color


def true_angle_for(idx, label, stem, by_index, by_color):
    if idx in by_index:
        return by_index[idx], "manifest"
    if stem in by_color:
        return by_color[stem], "manifest"
    if REQUIRE_MANIFEST:
        raise SystemExit(f"no manifest row for index {idx:03d} ({stem})")
    print(f"[MANI] index {idx:03d}: no row -- using filename angle {label}")
    return float(label), "filename"


def polyline(img, uv, color):
    pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, color, DRAW_THICK, cv2.LINE_AA)


# ===========================================================================
# viewer -- per-angle toggles, single axis approach
# ===========================================================================
def comparison_viewer(variants, angles, statics, frames=None):
    """variants: {approach_name: {angle: point_cloud}}, already transformed."""
    names = list(variants)
    default = names[0]
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

    print(f"[VIEW] approach: {default}")
    print(f"[VIEW] angles: {angles}")
    print("[VIEW] the side panel also has a checkbox per angle")
    o3d.visualization.draw(
        items, title="turntable alignment", show_ui=True, point_size=POINT_SIZE,
        actions=[("Merged", merged_only), ("All angles", all_angles),
                 ("None", none), (f"Every {VIEW_STEP_DEG} deg", every_n),
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
    mani_idx, mani_color = load_manifest(MANIFEST_FILE)

    colors = scan(COLOR_DIR, COLOR_PREFIX, (COLOR_EXT,))
    depths = scan(DEPTH_DIR, DEPTH_PREFIX, DEPTH_EXTS)
    if not colors:
        raise SystemExit(f"no {COLOR_PREFIX}*{COLOR_EXT} in {COLOR_DIR}")

    frames = [(idx,) + colors[idx] for idx in sorted(colors)]
    print(f"[LOAD] {len(frames)} colour frames | labels from filenames, "
          f"rotations from manifest '{MANIFEST_TRUE_COL}'")

    # label angle -> true stepper angle (the only angle the math uses)
    true_ang, ang_src = {}, {}
    for idx, angle, suffix, cpath in frames:
        stem = os.path.splitext(os.path.basename(cpath))[0]
        td, src = true_angle_for(idx, angle, stem, mani_idx, mani_color)
        true_ang[angle] = td
        ang_src[angle] = src
    dev = [true_ang[a] - a for a in sorted(true_ang)]
    print(f"[MANI] true - commanded: mean {np.mean(dev):+.4f} deg  "
          f"max|.| {np.max(np.abs(dev)):.4f} deg")

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
            t["true_deg"] = true_ang[angle]
        obs.append(dict(idx=idx, angle=angle, true_deg=true_ang[angle],
                        suffix=suffix, path=cpath, tags=tags))
        print(f"[TAGS] {angle:03d} (true {true_ang[angle]:8.4f})  "
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

    # ---- axis: seed -> two-ring solve --------------------------------------
    u_seed = unit(np.stack([t["u"] for t in all_tags]).mean(0))
    print(f"\n[AXIS] seed from tag-face normals  u={u_seed.round(4)}  "
          f"tilt {tilt_deg(u_seed):.3f} deg")

    u_axis, top, bot = solve_two_ring_axis(top_tags, bot_tags, u_seed)
    P_axis = bot["C3"]
    sep = float(np.linalg.norm(top["C3"] - bot["C3"]))
    print(f"[AXIS] two independent rings, {sep*100:.2f} cm baseline")
    print("[AXIS] top radii (cm): " +
          ", ".join("id%d=%.2f" % (k, v * 100) for k, v in sorted(top["radii"].items())))
    print("[AXIS] bot radii (cm): " +
          ", ".join("id%d=%.2f" % (k, v * 100) for k, v in sorted(bot["radii"].items())))
    print(f"[AXIS] circle scatter: top {top['rms']*1000:.2f} mm, "
          f"bottom {bot['rms']*1000:.2f} mm")
    moved = float(np.degrees(np.arccos(np.clip(u_seed @ u_axis, -1, 1))))
    print(f"[AXIS] moved {moved:.3f} deg away from the seed")

    X_all = np.stack([t["center"] for t in all_tags])
    h_all, r_all = axis_terms(u_axis, P_axis, X_all)
    bias_report(all_tags, h_all, r_all)

    print("\n" + "=" * 70)
    print("  TURNTABLE AXIS UNIT VECTOR (two-ring fit, +Y of table)")
    print(f"    u = [{u_axis[0]: .6f}, {u_axis[1]: .6f}, {u_axis[2]: .6f}]")
    print(f"    tilt from camera-up = {tilt_deg(u_axis):.3f} deg")
    print("  AXIS POINT P (bottom-ring centre, camera frame, metres)")
    print(f"    P = [{P_axis[0]: .6f}, {P_axis[1]: .6f}, {P_axis[2]: .6f}]")
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
        print(f"[ZCHK]   geometric axis z     = {P_axis[2]:.4f} m")
        print(f"[ZCHK]   manual - geometric   = {(z_meas + total - P_axis[2])*1000:+.1f} mm")

    # ---- rotation sign (against the true stepper angles) ------------------
    e1, e2 = basis_from_axis(u_axis)
    c2 = np.array([P_axis @ e1, P_axis @ e2])

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
                beta[k] = cmean([th - s_try * np.deg2rad(true_ang[a])
                                 for a, v in theta.items() for i, th in v if i == k])
            for a, v in theta.items():
                for i, th in v:
                    resid.append(np.angle(np.exp(
                        1j * (th - beta[i] - s_try * np.deg2rad(true_ang[a])))))
            report[s_try] = float(np.degrees(np.sqrt(np.mean(np.square(resid)))))
            print(f"\n[SIGN] angle_sign={s_try:+d}  residual RMS={report[s_try]:7.2f} deg")
        s_meas = min(report, key=report.get)
        undo_sign = -s_meas
        print(f"[SIGN] measured sign = {s_meas:+d} -> undo sign = {undo_sign:+d}")

    beta = {}
    for k in {i for v in theta.values() for i, _ in v}:
        beta[k] = cmean([th - s_meas * np.deg2rad(true_ang[a])
                         for a, v in theta.items() for i, th in v if i == k])
    phi = {a: cmean([th - beta[i] for i, th in v]) for a, v in theta.items()}
    ref_phi = phi.get(REF_ANGLE_DEG, 0.0)
    ref_true = true_ang.get(REF_ANGLE_DEG, 0.0)
    slip = {a: float(np.degrees(np.angle(np.exp(
        1j * (phi[a] - ref_phi - s_meas * np.deg2rad(true_ang[a] - ref_true))))))
        for a in sorted(phi)}
    if slip:
        print(f"[ANGL] apparent slip vs true angles: "
              f"mean|.|={np.mean(np.abs(list(slip.values()))):.2f} "
              f"deg  max|.|={np.max(np.abs(list(slip.values()))):.2f} deg")
        print(f"[ANGL] closure at {max(slip):03d} deg = {slip[max(slip)]:+.2f} deg")

    # ---- debug overlays --------------------------------------------------
    if WRITE_DEBUG_IMAGES:
        axis_pts = np.stack([P_axis + u_axis * t for t in
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
            cv2.putText(vis, f"angle {o['angle']:03d}  true {o['true_deg']:.3f}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
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
        if angle not in true_ang:
            stem = os.path.splitext(os.path.basename(path))[0]
            td, src = true_angle_for(idx, angle, stem, mani_idx, mani_color)
            true_ang[angle] = td
            ang_src[angle] = src
        pcd = o3d.io.read_point_cloud(path)
        if VOXEL_M > 0:
            pcd = pcd.voxel_down_sample(VOXEL_M)
        raw[angle] = (suffix, pcd)
    if not raw:
        raise SystemExit("no clouds selected (check LOAD_ANGLES)")
    angles = sorted(raw)
    res = f"voxel {VOXEL_M*1000:.1f} mm" if VOXEL_M > 0 else "full resolution"
    print(f"\n[PCD ] {len(angles)} clouds loaded ({res})")

    u_used, P_used = u_axis, P_axis

    clouds, shot_frames, transforms = {}, {}, {}
    for a in angles:
        suffix, pcd = raw[a]
        deg = undo_sign * true_ang[a]
        T = rotation_about(P_used, u_used, deg)
        pc = o3d.geometry.PointCloud(pcd)
        pc.transform(T)
        clouds[a] = pc
        transforms[f"{a:03d}"] = dict(suffix=suffix, commanded_deg=float(a),
                                      true_deg=float(true_ang[a]),
                                      undo_deg=float(deg), source=ang_src.get(a, "?"),
                                      T=T.tolist())
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=FRAME_AXIS_SIZE_M)
        cf.transform(T)
        shot_frames[a] = cf
        print(f"[XFRM] {a:03d}  true={true_ang[a]:8.4f}  undo={deg:+9.4f} deg  "
              f"pts={len(pc.points):,}")

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
                axis_dir_seed=u_seed.tolist(),
                axis_method="two_ring", undo_sign=int(undo_sign),
                angle_source=MANIFEST_TRUE_COL,
                manifest=os.path.abspath(MANIFEST_FILE),
                ring_separation_m=sep,
                ring_rms_top_m=top["rms"], ring_rms_bottom_m=bot["rms"],
                debug_tag_id=DEBUG_TAG_ID, debug_tag_z_m=z_meas,
                slip_deg=slip, frames=transforms), f, indent=2)
        print(f"[SAVE] {out_dir}/merged.ply")
        print(f"[SAVE] {out_dir}/axis_refined.json")

    # =======================================================================
    # viewer: true colour, per-angle selection
    # =======================================================================
    if SHOW_VIEWER:
        view_angles = [a for a in angles
                       if VIEW_ANGLES is None or a in set(VIEW_ANGLES)]
        if not view_angles:
            view_angles = angles
        print(f"\n[VIEW] building {len(view_angles)} angle(s)  "
              f"({'true colour' if VIEW_TRUE_COLOR else 'per-frame palette'})")

        per_angle = {}
        for i, a in enumerate(view_angles):
            pc = o3d.geometry.PointCloud(raw[a][1])
            pc.transform(rotation_about(P_used, u_used, undo_sign * true_ang[a]))
            if not VIEW_TRUE_COLOR or not pc.has_colors():
                pc.paint_uniform_color(PALETTE[i % len(PALETTE)])
            if VIEW_VOXEL_M > 0:
                pc = pc.voxel_down_sample(VIEW_VOXEL_M)
            per_angle[a] = pc
        variants = {VIEW_APPROACH_NAME: per_angle}

        comparison_viewer(variants, view_angles, statics, shot_frames)


if __name__ == "__main__":
    main()