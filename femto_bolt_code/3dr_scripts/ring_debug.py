#!/usr/bin/env python3
"""
ring_projection.py  --  is the tag ring actually a circle, and do the true
angles actually undo the rotation?

Projects every AprilTag centre onto the plane perpendicular to the fitted
turntable axis and writes four PNGs:

    ring_top_raw.png         top ring, no rotation applied
    ring_bottom_raw.png      bottom ring, no rotation applied
    ring_top_derotated.png   top ring, rotation undone with manifest true_deg
    ring_bottom_derotated.png

How to read them
----------------
RAW        every observation of one tag traces an ARC as the table turns, and
           the six arcs should tile ONE circle centred on the origin (the
           fitted axis).  A tag whose arc sits at a different radius is either
           mis-measured or the ring is not concentric.  An overall oval instead
           of a circle means the axis direction is wrong.

DEROTATED  every observation of one tag should collapse onto a SINGLE POINT --
           the tag's fixed position on the pot.  Six tight dots at 60 deg
           spacing.  Smear along the circle (tangential) = the angles are
           wrong.  Smear across it (radial) = depth noise.  The panel reports
           both separately, which is the whole point of the experiment.

Standalone: reads the same captures as turntable_align.py, imports nothing
from it.
"""

import configparser
import csv
import glob
import os
import re

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from pupil_apriltags import Detector

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
INPUT_DIR = "../../flora_turntable/captures/yellow_pot_09-08_deg_10_speed_5"

COLOR_DIR = os.path.join(INPUT_DIR, "rgb")
DEPTH_DIR = os.path.join(INPUT_DIR, "depth")
OUT_SUBDIR = os.path.join("aligned", "ring_projection")

MANIFEST_FILE = os.path.join(INPUT_DIR, "manifest.csv")
MANIFEST_INDEX_COL = "index"
MANIFEST_TRUE_COL = "true_deg"
MANIFEST_COLOR_COL = "color"
REQUIRE_MANIFEST = True

INTRINSICS_FILE = ("../scripts/calibration_parameters/"
                   "CameraParam_Femto_BoltCL8956200ZM_Color1280x720_Depth640x576.ini")
INTRINSIC_SECTION = "ColorIntrinsic"
DISTORTION_SECTION = "ColorDistortion"
UNDISTORT_RAYS = True

COLOR_PREFIX = "color_"
DEPTH_PREFIX = "depth_"
COLOR_EXT = ".png"
DEPTH_EXTS = (".npy", ".png")

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
# angles / axis
# ---------------------------------------------------------------------------
ANGLE_INCREMENT_DEG = 10.0
FORCE_INCREMENT_ANGLE = False
ANGLE_SIGN = "auto"           # "auto" | +1 | -1

# "fit"    : two-ring fit from the tags themselves (recommended)
# "manual" : force the axis below, to test a hand-measured one
AXIS_SOURCE = "fit"
MANUAL_AXIS_DIR = (0.0, -1.0, 0.0)
MANUAL_AXIS_POINT = (-0.0635, 0.1397, 0.41275)

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
FIG_W_IN, FIG_H_IN = 11.0, 7.5
DPI = 200
MARKER_PT = 46
PAD_FRAC = 0.34               # padding around the data (leaves room for labels)

# Validated categorical palette: six hues, all-pairs CVD dE >= 9.6 and
# normal-vision dE >= 18.1 in both light and dark. Assigned in fixed order to
# the sorted tag ids of each ring -- never cycled.
PALETTE = ["#9b496a", "#c77f46", "#65710e", "#23a991", "#0071a2", "#9a82d0"]
# Marker shape is a second, redundant channel so identity never rests on
# colour alone (one slot sits below 3:1 on the surface).
MARKERS = ["o", "s", "^", "D", "v", "P"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS_RULE = "#c3c2b7"

NAME_RE = re.compile(r"^(?P<idx>\d+)(?:_ang(?P<ang>\d+))?$")


# ===========================================================================
# geometry
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
# tag detection  (depth-derived frame, no solvePnP)
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
        R, _ = kabsch(canon, np.asarray(meas))
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
        out.append(dict(id=int(det.tag_id), center=centre, u=u_hat,
                        obliq=obliq, tilt=tilt))
    return out


# ===========================================================================
# two-ring axis fit
# ===========================================================================
def fit_shared_circle(pts2d, ids):
    uid = sorted(set(ids))
    idx = {k: i for i, k in enumerate(uid)}
    A = np.zeros((len(pts2d), 2 + len(uid)))
    b = np.zeros(len(pts2d))
    for i, (p, k) in enumerate(zip(pts2d, ids)):
        A[i, 0], A[i, 1] = 2.0 * p[0], 2.0 * p[1]
        A[i, 2 + idx[k]] = 1.0
        b[i] = p @ p
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    radii = {k: float(np.sqrt(max(sol[2 + idx[k]] + c @ c, 0.0))) for k in uid}
    return c, radii


def fit_ring(tags, u, e1, e2, label):
    pts2d = np.array([[t["center"] @ e1, t["center"] @ e2] for t in tags])
    ids = [t["id"] for t in tags]
    if len(pts2d) < 2 + len(set(ids)) + 1:
        raise SystemExit(f"{label} ring: too few observations")
    c2, radii = fit_shared_circle(pts2d, ids)
    height = float(np.mean([t["center"] @ u for t in tags]))
    return dict(C3=c2[0] * e1 + c2[1] * e2 + height * u, radii=radii)


def solve_two_ring_axis(top_tags, bot_tags, u_seed):
    u = unit(u_seed)
    for _ in range(RING_ITERS):
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
        if delta < RING_TOL_DEG:
            break
    e1, e2 = basis_from_axis(u)
    return u, fit_ring(bot_tags, u, e1, e2, "bottom")["C3"]


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
    by_index, by_color = {}, {}
    if not os.path.isfile(path):
        if REQUIRE_MANIFEST:
            raise SystemExit(f"manifest not found: {path}")
        return by_index, by_color
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get(MANIFEST_TRUE_COL):
                continue
            td = float(row[MANIFEST_TRUE_COL]) % 360.0
            if row.get(MANIFEST_INDEX_COL) not in (None, ""):
                by_index[int(row[MANIFEST_INDEX_COL])] = td
            nm = row.get(MANIFEST_COLOR_COL)
            if nm:
                by_color[os.path.splitext(os.path.basename(nm))[0]] = td
    print(f"[MANI] {len(by_index)} true angles from {os.path.basename(path)}")
    return by_index, by_color


def true_angle_for(idx, label, stem, by_index, by_color):
    if idx in by_index:
        return by_index[idx]
    if stem in by_color:
        return by_color[stem]
    if REQUIRE_MANIFEST:
        raise SystemExit(f"no manifest row for index {idx:03d} ({stem})")
    return float(label)


# ===========================================================================
# gather observations
# ===========================================================================
def gather():
    K, dist, src_w, src_h = load_intrinsics(INTRINSICS_FILE)
    mani_idx, mani_col = load_manifest(MANIFEST_FILE)
    colors = scan(COLOR_DIR, COLOR_PREFIX, (COLOR_EXT,))
    depths = scan(DEPTH_DIR, DEPTH_PREFIX, DEPTH_EXTS)
    if not colors:
        raise SystemExit(f"no {COLOR_PREFIX}*{COLOR_EXT} in {COLOR_DIR}")

    probe = cv2.imread(colors[sorted(colors)[0]][2], cv2.IMREAD_COLOR)
    if probe is None:
        raise SystemExit("cannot read the first colour frame")
    h_img, w_img = probe.shape[:2]
    Ks = scale_K(K, (src_w, src_h), (w_img, h_img))
    xn, yn = ray_map(Ks, dist, w_img, h_img)
    detector = Detector(families=TAG_FAMILY)
    ring_ids = set(TOP_RING_IDS) | set(BOTTOM_RING_IDS)

    obs = []
    for idx in sorted(colors):
        angle, suffix, cpath = colors[idx]
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
        stem = os.path.splitext(os.path.basename(cpath))[0]
        td = true_angle_for(idx, angle, stem, mani_idx, mani_col)
        for t in detect_tags(color, depth, xn, yn, detector, ring_ids):
            if t["tilt"] >= MAX_TILT_DEG:
                continue
            t["ring"] = "top" if t["id"] in TOP_RING_IDS else "bottom"
            t["angle"] = angle
            t["true_deg"] = td
            obs.append(t)
    return obs


# ===========================================================================
# rotation sign, from the tag angular positions
# ===========================================================================
def choose_sign(obs, u, P):
    if ANGLE_SIGN in (1, -1):
        print(f"[SIGN] forced: undo sign = {int(ANGLE_SIGN):+d}")
        return int(ANGLE_SIGN)
    e1, e2 = basis_from_axis(u)
    c2 = np.array([P @ e1, P @ e2])

    def th(t):
        q = np.array([t["center"] @ e1, t["center"] @ e2]) - c2
        return float(np.arctan2(q[1], q[0]))

    def cmean(a):
        return float(np.angle(np.mean(np.exp(1j * np.asarray(a)))))

    best, report = None, {}
    for s in (1, -1):
        beta = {}
        for k in {t["id"] for t in obs}:
            beta[k] = cmean([th(t) - s * np.deg2rad(t["true_deg"])
                             for t in obs if t["id"] == k])
        r = [np.angle(np.exp(1j * (th(t) - beta[t["id"]]
                                   - s * np.deg2rad(t["true_deg"])))) for t in obs]
        report[s] = float(np.degrees(np.sqrt(np.mean(np.square(r)))))
        print(f"[SIGN] angle_sign={s:+d}  residual RMS = {report[s]:6.2f} deg")
        if best is None or report[s] < report[best]:
            best = s
    undo = -best
    print(f"[SIGN] measured sign = {best:+d} -> undo sign = {undo:+d}")
    return undo


# ===========================================================================
# the figure
# ===========================================================================
def algebraic_circle(x, y):
    """Least-squares circle through the points. Returns (cx, cy, r, rms)."""
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 0.0)))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), r, rms


def ring_figure(rows, ring_name, derotated, out_path, axis_note):
    """rows: list of dicts with id, x_mm, y_mm (already projected & centred)."""
    ids = sorted({r["id"] for r in rows})
    if len(ids) > len(PALETTE):
        raise SystemExit(f"{len(ids)} tags on the {ring_name} ring but only "
                         f"{len(PALETTE)} validated colours -- extend the palette")
    colour = {k: PALETTE[i] for i, k in enumerate(ids)}
    marker = {k: MARKERS[i % len(MARKERS)] for i, k in enumerate(ids)}

    X = np.array([r["x_mm"] for r in rows])
    Y = np.array([r["y_mm"] for r in rows])
    R = np.hypot(X, Y)
    r_mean = float(R.mean())
    cx, cy, r_fit, fit_rms = algebraic_circle(X, Y)

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI, facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.05, 1.0], wspace=0.04,
                          left=0.075, right=0.975, top=0.845, bottom=0.085)
    ax = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    panel = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    panel.axis("off")

    # --- reference circle at the mean radius (an annotation, not a gridline)
    ax.add_patch(Circle((0, 0), r_mean, fill=False, ls=(0, (5, 4)),
                        lw=1.0, ec=INK_MUTED, zorder=1))
    ax.annotate(f"mean radius {r_mean:.2f} mm about the fitted axis",
                xy=(0, r_mean), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, color=INK_MUTED, zorder=6)

    # --- the fitted axis, and where a free circle fit would put the centre
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.plot(0, 0, marker="+", ms=11, mew=1.4, color=INK_2, zorder=6)
    ax.annotate("fitted axis", xy=(0, 0), xytext=(8, -13),
                textcoords="offset points", fontsize=8, color=INK_2, zorder=6)
    if np.hypot(cx, cy) > 0.05 * r_mean:
        ax.plot(cx, cy, marker="x", ms=8, mew=1.4, color=INK_MUTED, zorder=6)
        ax.annotate(f"free circle-fit centre\n({np.hypot(cx, cy):.2f} mm off)",
                    xy=(cx, cy), xytext=(8, 8), textcoords="offset points",
                    fontsize=7.5, color=INK_MUTED, zorder=6)

    # --- the observations, one series per tag id
    stats = []
    for k in ids:
        m = np.array([r["id"] == k for r in rows])
        xk, yk = X[m], Y[m]
        rk = np.hypot(xk, yk)
        ax.scatter(xk, yk, s=MARKER_PT, marker=marker[k], c=colour[k],
                   linewidths=0.9, edgecolors=SURFACE, zorder=4, alpha=0.95)
        ang = np.arctan2(yk, xk)
        ang_c = np.angle(np.mean(np.exp(1j * ang)))
        dang = np.angle(np.exp(1j * (ang - ang_c)))
        st = dict(id=k, n=int(len(xk)), r=float(rk.mean()),
                  radial=float(rk.std()),
                  tang=float(np.std(dang) * rk.mean()),
                  tang_deg=float(np.degrees(np.std(dang))),
                  span_deg=float(np.degrees(dang.max() - dang.min())))
        stats.append(st)
        # Direct labels only where the clusters actually separate. In the raw
        # plot every tag walks through the same camera-facing sector, so labels
        # would pile up -- there, identity rides on the legend + marker shape.
        if derotated:
            n = np.hypot(xk.mean(), yk.mean()) or 1.0
            off = r_mean * 0.19
            ax.annotate(f"{k}",
                        xy=(xk.mean() + xk.mean() / n * off,
                            yk.mean() + yk.mean() / n * off),
                        ha="center", va="center", fontsize=10,
                        fontweight="bold", color=INK, zorder=7)

    span = max(R.max(), r_mean) * (1.0 + PAD_FRAC)
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal", adjustable="box")   # a circle must LOOK like a circle
    ax.grid(True, color=GRID, lw=0.6, ls="-", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS_RULE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.8)
    ax.set_xlabel("e1  (mm, in the plane perpendicular to the axis)",
                  fontsize=9, color=INK_2)
    ax.set_ylabel("e2  (mm, in the plane perpendicular to the axis)",
                  fontsize=9, color=INK_2)

    # --- titles
    if derotated:
        what = "rotation UNDONE with manifest true_deg"
        expect = ("each tag should collapse to a single dot; a smear ALONG the "
                  "circle means the angles are wrong")
    else:
        what = "NO rotation applied"
        expect = ("every tag walks the same camera-facing sector; they should "
                  "all sit on one circular arc of constant radius")
    fig.text(0.075, 0.950, f"{ring_name.capitalize()} ring — {what}",
             fontsize=15, color=INK, fontweight="bold", ha="left", va="center")
    fig.text(0.075, 0.906, f"Tag centres projected onto the plane "
             f"perpendicular to the turntable axis",
             fontsize=9.5, color=INK_2, ha="left", va="center")
    fig.text(0.075, 0.878, expect, fontsize=8.5, color=INK_MUTED,
             ha="left", va="center")

    # --- right-hand panel: colour swatch + tag id + per-tag numbers
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    y = 0.972
    panel.text(0.0, y, "TAG ID", fontsize=9.5, color=INK, fontweight="bold",
               va="top", ha="left")
    y -= 0.040
    hdr = "radius radial  tang" if derotated else "radius radial   arc"
    panel.text(0.085, y, hdr, fontsize=7, color=INK_MUTED, va="top", ha="left",
               family="monospace")
    y -= 0.016
    panel.plot([0.0, 1.0], [y, y], color=GRID, lw=0.8)
    y -= 0.032

    for st in stats:
        k = st["id"]
        panel.add_patch(Rectangle((0.0, y - 0.024), 0.060, 0.028,
                                  fc=colour[k], ec="none", clip_on=False))
        panel.plot([0.030], [y - 0.010], marker=marker[k], ms=4.2,
                   color=SURFACE, mec=SURFACE, mew=0, clip_on=False)
        panel.text(0.082, y - 0.010, f"id {k}", fontsize=9.5, color=INK,
                   va="center", ha="left", fontweight="bold")
        panel.text(0.40, y - 0.010, f"n={st['n']}", fontsize=8,
                   color=INK_MUTED, va="center", ha="left", family="monospace")
        y -= 0.038
        third = f"{st['tang']:5.2f}" if derotated else f"{st['span_deg']:4.0f}°"
        panel.text(0.082, y - 0.006,
                   f"{st['r']:6.2f} {st['radial']:5.2f} {third}",
                   fontsize=7.5, color=INK_2, va="center", ha="left",
                   family="monospace")
        y -= 0.046

    y -= 0.008
    panel.plot([0.0, 1.0], [y, y], color=GRID, lw=0.8)
    y -= 0.038
    rad_all = float(np.sqrt(np.mean([s["radial"] ** 2 for s in stats])))
    rspread = max(s["r"] for s in stats) - min(s["r"] for s in stats)
    lines = [("mm unless marked", INK_MUTED, 7.0, False)]
    if derotated:
        tan_all = float(np.sqrt(np.mean([s["tang"] ** 2 for s in stats])))
        deg_all = float(np.sqrt(np.mean([s["tang_deg"] ** 2 for s in stats])))
        lines += [(f"radial RMS  {rad_all:5.2f}", INK, 8.5, True),
                  (f"tangential  {tan_all:5.2f}", INK, 8.5, True),
                  (f"  = {deg_all:5.2f} deg of", INK, 8.5, True),
                  ("    angle error", INK, 8.5, True),
                  ("", INK, 8.5, False),
                  ("tangential >> radial", INK_2, 7.0, False),
                  ("=> the ANGLES are wrong", INK_2, 7.0, False),
                  ("both similar => just", INK_2, 7.0, False),
                  ("   depth noise", INK_2, 7.0, False)]
    else:
        lines += [(f"radial RMS  {rad_all:5.2f}", INK, 8.5, True),
                  (f"circle fit  {fit_rms:5.2f}", INK, 8.5, True),
                  (f"radius rng  {rspread:5.2f}", INK, 8.5, True),
                  (f"centre off  {np.hypot(cx, cy):5.2f}", INK, 8.5, True),
                  ("", INK, 8.5, False),
                  ("circle fit ~ radial RMS", INK_2, 7.0, False),
                  ("=> it IS a circle", INK_2, 7.0, False),
                  ("centre off >> radial RMS", INK_2, 7.0, False),
                  ("=> axis is off-centre", INK_2, 7.0, False)]
    for txt, col, fs, mono in lines:
        panel.text(0.0, y, txt, fontsize=fs, color=col, va="top", ha="left",
                   family="monospace" if mono else None)
        y -= 0.034

    fig.text(0.075, 0.028, axis_note, fontsize=7.5, color=INK_MUTED,
             ha="left", va="center")

    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"[SAVE] {os.path.basename(out_path)}   "
          f"radial RMS {rad_all:.2f} mm" +
          (f", tangential {np.sqrt(np.mean([s['tang']**2 for s in stats])):.2f} mm"
           if derotated else f", circle-fit RMS {fit_rms:.2f} mm"))
    return stats


# ===========================================================================
# main
# ===========================================================================
def build_rows(obs, u, P, undo_sign, ring, derotate):
    e1, e2 = basis_from_axis(u)
    rows = []
    for t in obs:
        if t["ring"] != ring:
            continue
        X = t["center"]
        if derotate:
            T = rotation_about(P, u, undo_sign * t["true_deg"])
            X = T[:3, :3] @ X + T[:3, 3]
        d = X - P
        rows.append(dict(id=t["id"], x_mm=float(d @ e1) * 1000.0,
                         y_mm=float(d @ e2) * 1000.0))
    return rows


def main():
    out_dir = os.path.join(INPUT_DIR, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    obs = gather()
    top = [t for t in obs if t["ring"] == "top"]
    bot = [t for t in obs if t["ring"] == "bottom"]
    print(f"\n[TAGS] {len(obs)} observations ({len(top)} top, {len(bot)} bottom)")
    if len(top) < 4 or len(bot) < 4:
        raise SystemExit("not enough observations on one of the rings")

    if AXIS_SOURCE == "manual":
        u = unit(MANUAL_AXIS_DIR)
        P = np.asarray(MANUAL_AXIS_POINT, dtype=np.float64)
        note = "axis: MANUAL (config)"
    else:
        u_seed = unit(np.stack([t["u"] for t in obs]).mean(0))
        u, P = solve_two_ring_axis(top, bot, u_seed)
        note = "axis: two-ring fit from the tags"
    tilt = float(np.degrees(np.arccos(np.clip(u @ np.array([0, -1.0, 0]), -1, 1))))
    print(f"[AXIS] u = {u.round(6)}   tilt from camera-up {tilt:.3f} deg")
    print(f"[AXIS] P = {P.round(6)}")
    axis_note = (f"{note}   u = [{u[0]:+.4f}, {u[1]:+.4f}, {u[2]:+.4f}]   "
                 f"P = [{P[0]:+.4f}, {P[1]:+.4f}, {P[2]:+.4f}] m   "
                 f"tilt {tilt:.3f}°")

    print()
    undo_sign = choose_sign(obs, u, P)
    print()

    for ring in ("top", "bottom"):
        for derot in (False, True):
            rows = build_rows(obs, u, P, undo_sign, ring, derot)
            name = f"ring_{ring}_{'derotated' if derot else 'raw'}.png"
            ring_figure(rows, ring, derot, os.path.join(out_dir, name),
                        axis_note)

    print(f"\n[DONE] four figures in {out_dir}")


if __name__ == "__main__":
    main()