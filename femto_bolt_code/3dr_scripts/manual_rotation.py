#!/usr/bin/env python3
"""
turntable_align_manual.py  --  turntable alignment from a hand-measured axis.

No AprilTags.  The rotation axis is supplied entirely by configuration:

    direction u : the camera's Y axis (optical +Y points DOWN, so the table
                  "up" direction is -Y).  Set AXIS_DIR_OVERRIDE to use a
                  different vector.
    point     P : BASE_CENTER_M, the measured (x, y, z) of the centre of the
                  base of the cylinder in the camera frame, in metres.

Only the component of P perpendicular to u matters -- sliding P along the
axis changes nothing -- so with u = camera Y the measured y value is ignored
by the math.  x and z are what count.

Rotation amounts come from the true stepper angles in manifest.csv.
Everything else (transforms, residuals, merge, JSON, Open3D viewer, axis
overlay on the colour frames) is unchanged from the tag-based script.

Filenames:  color_012_ang120.png  cloud_000_ang000.ply
"""

import configparser
import csv
import glob
import json
import os
import re

import numpy as np
import open3d as o3d

# ---------------------------------------------------------------------------
# THE AXIS  --  the only two things this script needs
# ---------------------------------------------------------------------------

BASE_CENTER_M = (-0.045, 0.1397, 0.45775)     # <-- (-1.75, 5.5, 18.0) inches in camera frame

# Axis direction.  "camera_y" = [0, -1, 0] (optical +Y is down, table up is -Y).
AXIS_DIR = "camera_y"
AXIS_DIR_OVERRIDE = None                  # e.g. (0.0, -0.999, 0.045)

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
INPUT_DIR = "../../flora_turntable/captures/yellow_pot_09-08_deg_10_speed_5"

COLOR_DIR = os.path.join(INPUT_DIR, "rgb")
CLOUD_DIR = os.path.join(INPUT_DIR, "pointcloud")

OUT_SUBDIR = "aligned_manual"
DEBUG_SUBDIR = "debug"

MANIFEST_FILE = os.path.join(INPUT_DIR, "manifest.csv")
MANIFEST_INDEX_COL = "index"
MANIFEST_TRUE_COL = "true_deg"
MANIFEST_COLOR_COL = "color"
REQUIRE_MANIFEST = True       # False = fall back to the filename angle

COLOR_PREFIX = "color_"
CLOUD_PREFIX = "cloud_"
COLOR_EXT = ".png"
CLOUD_EXT = ".ply"

# ---------------------------------------------------------------------------
# angles
# ---------------------------------------------------------------------------
ANGLE_INCREMENT_DEG = 10.0    # label fallback only
FORCE_INCREMENT_ANGLE = False
ANGLE_SIGN = "-1"           # "auto" | +1 | -1
SIGN_TEST_STEP_DEG = 30       # pair spacing used by the auto sign test
SIGN_TEST_MAX_PTS = 2000
REF_ANGLE_DEG = 0

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

RESIDUAL_MODE = "both"        # "neighbour" | "ref" | "both" | "none"

# ---------------------------------------------------------------------------
# viewer
# ---------------------------------------------------------------------------
VIEW_APPROACH_NAME = "manual"
VIEW_ANGLES = None            # e.g. (0, 90, 180, 270); None = every loaded angle
VIEW_PER_ANGLE = True
VIEW_TRUE_COLOR = True
VIEW_VOXEL_M = 0.0
VIEW_STEP_DEG = 90
VIEW_SHOW_FRAMES = False

# ---------------------------------------------------------------------------
# axis overlay on the colour frames (needs the intrinsics file)
# ---------------------------------------------------------------------------
WRITE_DEBUG_IMAGES = True
INTRINSICS_FILE = ("../scripts/calibration_parameters/"
                   "CameraParam_Femto_BoltCL8956200ZM_Color1280x720_Depth640x576.ini")
INTRINSIC_SECTION = "ColorIntrinsic"
DISTORTION_SECTION = "ColorDistortion"
AXIS_DRAW_HALF_LEN_M = 0.12
AXIS_SAMPLES = 40
DRAW_THICK = 2

PALETTE = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.70, 0.20], [0.15, 0.35, 0.95], [1.00, 0.75, 0.00],
    [0.85, 0.15, 0.85], [0.00, 0.80, 0.80], [1.00, 0.45, 0.00], [0.55, 0.85, 0.55],
])

NAME_RE = re.compile(r"^(?P<idx>\d+)(?:_ang(?P<ang>\d+))?$")


# ===========================================================================
# geometry
# ===========================================================================
def unit(v):
    return np.asarray(v, dtype=np.float64) / np.linalg.norm(v)


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


def build_axis():
    if AXIS_DIR_OVERRIDE is not None:
        u = unit(AXIS_DIR_OVERRIDE)
        src = "override"
    elif AXIS_DIR == "camera_y":
        u = np.array([0.0, -1.0, 0.0])
        src = "camera -Y"
    else:
        raise SystemExit(f"unknown AXIS_DIR '{AXIS_DIR}'")
    P = np.asarray(BASE_CENTER_M, dtype=np.float64)
    return u, P, src


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
        print(f"[MANI] no manifest at {path} -- using filename angles")
        return by_index, by_color
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get(MANIFEST_TRUE_COL):
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


# ===========================================================================
# rotation sign, decided on the clouds
# ===========================================================================
def choose_sign(raw, angles, true_ang, u, P):
    if ANGLE_SIGN in (1, -1):
        print(f"[SIGN] forced by config: undo sign = {int(ANGLE_SIGN):+d}")
        return int(ANGLE_SIGN)

    aset = set(angles)
    pairs = [(a, (a + SIGN_TEST_STEP_DEG) % 360) for a in angles
             if (a + SIGN_TEST_STEP_DEG) % 360 in aset]
    if not pairs:
        print("[SIGN] no usable pairs -- defaulting to undo sign = -1")
        return -1

    rng = np.random.default_rng(0)
    samp = {}
    for a in {x for p in pairs for x in p}:
        X = np.asarray(raw[a][1].points)
        if len(X) > SIGN_TEST_MAX_PTS:
            X = X[rng.choice(len(X), SIGN_TEST_MAX_PTS, replace=False)]
        samp[a] = np.ascontiguousarray(X)

    def cost(s):
        vals = []
        for a, b in pairs:
            Ta = rotation_about(P, u, s * true_ang[a])
            Tb = rotation_about(P, u, s * true_ang[b])
            A = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                samp[a] @ Ta[:3, :3].T + Ta[:3, 3]))
            B = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
                samp[b] @ Tb[:3, :3].T + Tb[:3, 3]))
            vals.append(np.median(np.asarray(A.compute_point_cloud_distance(B))))
        return float(np.mean(vals))

    scores = {s: cost(s) for s in (1, -1)}
    for s in (1, -1):
        print(f"[SIGN] undo sign {s:+d}: median pair distance = {scores[s]*1000:7.2f} mm "
              f"({len(pairs)} pairs at {SIGN_TEST_STEP_DEG} deg)")
    best = min(scores, key=scores.get)
    print(f"[SIGN] chosen undo sign = {best:+d}")
    return best


# ===========================================================================
# optional axis overlay on the colour frames
# ===========================================================================
def draw_axis_overlays(u, P, out_dir, true_ang):
    import cv2

    if not os.path.isfile(INTRINSICS_FILE):
        print(f"[DBG ] intrinsics not found ({INTRINSICS_FILE}) -- overlay skipped")
        return
    cfg = configparser.ConfigParser()
    with open(INTRINSICS_FILE) as f:
        cfg.read_string(f.read())
    i, d = cfg[INTRINSIC_SECTION], cfg[DISTORTION_SECTION]
    K = np.array([[float(i["fx"]), 0.0, float(i["cx"])],
                  [0.0, float(i["fy"]), float(i["cy"])],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([float(d["k1"]), float(d["k2"]), float(d["p1"]), float(d["p2"]),
                     float(d["k3"]), float(d["k4"]), float(d["k5"]), float(d["k6"])])
    src_w, src_h = int(i["width"]), int(i["height"])

    colors = scan(COLOR_DIR, COLOR_PREFIX, (COLOR_EXT,))
    if not colors:
        print(f"[DBG ] no colour frames in {COLOR_DIR} -- overlay skipped")
        return

    probe = cv2.imread(colors[sorted(colors)[0]][2], cv2.IMREAD_COLOR)
    if probe is None:
        print("[DBG ] cannot read a colour frame -- overlay skipped")
        return
    h_img, w_img = probe.shape[:2]
    sx, sy = w_img / src_w, h_img / src_h
    Ks = K.copy()
    Ks[0, 0] *= sx
    Ks[0, 2] *= sx
    Ks[1, 1] *= sy
    Ks[1, 2] *= sy

    def project(pts):
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), Ks, dist)
        return uv.reshape(-1, 2)

    axis_pts = np.stack([P + u * t for t in np.linspace(
        -AXIS_DRAW_HALF_LEN_M, AXIS_DRAW_HALF_LEN_M, AXIS_SAMPLES)])
    axis_uv = project(axis_pts)
    base_uv = project([P])[0]

    for idx in sorted(colors):
        angle, suffix, path = colors[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        cv2.polylines(img, [np.round(axis_uv).astype(np.int32).reshape(-1, 1, 2)],
                      False, (0, 0, 255), DRAW_THICK, cv2.LINE_AA)
        bu, bv = np.round(base_uv).astype(int)
        cv2.circle(img, (int(bu), int(bv)), 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, "base centre", (int(bu) + 10, int(bv) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        td = true_ang.get(angle, float(angle))
        cv2.putText(img, f"angle {angle:03d}  true {td:.3f}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"axis_{angle:03d}_{suffix}.png"), img)
    print(f"[SAVE] {out_dir}/axis_*.png  "
          f"(check the red line runs up the middle of the cylinder)")


# ===========================================================================
# viewer -- per-angle toggles
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
                          is_visible=True))
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
        items, title="turntable alignment (manual axis)", show_ui=True,
        point_size=POINT_SIZE,
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

    u_axis, P_axis, dir_src = build_axis()
    print("=" * 70)
    print(f"  AXIS FROM MANUAL MEASUREMENT  (direction: {dir_src})")
    print(f"    u = [{u_axis[0]: .6f}, {u_axis[1]: .6f}, {u_axis[2]: .6f}]")
    print(f"    P = [{P_axis[0]: .6f}, {P_axis[1]: .6f}, {P_axis[2]: .6f}]  "
          "(measured base centre, m)")
    perp = P_axis - (P_axis @ u_axis) * u_axis
    print(f"    only the perpendicular part of P is used: "
          f"[{perp[0]: .6f}, {perp[1]: .6f}, {perp[2]: .6f}]")
    print("=" * 70)

    mani_idx, mani_color = load_manifest(MANIFEST_FILE)

    plys = scan(CLOUD_DIR, CLOUD_PREFIX, (CLOUD_EXT,))
    if not plys:
        raise SystemExit(f"no {CLOUD_PREFIX}*{CLOUD_EXT} in {CLOUD_DIR}")

    raw, true_ang, ang_src = {}, {}, {}
    for idx in sorted(plys):
        angle, suffix, path = plys[idx]
        if LOAD_ANGLES is not None and angle not in LOAD_ANGLES:
            continue
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
    dev = [true_ang[a] - a for a in angles]
    print(f"[MANI] true - commanded: mean {np.mean(dev):+.4f} deg  "
          f"max|.| {np.max(np.abs(dev)):.4f} deg")

    print()
    undo_sign = choose_sign(raw, angles, true_ang, u_axis, P_axis)

    clouds, shot_frames, transforms = {}, {}, {}
    print()
    for a in angles:
        suffix, pcd = raw[a]
        deg = undo_sign * true_ang[a]
        T = rotation_about(P_axis, u_axis, deg)
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

    if WRITE_DEBUG_IMAGES:
        print()
        draw_axis_overlays(u_axis, P_axis, dbg_dir, true_ang)

    a0 = P_axis - u_axis * AXIS_HALF_LEN_M
    a1 = P_axis + u_axis * AXIS_HALF_LEN_M
    axis_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.stack([a0, a1])),
        lines=o3d.utility.Vector2iVector([[0, 1]]))
    axis_ls.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])
    world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    world.translate(P_axis)
    statics = [("rotation_axis", axis_ls), ("axis_origin_P", world)]

    if SAVE_MERGED:
        o3d.io.write_point_cloud(os.path.join(out_dir, "merged.ply"), merged)
        with open(os.path.join(out_dir, "axis_manual.json"), "w") as f:
            json.dump(dict(
                axis_dir=u_axis.tolist(), axis_point=P_axis.tolist(),
                axis_method="manual", axis_dir_source=dir_src,
                base_center_m=list(BASE_CENTER_M),
                undo_sign=int(undo_sign), angle_source=MANIFEST_TRUE_COL,
                manifest=os.path.abspath(MANIFEST_FILE),
                frames=transforms), f, indent=2)
        print(f"[SAVE] {out_dir}/merged.ply")
        print(f"[SAVE] {out_dir}/axis_manual.json")

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
            pc.transform(rotation_about(P_axis, u_axis, undo_sign * true_ang[a]))
            if not VIEW_TRUE_COLOR or not pc.has_colors():
                pc.paint_uniform_color(PALETTE[i % len(PALETTE)])
            if VIEW_VOXEL_M > 0:
                pc = pc.voxel_down_sample(VIEW_VOXEL_M)
            per_angle[a] = pc

        comparison_viewer({VIEW_APPROACH_NAME: per_angle}, view_angles,
                          statics, shot_frames)


if __name__ == "__main__":
    main()