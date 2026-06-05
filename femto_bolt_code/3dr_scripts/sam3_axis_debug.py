import configparser
import glob
import json
import os
import re

import cv2
import numpy as np
from pupil_apriltags import Detector

INPUT_DIR = "../captures/manual_rotation_test/"

INTRINSICS_FILE = "../scripts/calibration_parameters/CameraParam_Femto BoltCL8956200ZM_Color1280x720_Depth640x576.ini"
INTRINSIC_SECTION = "ColorIntrinsic"
DISTORTION_SECTION = "ColorDistortion"
UNDISTORT_RAYS = True

RGB_SUBDIR = "rgb"
DEPTH_SUBDIR = "depth"
OUT_SUBDIR = "aligned"
DEBUG_SUBDIR = "debug"
COLOR_GLOB = "color_*.png"
DEPTH_EXTS = (".npy", ".png")

DEPTH_SCALE_M = 0.001
DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 1.50

TAG_FAMILY = "tag36h11"
TAG_IDS = None
TAG_SIZE_M = 0.028  #confirm this omg
REF_ANGLE_DEG = 0

AXIS_TAG_ID = 10
TAG_TO_AXIS_M = 0.045
AXIS_DEPTH_SOURCE = "tag"
AXIS_LATERAL_SOURCE = "circle"
DEPTH_WIN_PX = 5
MAX_TILT_DEG = 25.0

PINK_HSV_RANGES = (((140, 60, 50), (179, 255, 255)),
                   ((0, 60, 50), (10, 255, 255)))
PINK_OPEN_PX = 3
LAT_PCTL = 1.0
MIN_POT_POINTS = 500

AXIS_HALF_LEN_M = 0.12
AXIS_SAMPLES = 40
DRAW_THICK = 2


def load_intrinsics(path):
    cfg = configparser.ConfigParser()
    with open(path) as f:
        cfg.read_string(f.read())
    i, d = cfg[INTRINSIC_SECTION], cfg[DISTORTION_SECTION]
    K = np.array([[float(i["fx"]), 0.0, float(i["cx"])],
                  [0.0, float(i["fy"]), float(i["cy"])],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([float(d["k1"]), float(d["k2"]), float(d["p1"]), float(d["p2"]),
                     float(d["k3"]), float(d["k4"]), float(d["k5"]), float(d["k6"])], dtype=np.float64)
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
    return ((xs - Ks[0, 2]) / Ks[0, 0]).astype(np.float32), ((ys - Ks[1, 2]) / Ks[1, 1]).astype(np.float32)


def project(pts3d, Ks, dist):
    p = np.asarray(pts3d, dtype=np.float64).reshape(-1, 1, 3)
    uv, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), Ks, dist)
    return uv.reshape(-1, 2)


def find_depth(folder, suffix):
    for ext in DEPTH_EXTS:
        path = os.path.join(folder, f"depth_{suffix}{ext}")
        if os.path.exists(path):
            return (np.load(path) if ext == ".npy" else cv2.imread(path, cv2.IMREAD_UNCHANGED))
    return None


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


def detect_tags(color_bgr, depth, xn, yn, detector):
    s = TAG_SIZE_M / 2.0
    canon = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]])
    up_cam = np.array([0.0, -1.0, 0.0])
    out = []
    for det in detector.detect(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)):
        if TAG_IDS is not None and det.tag_id not in TAG_IDS:
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
        u_hat = u_hat / np.linalg.norm(u_hat)
        tilt = float(np.degrees(np.arccos(np.clip(u_hat @ up_cam, -1, 1))))
        out.append(dict(id=int(det.tag_id), center=unproject(xn, yn, cu, cv_, zc),
                        u=u_hat, tilt=tilt, rms=fit_rms, corners=np.asarray(det.corners)))
    return out


def pot_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in PINK_HSV_RANGES:
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    if PINK_OPEN_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PINK_OPEN_PX + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    return m.astype(bool)


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
    return c, radii, float(np.sqrt(np.mean(resid ** 2)))


def polyline(img, uv, color):
    pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, color, DRAW_THICK, cv2.LINE_AA)


def main():
    rgb_dir = os.path.join(INPUT_DIR, RGB_SUBDIR)
    depth_dir = os.path.join(INPUT_DIR, DEPTH_SUBDIR)
    out_dir = os.path.join(INPUT_DIR, OUT_SUBDIR)
    dbg_dir = os.path.join(out_dir, DEBUG_SUBDIR)
    os.makedirs(dbg_dir, exist_ok=True)

    K, dist, src_w, src_h = load_intrinsics(INTRINSICS_FILE)

    frames = []
    for path in sorted(glob.glob(os.path.join(rgb_dir, COLOR_GLOB))):
        stem = os.path.splitext(os.path.basename(path))[0]
        suffix = stem.split("color_", 1)[-1]
        m = re.search(r"(\d{3})$", stem)
        if m:
            frames.append((int(m.group(1)) % 360, suffix, path))
    frames.sort()
    if not frames:
        raise SystemExit(f"no angle-tagged frames in {rgb_dir}")
    print(f"[LOAD] {len(frames)} frames | filename angles: "
          f"{[f[0] for f in frames[:6]]}{' ...' if len(frames) > 6 else ''}")

    probe = cv2.imread(frames[0][2], cv2.IMREAD_COLOR)
    h, w = probe.shape[:2]
    Ks = scale_K(K, (src_w, src_h), (w, h))
    xn, yn = ray_map(Ks, dist, w, h)
    detector = Detector(families=TAG_FAMILY)

    obs = []
    for angle, suffix, _ in frames:
        raw = cv2.imread(os.path.join(INPUT_DIR, f"color_{suffix}.png"), cv2.IMREAD_COLOR)
        raw_depth = find_depth(INPUT_DIR, suffix)
        if raw is None or raw_depth is None:
            print(f"[SKIP] missing unmasked frame {suffix}")
            continue
        tags = detect_tags(raw, raw_depth, xn, yn, detector)
        good = [t for t in tags if t["tilt"] < MAX_TILT_DEG]
        obs.append(dict(angle=angle, suffix=suffix, tags=good))
        ids = sorted({t["id"] for t in good})
        tilts = ", ".join("%.1f" % t["tilt"] for t in good)
        rmss = ", ".join("%.1f" % (t["rms"] * 1000.0) for t in good)
        print(f"[TAGS] {angle:03d}  ids={ids}  tilt_deg=[{tilts}]  corner_rms_mm=[{rmss}]")

    all_tags = [t for o in obs for t in o["tags"]]
    if len(all_tags) < 6:
        raise SystemExit("too few tag observations to fit the rotation circle")

    U = np.stack([t["u"] for t in all_tags])
    u_hat = U.mean(0)
    u_hat /= np.linalg.norm(u_hat)
    spread = np.degrees(np.arccos(np.clip(U @ u_hat, -1, 1)))
    up_cam = np.array([0.0, -1.0, 0.0])
    print(f"\n[VERT] averaged over {len(U)} tag observations")
    print(f"[VERT] u_hat = {u_hat.round(4)}   tilt from camera-up = "
          f"{np.degrees(np.arccos(np.clip(u_hat @ up_cam, -1, 1))):.2f} deg")
    print(f"[VERT] per-observation spread: median {np.median(spread):.2f} deg, "
          f"p90 {np.percentile(spread, 90):.2f} deg  <-- single-frame axis noise")

    e1 = np.cross(u_hat, np.array([0.0, 0.0, 1.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u_hat, e1)
    pts2d = np.array([[t["center"] @ e1, t["center"] @ e2] for t in all_tags])
    ids = [t["id"] for t in all_tags]
    c2, radii, circ_rms = fit_shared_circle(pts2d, ids)
    print(f"\n[CIRC] shared-centre fit over {len(pts2d)} tag centres")
    radii_cm = ", ".join("id%d=%.2f" % (k, v * 100.0) for k, v in sorted(radii.items()))
    print(f"[CIRC] radii by tag id (cm): {radii_cm}")
    print(f"[CIRC] residual RMS = {circ_rms*1000:.2f} mm  <-- how circular the motion really is")

    anchor = float(np.mean([t["center"] @ u_hat for t in all_tags]))
    P_circ = c2[0] * e1 + c2[1] * e2 + anchor * u_hat
    print(f"[CIRC] axis point = {P_circ.round(4)}  dir = {u_hat.round(4)}")

    ref_obs = next((o for o in obs if o["angle"] == REF_ANGLE_DEG), None)
    ref_tag = None
    if ref_obs and ref_obs["tags"]:
        ref_tag = next((t for t in ref_obs["tags"] if t["id"] == AXIS_TAG_ID), None)
        if ref_tag is None:
            ref_tag = ref_obs["tags"][0]
            print(f"[WARN] tag {AXIS_TAG_ID} absent at {REF_ANGLE_DEG:03d}; "
                  f"using id {ref_tag['id']} instead")

    axis_z_tag = None
    if ref_tag is not None:
        tag_z = float(ref_tag["center"][2])
        axis_z_tag = tag_z + TAG_TO_AXIS_M
        implied = float(P_circ[2]) - tag_z
        print(f"\n[DEPTH] tag id{ref_tag['id']} at {REF_ANGLE_DEG:03d}: camera z = {tag_z:.4f} m")
        print(f"[DEPTH] + TAG_TO_AXIS_M ({TAG_TO_AXIS_M*100:.2f} cm) -> axis z = {axis_z_tag:.4f} m")
        print(f"[DEPTH] circle-fit axis z = {P_circ[2]:.4f} m  "
              f"(delta = {(axis_z_tag - P_circ[2])*1000:+.1f} mm)")
        print(f"[DEPTH] offset implied by the circle fit = {implied*100:.2f} cm  "
              f"<-- compare against your measured 4.5 cm")

    if AXIS_DEPTH_SOURCE == "tag":
        if axis_z_tag is None:
            raise SystemExit(f"no tag at angle {REF_ANGLE_DEG:03d}; cannot use "
                             f"AXIS_DEPTH_SOURCE='tag'")
        dep = (axis_z_tag - anchor * u_hat[2]) / e2[2]
    else:
        dep = float(c2[1])

    if AXIS_LATERAL_SOURCE == "tag":
        if ref_tag is None:
            raise SystemExit(f"no tag at angle {REF_ANGLE_DEG:03d}; cannot use "
                             f"AXIS_LATERAL_SOURCE='tag'")
        lat = float(ref_tag["center"] @ e1)
    else:
        lat = float(c2[0])

    c_used = np.array([lat, dep])
    P = lat * e1 + dep * e2 + anchor * u_hat
    print(f"[AXIS] depth from '{AXIS_DEPTH_SOURCE}', lateral from '{AXIS_LATERAL_SOURCE}'")
    print(f"[AXIS] axis point = {P.round(4)}  camera z = {P[2]:.4f} m  "
          f"dir = {u_hat.round(4)}")

    def circ_angle(p3):
        q = np.array([p3 @ e1, p3 @ e2]) - c_used
        return float(np.arctan2(q[1], q[0]))

    def cmean(a):
        return float(np.angle(np.mean(np.exp(1j * np.asarray(a)))))

    theta = {o["angle"]: [(t["id"], circ_angle(t["center"])) for t in o["tags"]]
             for o in obs if o["tags"]}
    if REF_ANGLE_DEG not in theta:
        print(f"[WARN] reference angle {REF_ANGLE_DEG:03d} has no usable tag")

    report = {}
    for s_try in (1, -1):
        beta, resid = {}, []
        for k in {i for v in theta.values() for i, _ in v}:
            beta[k] = cmean([th - s_try * np.deg2rad(a)
                             for a, v in theta.items() for i, th in v if i == k])
        for a, v in theta.items():
            for i, th in v:
                resid.append(np.angle(np.exp(1j * (th - beta[i] - s_try * np.deg2rad(a)))))
        report[s_try] = (beta, float(np.degrees(np.sqrt(np.mean(np.square(resid))))))
        print(f"[SIGN] angle_sign={s_try:+d}  residual RMS={report[s_try][1]:7.2f} deg")
    best = min(report, key=lambda k: report[k][1])
    beta = report[best][0]
    print(f"[SIGN] -> measured rotation matches ANGLE_SIGN = {-best:+d}")

    phi, undo = {}, {}
    for o in obs:
        a = o["angle"]
        if a not in theta:
            continue
        phi[a] = cmean([th - beta[i] for i, th in theta[a]])
        undo[o["suffix"]] = float(-np.degrees(phi[a]))
    ref_phi = phi.get(REF_ANGLE_DEG, 0.0)
    for o in obs:
        if o["suffix"] in undo:
            undo[o["suffix"]] += float(np.degrees(ref_phi))

    print(f"[ANGL] measured rotation vs filename ({len(undo)} frames with tags)")
    slip = []
    for a in sorted(phi):
        nominal = best * np.deg2rad(a) - ref_phi + phi[a] * 0
        d = np.degrees(np.angle(np.exp(1j * (phi[a] - ref_phi - best * np.deg2rad(a)))))
        slip.append(d)
        print(f"   file={a:03d}  measured={(np.degrees(phi[a] - ref_phi) * best) % 360:7.2f}  "
              f"slip={d:+7.2f} deg")
    if slip:
        print(f"[ANGL] hand-rotation slip: mean|.|={np.mean(np.abs(slip)):.2f} deg  "
              f"max|.|={np.max(np.abs(slip)):.2f} deg")

    axis_pts = np.stack([P + u_hat * t for t in
                         np.linspace(-AXIS_HALF_LEN_M, AXIS_HALF_LEN_M, AXIS_SAMPLES)])
    axis_uv = project(axis_pts, Ks, dist)

    for angle, suffix, rgb_path in frames:
        raw = cv2.imread(os.path.join(INPUT_DIR, f"color_{suffix}.png"), cv2.IMREAD_COLOR)
        rgbm = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        depth = find_depth(depth_dir, suffix)
        if raw is None or rgbm is None or depth is None:
            continue
        vis = raw.copy()

        mask = pot_mask(cv2.cvtColor(rgbm, cv2.COLOR_BGR2RGB))
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 0, 255), 1)

        z = depth.astype(np.float32) * DEPTH_SCALE_M
        valid = mask & (z > DEPTH_MIN_M) & (z < DEPTH_MAX_M)
        ys, xs = np.nonzero(valid)
        h_hat = np.cross(u_hat, np.array([0.0, 0.0, 1.0]))
        h_hat /= np.linalg.norm(h_hat)
        if len(xs) >= MIN_POT_POINTS:
            zz = z[ys, xs]
            pts = np.stack([xn[ys, xs] * zz, yn[ys, xs] * zz, zz], axis=1)
            hh = pts @ h_hat
            lo, hi = np.percentile(hh, [LAT_PCTL, 100.0 - LAT_PCTL])
            mid = 0.5 * (lo + hi)
            dm = float(np.median(pts @ np.cross(h_hat, u_hat)))
            base = mid * h_hat + float(np.median(pts @ u_hat)) * u_hat + dm * np.cross(h_hat, u_hat)
            mid_line = np.stack([base + u_hat * t for t in
                                 np.linspace(-AXIS_HALF_LEN_M, AXIS_HALF_LEN_M, AXIS_SAMPLES)])
            polyline(vis, project(mid_line, Ks, dist), (0, 200, 255))
            for val, col in ((lo, (255, 200, 0)), (hi, (255, 200, 0))):
                edge = val * h_hat + float(np.median(pts @ u_hat)) * u_hat + dm * np.cross(h_hat, u_hat)
                seg = np.stack([edge + u_hat * t for t in (-0.05, 0.05)])
                polyline(vis, project(seg, Ks, dist), col)
            cv2.putText(vis, f"pot mid {mid*100:+.2f} cm", (12, h - 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        polyline(vis, axis_uv, (0, 0, 255))

        o = next((x for x in obs if x["suffix"] == suffix), None)
        if o:
            for t in o["tags"]:
                cv2.polylines(vis, [np.round(t["corners"]).astype(np.int32).reshape(-1, 1, 2)],
                              True, (0, 255, 0), 2, cv2.LINE_AA)
                seg = np.stack([t["center"], t["center"] + t["u"] * 0.06])
                polyline(vis, project(seg, Ks, dist), (0, 255, 0))
                cu, cv_ = np.round(project([t["center"]], Ks, dist)[0]).astype(int)
                cv2.putText(vis, f"id{t['id']} tilt{t['tilt']:.1f}", (cu + 6, cv_ - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(vis, f"angle {angle:03d}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "red=fitted axis  yellow=pot mid  green=tag vertical", (12, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(dbg_dir, f"axis_{angle:03d}_{suffix}.png"), vis)

    with open(os.path.join(out_dir, "axis_refined.json"), "w") as f:
        json.dump({"axis_point": P.tolist(), "axis_dir": u_hat.tolist(),
                   "angle_sign": int(-best), "circle_rms_m": circ_rms,
                   "radii_m": {str(k): v for k, v in radii.items()},
                   "vertical_spread_deg_median": float(np.median(spread)),
                   "axis_depth_source": AXIS_DEPTH_SOURCE,
                   "axis_lateral_source": AXIS_LATERAL_SOURCE,
                   "tag_to_axis_m": TAG_TO_AXIS_M,
                   "axis_z_tag": axis_z_tag,
                   "axis_z_circle": float(P_circ[2]),
                   "undo_angle_deg": undo}, f, indent=2)
    print(f"\n[SAVE] {dbg_dir}/axis_*.png")
    print(f"[SAVE] {out_dir}/axis_refined.json")


if __name__ == "__main__":
    main()