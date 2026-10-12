#!/usr/bin/env python3
"""
tag_stitch.py  --  fiducial-anchored fusion of a turntable sweep.

(The earlier single-tag, 4-corner Kabsch viewer lives on as tag_stitch_single_tag.py.)

Per adjacent pair (shot i, shot i+1), and the closing pair (last, 0) on a full loop:

1. Preprocess   averaged depth + weight + validity (average_confidence.py) ->
                cloud in the shot's COLOUR-camera frame; drop invalid and
                low-weight points; crop to --roi-radius around the visible tags
                (static background must not reach ICP); weighted voxel grid;
                normals on that cleaned cloud, oriented toward the camera.
2. Coarse       AprilTags in both colour images. Default (--corner-depth pnp): each
                tag's 3D corners come from colour-only PnP with the known
                --tag-size-mm, because ToF depth on paper/plastic tag cards carries a
                view-dependent bias of several mm, which is large next to the ~10 mm
                a tag moves per 10 deg step. Alternatives: "plane" (colour rays on a
                weighted RANSAC plane fitted to the tag's depth points) and "local".
                Tags are dropped if seen beyond --max-obliquity, or (plane mode) if
                their 3D sides disagree with --tag-size-mm. Shared tags -> corner
                correspondences (>= --min-corr, 16 by default) -> Kabsch on the
                largest set of mutually consistent tags (minimal-subset consensus).
                --use-tag-centers uses one plane-centre per tag instead (>= 3 tags).
3. Refine       point-to-plane ICP started from step 2: weights = w_src * w_tgt *
                Huber(residual); restricted to the two-way overlap under the tag
                pose; per-iteration distance gate (shrinking) + normal-angle gate.
4. Quality      T_i->i+1, weighted ICP residual, overlap, eigenvalues + condition
                number of the (centred, scale-normalised) 6x6 point-to-plane normal
                matrix, tag residual, ICP-vs-tag correction -> quality flags.
5. Compose      open sweep: chain the pair transforms into shot 0's frame.
                closed loop: Open3D multiway pose-graph optimisation over the
                odometry edges + loop-closure edge (Open3D's multiway-registration
                recipe), instead of chaining.

A pair with too few shared tags (< --min-corr) is not given up on by default:
the turntable steps are equal, so the nearest tag-initialised edge's transform
seeds ICP instead. Such an edge is kept only if ICP converges and passes the
residual + overlap gates, and it is flagged no_tag_init / init_from_edge_*.
Disable with --no-neighbour-init.

If ICP moves every edge the same way (icp_large_correction everywhere) and the
loop-closure error is large while the tag-only loop closes, the depth surface
itself is biased (seen on a smooth yellow pot: ~12.3 deg/step from depth vs the
true 10). Trust the tags there: --no-icp.

Anything STATIC inside the ROI that is not rotationally symmetric about the
turntable axis (a faceted base, a clamp, cables) pulls ICP toward identity and
shows up as a same-sign rotation bias on every edge and a large loop-closure
error. Shrink --roi-radius, or mask it out, if you see that.

    python tag_stitch.py captures/plant_01
    python tag_stitch.py captures/plant_01 --use-tag-centers --loop yes --view
    python tag_stitch.py captures/plant_01 --exclude-tags 15 --out stitch_holdout15
    python tag_stitch.py captures/plant_01 --indices 0-8 --no-icp
    python tag_stitch.py captures/plant_01 --mask          # after sam3_batch_masks.py

I/O contract
------------
in   <capture>/camera_params.json   sweep_capture.py: native depth + colour
                                    intrinsics/distortion, depth->colour extrinsic
     <capture>/manifest.csv         columns index, true_deg, color, stop_dir
     <capture>/<color>              colour PNG per stop (AprilTags are found here)
     <capture>/<stop_dir>/avg/      average_confidence.py: depth_mean_mm.npy,
                                    weight.npy, valid_mask.png (native depth grid)
     <capture>/<stop_dir>/mask.png  only with --mask: sam3_batch_masks.py pot + plant
                                    mask, colour resolution. Applied AFTER the tags are
                                    measured, to the ICP cloud and fused.ply only.

out  <capture>/<--out, default "stitch">/
     pairs.json        every edge: T_src_to_tgt (tag-only and final), tag stats,
                       ICP residual/overlap/eigenvalues/condition, flags
     pairs.csv         one-line-per-edge summary of the above
     poses.json        T_shot_to_ref per shot (4x4), method, loop-closure error
     tags_per_shot.json  every detected tag per shot, in that shot's colour frame:
                       plane corners, local-depth corners, plane centre, normal,
                       plane rms -- read by spread_analysis.py
     fused.ply         all shots in shot 0's frame, confidence-weighted voxel merge
     depth_correction.json  only with --refine-depth: the rigid depth->colour correction
                       fitted from the tags (depth-plane vs PnP corners), with its held-out
                       before/after numbers; reuse it with --depth-correction

Frames and units: metres. Each shot's frame is its COLOUR camera (x right,
y down, z forward). T_src_to_tgt maps src points into tgt:  X_tgt = T @ X_src.
T_shot_to_ref maps a shot into the reference (first selected) shot's frame.

Assumptions
-----------
* pupil_apriltags returns a tag's 4 corners in a fixed order tied to the tag's
  own frame, so the same corner index is the same physical corner in every view.
* The depth->colour extrinsic from the stream profile has t in millimetres
  (Orbbec SDK convention; same as fetch_intrinsics.py assumes).
* Distortion coefficients are in OpenCV order [k1 k2 p1 p2 k3 k4 k5 k6].
"""

import argparse
import csv
import json
import sys
from collections import deque
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from pupil_apriltags import Detector

UNDISTORT_CRIT = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)


# ===========================================================================
# small numeric helpers
# ===========================================================================
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def delta(Ta, Tb):
    """rotation (deg) and translation (mm) of Ta^-1 Tb."""
    D = np.linalg.inv(Ta) @ Tb
    return rot_deg(D[:3, :3]), float(np.linalg.norm(D[:3, 3]) * 1000)


def kabsch(P, Q):
    """R, t minimising sum |R P + t - Q|^2 (Umeyama without scale)."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - R @ pc


def parse_indices(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def parse_ids(text):
    return set(parse_indices(text)) if text else set()


# ===========================================================================
# camera + capture loading
# ===========================================================================
def _K(i):
    return np.array([[i["fx"], 0.0, i["cx"]], [0.0, i["fy"], i["cy"]], [0.0, 0.0, 1.0]])


def load_camera(capture):
    path = capture / "camera_params.json"
    if not path.is_file():
        sys.exit(f"missing {path} (written by sweep_capture.py)")
    c = json.loads(path.read_text())
    try:
        cam = dict(
            Kd=_K(c["depth_intrinsic"]), dist_d=np.array(c["depth_distortion"]["opencv_order"], float),
            d_wh=(c["depth_intrinsic"]["width"], c["depth_intrinsic"]["height"]),
            Kc=_K(c["color_intrinsic"]), dist_c=np.array(c["color_distortion"]["opencv_order"], float),
            c_wh=(c["color_intrinsic"]["width"], c["color_intrinsic"]["height"]),
            R_dc=np.array(c["depth_to_color"]["R"], float),
            t_dc=np.array(c["depth_to_color"]["t_mm"], float) / 1000.0)
    except KeyError as e:
        sys.exit(f"{path}: missing key {e}")
    # the SDK extrinsic is not always exactly orthonormal -> use the nearest rotation
    U, sv, Vt = np.linalg.svd(cam["R_dc"])
    if np.abs(sv - 1).max() > 1e-3:
        print(f"[CAM ] depth->colour R is not a pure rotation (singular values "
              f"{np.round(sv, 4).tolist()}); using the nearest rotation")
    cam["R_dc"] = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    w, h = cam["d_wh"]
    uv = np.stack(np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)), -1)
    cam["rays_d"] = rays(uv.reshape(-1, 2), cam["Kd"], cam["dist_d"]).reshape(h, w, 3)
    return cam


def rays(uv, K, dist):
    """pixels (N,2) -> undistorted rays (N,3) with z = 1."""
    xy = cv2.undistortPointsIter(np.asarray(uv, np.float64).reshape(-1, 1, 2), K, dist,
                                 None, None, UNDISTORT_CRIT).reshape(-1, 2)
    return np.column_stack([xy, np.ones(len(xy))])


def project(P, K, dist):
    uv, _ = cv2.projectPoints(np.asarray(P, np.float64).reshape(-1, 1, 3),
                              np.zeros(3), np.zeros(3), K, dist)
    return uv.reshape(-1, 2)


def load_manifest(capture, indices):
    path = capture / "manifest.csv"
    if not path.is_file():
        sys.exit(f"missing {path}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"index", "true_deg", "color", "stop_dir"}
    if not rows or not need <= set(rows[0]):
        sys.exit(f"{path} needs columns {sorted(need)} -- capture with the burst-capable "
                 f"sweep_capture.py and run average_confidence.py first")
    rows = [dict(index=int(r["index"]), true_deg=float(r["true_deg"]),
                 color=r["color"], stop_dir=r["stop_dir"]) for r in rows]
    rows.sort(key=lambda r: r["index"])
    if indices is not None:
        have = {r["index"] for r in rows}
        missing = sorted(set(indices) - have)
        if missing:
            sys.exit(f"--indices not in manifest: {missing}")
        rows = [r for r in rows if r["index"] in set(indices)]
    if len(rows) < 2:
        sys.exit("need at least 2 shots")
    for r in rows:
        avg = capture / r["stop_dir"] / "avg"
        for f in ("depth_mean_mm.npy", "weight.npy", "valid_mask.png"):
            if not (avg / f).is_file():
                sys.exit(f"missing {avg / f} -- run average_confidence.py on this capture")
        if not (capture / r["color"]).is_file():
            sys.exit(f"missing colour image {capture / r['color']}")
    return rows


# ===========================================================================
# step 1: cleaned, confidence-carrying cloud per shot (colour-camera frame)
# ===========================================================================
def shot_cloud(capture, row, cam, args, color_bgr):
    avg = capture / row["stop_dir"] / "avg"
    z = np.load(avg / "depth_mean_mm.npy").astype(np.float64) / 1000.0
    w = np.load(avg / "weight.npy").astype(np.float64)
    valid = cv2.imread(str(avg / "valid_mask.png"), cv2.IMREAD_GRAYSCALE)
    if valid is None:
        sys.exit(f"could not read {avg / 'valid_mask.png'}")
    if z.shape != cam["rays_d"].shape[:2] or w.shape != z.shape or valid.shape != z.shape:
        sys.exit(f"{avg}: map sizes {z.shape}/{w.shape}/{valid.shape} do not match the "
                 f"depth intrinsics {cam['rays_d'].shape[:2]}")
    keep = (valid > 0) & (z > 0) & (w >= args.w_min)
    P = apply(make_T(cam["R_dc"], cam["t_dc"]), cam["rays_d"][keep] * z[keep, None])
    uv = project(P, cam["Kc"], cam["dist_c"])
    h, wd = color_bgr.shape[:2]
    ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inb = (ui >= 0) & (ui < wd) & (vi >= 0) & (vi < h)
    col = np.full((len(P), 3), 0.5)
    col[inb] = color_bgr[vi[inb], ui[inb], ::-1] / 255.0
    return P, col, w[keep], uv, inb


def load_mask(capture, row, args):
    """SAM3 pot + plant mask (colour pixels), eroded so silhouette pixels stay out."""
    path = capture / row["stop_dir"] / "mask.png"
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        sys.exit(f"--mask: missing or unreadable {path} (run sam3_batch_masks.py)")
    if args.mask_erode > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.mask_erode + 1,) * 2)
        m = cv2.erode(m, k)
    return m > 0


def in_mask(mask, uv, inb):
    h, w = mask.shape
    keep = np.zeros(len(uv), bool)
    ui = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    vi = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    keep[inb] = mask[vi[inb], ui[inb]]
    return keep


# ---------------------------------------------------------------------------
# depth -> colour calibration refinement from the tags (--refine-depth)
# ---------------------------------------------------------------------------
def _fit_rigid(A, B):
    R, t = kabsch(A.reshape(-1, 3), B.reshape(-1, 3))
    return make_T(R, t)


def _corner_err_mm(T, A, B):
    """mean corner distance per tag (mm) after moving A by T; A, B are (tags, 4, 3)."""
    return np.linalg.norm(apply(T, A.reshape(-1, 3)).reshape(A.shape) - B, axis=2).mean(1) * 1000


def estimate_depth_correction(tag_sets, reject_mm=6.0):
    """One rigid T (colour frame) that moves depth-derived geometry onto the colour-only
    PnP tags, from every tag measured both ways (depth-plane corners vs PnP corners).

    tag_sets: one measure_tags() dict per shot. Tags far off after a first fit (plane
    caught the background, etc.) are dropped and the fit repeated. The reported "after"
    error is HELD-OUT: fitted on even shots and scored on odd shots, and vice versa.
    Returns (T or None, stats).
    """
    A, B, sh = [], [], []
    for k, tags in enumerate(tag_sets):
        for r in tags.values():
            if r.get("ok") and r.get("ok_pnp"):
                A.append(r["corners_plane"]), B.append(r["corners_pnp"]), sh.append(k)
    A, B, sh = np.asarray(A, float), np.asarray(B, float), np.asarray(sh)
    stats = dict(n_tags=int(len(A)), n_shots=int(len(set(sh.tolist()))))
    if len(A) < 6 or stats["n_shots"] < 4:
        stats["error"] = f"only {len(A)} tags in {stats['n_shots']} shots measured both ways"
        return None, stats
    keep = np.ones(len(A), bool)
    for _ in range(4):
        T = _fit_rigid(A[keep], B[keep])
        e = _corner_err_mm(T, A, B)
        new = e <= max(reject_mm, 3.0 * float(np.median(e[keep])))
        if (new == keep).all():
            break
        keep = new
    held = np.full(len(A), np.nan)
    for half in (0, 1):
        train, test = keep & (sh % 2 != half), sh % 2 == half
        if train.sum() >= 3:
            held[test] = _corner_err_mm(_fit_rigid(A[train], B[train]), A[test], B[test])
    ident = np.eye(4)
    view = B / np.linalg.norm(B, axis=2, keepdims=True)
    rng = lambda T: np.einsum("ijk,ijk->ij", apply(T, A.reshape(-1, 3)).reshape(A.shape) - B, view
                              ).mean(1)[keep] * 1000
    stats.update(n_used=int(keep.sum()), n_rejected=int((~keep).sum()),
                 before_mm=float(_corner_err_mm(ident, A, B)[keep].mean()),
                 after_heldout_mm=float(np.nanmean(held[keep])),
                 range_offset_before_mm=float(np.median(rng(ident))),
                 range_offset_after_mm=float(np.median(rng(T))),
                 rotation_deg=rot_deg(T[:3, :3]), translation_mm=(T[:3, 3] * 1000).tolist())
    return T, stats


def corrected_camera(cam, T):
    """Fold T into the depth->colour extrinsic: new = T o old (depth rays unchanged)."""
    cam = dict(cam)
    cam["R_dc"] = T[:3, :3] @ cam["R_dc"]
    cam["t_dc"] = T[:3, :3] @ cam["t_dc"] + T[:3, 3]
    return cam


def roi_center(tags, args):
    """Centroid of the shot's usable tag centres -- the tags ride on the turntable."""
    c = [t["center_plane"] if t["ok"] else t["center_pnp"] for t in tags.values()
         if t["ok"] or t.get("ok_pnp")]
    return np.mean(c, 0) if (c and args.roi_radius > 0) else None


def roi_mask(P, center, args):
    return np.linalg.norm(P - center, axis=1) <= args.roi_radius


def voxel_weighted(P, C, w, v):
    """Weighted voxel grid. Returns mean point, mean colour, mean weight, weight sum."""
    w = np.maximum(w, 1e-9)
    k = np.floor(P / v).astype(np.int64)
    k -= k.min(0)
    n = k.max(0) + 1
    key = (k[:, 0] * n[1] + k[:, 1]) * n[2] + k[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    ws = np.bincount(inv, weights=w)
    Pm = np.column_stack([np.bincount(inv, weights=w * P[:, j]) for j in range(3)]) / ws[:, None]
    Cm = np.column_stack([np.bincount(inv, weights=w * C[:, j]) for j in range(3)]) / ws[:, None]
    return Pm, Cm, ws / cnt, ws


def icp_cloud(P, C, w, args):
    Pv, Cv, wv, _ = voxel_weighted(P, C, w, args.voxel)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pv))
    pcd.colors = o3d.utility.Vector3dVector(Cv)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.normal_radius, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    return dict(pcd=pcd, P=Pv, N=np.asarray(pcd.normals), w=wv)


# ===========================================================================
# step 2a: tags in 3D -- plane fit inside the tag, rays onto the plane
# ===========================================================================
def fit_plane(P, w, thr, iters, rng):
    """Weighted RANSAC then weighted PCA. Returns n, d (n.x + d = 0), inliers, rms."""
    best, best_score = None, -1.0
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - s[0]) @ n) < thr
        score = w[inl].sum()
        if score > best_score:
            best, best_score = inl, score
    if best is None or best.sum() < 3:
        return None
    for _ in range(2):                                    # refit, re-select, refit
        Pi, wi = P[best], w[best]
        c = (wi[:, None] * Pi).sum(0) / wi.sum()
        X = Pi - c
        n = np.linalg.eigh((X * wi[:, None]).T @ X)[1][:, 0]
        d = -n @ c
        best = np.abs(P @ n + d) < thr
        if best.sum() < 3:
            return None
    r = P[best] @ n + d
    rms = float(np.sqrt(np.average(r ** 2, weights=w[best])))
    return n, d, best, rms


def ray_plane(R, n, d):
    denom = R @ n
    if np.any(np.abs(denom) < 1e-9):
        return None
    return R * (-d / denom)[:, None]


def pnp_tag(cpx, cam, args):
    """Colour-only tag pose from its 4 corners and known size (no depth involved).

    pupil_apriltags returns corners as [TR, TL, BL, BR] of the upright tag; OpenCV's
    IPPE_SQUARE wants [TL, TR, BR, BL] -- the permutation [1, 0, 3, 2] maps between the
    two (it is its own inverse). Both IPPE solutions are scored, the lower
    reprojection error wins.
    """
    s = args.tag_size_mm / 2000.0
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    order = [1, 0, 3, 2]
    img = cpx[order].reshape(-1, 1, 2)
    n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, cam["Kc"], cam["dist_c"],
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not n_sol:
        return dict(ok_pnp=False)
    k = int(np.argmin(np.asarray(errs).ravel()))
    R, t = cv2.Rodrigues(rvecs[k])[0], np.asarray(tvecs[k]).ravel()
    corners = (obj @ R.T + t)[order]
    n = R[:, 2] if R[:, 2] @ t < 0 else -R[:, 2]
    obliq = float(np.degrees(np.arccos(np.clip(abs(n @ unit(t)), 0, 1))))
    return dict(ok_pnp=obliq <= args.max_obliquity, corners_pnp=corners.tolist(),
                center_pnp=t.tolist(), normal_pnp=n.tolist(), obliquity_pnp_deg=obliq,
                pnp_reproj_px=float(np.asarray(errs).ravel()[k]))


def measure_tags(gray, P, w, uv, inb, cam, args, detector, rng):
    """All tags in one shot, in its colour-camera frame."""
    h, wd = gray.shape
    ui = np.clip(np.round(uv[:, 0]).astype(int), 0, wd - 1)
    vi = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    dets = {}
    for det in detector.detect(gray):
        if args.tag_ids and det.tag_id not in args.tag_ids:
            continue
        old = dets.get(det.tag_id)
        if old is None or det.decision_margin > old.decision_margin:
            dets[det.tag_id] = det

    out = {}
    erode = np.ones((2 * args.tag_erode + 1,) * 2, np.uint8)
    for tid, det in sorted(dets.items()):
        cpx = np.asarray(det.corners, np.float64)
        rec = dict(id=int(tid), px_corners=cpx.tolist(), px_center=[float(x) for x in det.center],
                   decision_margin=float(det.decision_margin), ok=False)
        rec.update(pnp_tag(cpx, cam, args))
        out[int(tid)] = rec
        poly = np.zeros((h, wd), np.uint8)
        cv2.fillConvexPoly(poly, np.round(cpx).astype(np.int32), 1)
        inner = cv2.erode(poly, erode)
        sel = inb & (inner[vi, ui] > 0)
        if sel.sum() < args.tag_min_pts:
            rec["error"] = f"only {int(sel.sum())} confident depth points on the tag"
            continue
        fit = fit_plane(P[sel], w[sel], args.plane_inlier_mm / 1000.0, args.ransac_iters, rng)
        if fit is None:
            rec["error"] = "plane fit failed"
            continue
        n, d, inl, prms = fit
        rc = rays(cpx, cam["Kc"], cam["dist_c"])
        corners = ray_plane(rc, n, d)
        center = ray_plane(rays(np.asarray(det.center)[None], cam["Kc"], cam["dist_c"]), n, d)
        if corners is None or center is None:
            rec["error"] = "tag plane is edge-on to the camera"
            continue
        center = center[0]
        if n @ center > 0:
            n, d = -n, -d

        # local-depth corners: median depth of confident points near each corner
        # (inside the slightly dilated tag outline) -- the "noisy corner z" variant
        near_pool = inb & (cv2.dilate(poly, np.ones((5, 5), np.uint8))[vi, ui] > 0)
        idx = np.nonzero(near_pool)[0]
        local = []
        for k in range(4):
            m = idx[np.hypot(*(uv[idx] - cpx[k]).T) <= args.corner_win]
            local.append((rc[k] * np.median(P[m, 2])).tolist() if len(m) >= 3 else None)
        local_ok = all(c is not None for c in local)

        sides = np.linalg.norm(corners - np.roll(corners, -1, 0), axis=1) * 1000
        obliq = float(np.degrees(np.arccos(np.clip(abs(n @ unit(center)), 0, 1))))
        rec.update(ok=True, n_pts=int(sel.sum()), n_inliers=int(inl.sum()),
                   plane_rms_mm=prms * 1000, normal=n.tolist(), obliquity_deg=obliq,
                   corners_plane=corners.tolist(), center_plane=center.tolist(),
                   corners_local=local,
                   center_corners_local=(np.mean(local, 0).tolist() if local_ok else None),
                   side_mm=sides.tolist())
        # sanity: a plane fitted to the background behind a silhouette tag, or a tag seen
        # too edge-on, gives wrong side lengths -- keep the numbers, mark it unusable
        bad_side = np.abs(sides - args.tag_size_mm).max()
        if args.tag_size_mm > 0 and bad_side > args.tag_side_tol * args.tag_size_mm:
            rec.update(ok=False, error=f"side lengths {np.round(sides, 1).tolist()} mm, "
                                       f"expected {args.tag_size_mm:g}")
        elif obliq > args.max_obliquity:
            rec.update(ok=False, error=f"obliquity {obliq:.0f} deg > {args.max_obliquity:g}")
    return out


# ===========================================================================
# step 2b: coarse pose from shared tags
# ===========================================================================
def tag_usable(rec, mode):
    return rec.get("ok_pnp", False) if mode == "pnp" else rec["ok"]


def tag_points(rec, mode):
    if mode == "pnp":
        return np.asarray(rec["corners_pnp"])
    if mode == "centers":
        return np.asarray([rec["center_plane"]])
    if mode == "plane":
        return np.asarray(rec["corners_plane"])
    if rec["corners_local"] is None or any(c is None for c in rec["corners_local"]):
        return None
    return np.asarray(rec["corners_local"])


def tag_align(ta, tb, args):
    """Kabsch from shot a's tags onto shot b's. Returns T_a_to_b + stats (or None)."""
    mode = "centers" if args.use_tag_centers else args.corner_depth
    shared = sorted(i for i in set(ta) & set(tb)
                    if tag_usable(ta[i], mode) and tag_usable(tb[i], mode)
                    and i not in args.exclude_tags)
    S, D, lab = [], [], []
    for i in shared:
        a, b = tag_points(ta[i], mode), tag_points(tb[i], mode)
        if a is None or b is None:
            continue
        S.append(a), D.append(b), lab.extend([i] * len(a))
    stats = dict(mode=mode, ids_shared=shared, ids_used=[], ids_rejected=[], n_corr=0)
    if not S:
        stats["error"] = "no shared usable tags"
        return None, stats
    S, D, lab = np.vstack(S), np.vstack(D), np.asarray(lab)
    tags = sorted(set(lab.tolist()))
    gate = args.tag_outlier_mm / 1000.0

    def fit(ids):
        m = np.isin(lab, ids)
        sv = np.linalg.svd(S[m] - S[m].mean(0), compute_uv=False)
        if sv[1] < 1e-3 * sv[0]:
            return None, None
        R, t = kabsch(S[m], D[m])
        res = np.linalg.norm(apply(make_T(R, t), S) - D, axis=1)
        return make_T(R, t), {i: float(np.sqrt(np.mean(res[lab == i] ** 2))) for i in tags}

    # consensus: fit every minimal tag subset, keep the one most tags agree with. One bad
    # tag cannot drag the others' residuals up the way a single all-tag fit lets it.
    k0 = min(3 if mode == "centers" else 2, len(tags))
    best = None
    for sub in combinations(tags, k0):
        T, per = fit(list(sub))
        if T is None:
            continue
        inl = [i for i in tags if per[i] <= gate]
        key = (len(inl), -sum(per[i] for i in inl))
        if best is None or key > best[0]:
            best = (key, inl)
    use = best[1] if best and best[1] else tags
    for _ in range(3):                                 # refit on the consensus, re-select
        T, per = fit(use)
        if T is None:
            break
        new = [i for i in tags if per[i] <= gate] or use
        if new == use:
            break
        use = new
    n_corr = int(np.isin(lab, use).sum())
    stats.update(ids_used=[int(i) for i in use], n_corr=n_corr,
                 ids_rejected=[int(i) for i in tags if i not in use])
    if T is None:
        stats["error"] = "tag points are collinear -- rotation is unconstrained"
        return None, stats
    stats["per_tag_rms_mm"] = {int(k): v * 1000 for k, v in per.items()}
    if n_corr < args.min_corr or len(use) < (3 if mode == "centers" else 1):
        stats["error"] = (f"{n_corr} correspondences from {len(use)} consistent tags "
                          f"(need >= {args.min_corr})")
        return None, stats
    m = np.isin(lab, use)
    stats["rms_mm"] = float(np.sqrt(np.mean(
        np.sum((apply(T, S[m]) - D[m]) ** 2, 1))) * 1000)
    return T, stats


# ===========================================================================
# step 3 + 4: confidence-weighted point-to-plane ICP with gates + conditioning
# ===========================================================================
class NN:
    def __init__(self, P):
        self.nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(np.ascontiguousarray(P)))
        self.nns.knn_index()

    def __call__(self, Q):
        i, d2 = self.nns.knn_search(o3d.core.Tensor(np.ascontiguousarray(Q)), 1)
        return i.numpy()[:, 0].astype(np.int64), np.sqrt(d2.numpy()[:, 0])


def p2p_system(p, q, nq, W, center=None, scale=1.0):
    """Weighted point-to-plane normal equations for xi = [omega, t]."""
    pc = p if center is None else (p - center)
    J = np.hstack([np.cross(pc, nq) / scale, nq])
    r = np.sum((p - q) * nq, 1)
    JW = J * W[:, None]
    return JW.T @ J, -JW.T @ r, r


def icp(src, tgt, T0, args):
    d0, d1 = args.icp_max_dist_mm / 1000.0, args.icp_min_dist_mm / 1000.0
    cos_gate = np.cos(np.radians(args.normal_angle_max))
    huber = args.huber_mm / 1000.0

    # overlap: keep only points that have a partner within d0 under the tag pose, both ways
    ps, ns, ws = src["P"], src["N"], src["w"]
    pt, nt, wt = tgt["P"], tgt["N"], tgt["w"]
    _, ds = NN(pt)(apply(T0, ps))
    ks = ds < d0
    if ks.sum() < args.icp_min_pts:
        return None, dict(error=f"overlap under the tag pose has only {int(ks.sum())} points")
    _, dt = NN(apply(T0, ps[ks]))(pt)
    kt = dt < d0
    ps, ns, ws, pt, nt, wt = ps[ks], ns[ks], ws[ks], pt[kt], nt[kt], wt[kt]
    nn = NN(pt)

    T = T0.copy()
    it, converged = 0, False
    for it in range(1, args.icp_iters + 1):
        dmax = max(d1, d0 * args.icp_shrink ** (it - 1))
        p, n = apply(T, ps), ns @ T[:3, :3].T
        idx, dist = nn(p)
        g = (dist < dmax) & (np.sum(n * nt[idx], 1) > cos_gate)
        if g.sum() < args.icp_min_pts:
            return None, dict(error=f"only {int(g.sum())} gated correspondences at iter {it}")
        q, nq = pt[idx[g]], nt[idx[g]]
        r = np.sum((p[g] - q) * nq, 1)
        W = ws[g] * wt[idx[g]] * np.minimum(1.0, huber / np.maximum(np.abs(r), 1e-12))
        A, b, _ = p2p_system(p[g], q, nq, W)
        xi = np.linalg.lstsq(A, b, rcond=None)[0]
        dT = make_T(cv2.Rodrigues(xi[:3])[0], xi[3:])
        T = dT @ T
        if dmax == d1 and np.linalg.norm(xi[:3]) < 1e-5 and np.linalg.norm(xi[3:]) < 1e-5:
            converged = True
            break

    # final evaluation at the tightest gate
    p, n = apply(T, ps), ns @ T[:3, :3].T
    idx, dist = nn(p)
    g = (dist < d1) & (np.sum(n * nt[idx], 1) > cos_gate)
    if g.sum() < args.icp_min_pts:
        return None, dict(error=f"only {int(g.sum())} inliers after convergence")
    q, nq = pt[idx[g]], nt[idx[g]]
    r = np.sum((p[g] - q) * nq, 1)
    W = ws[g] * wt[idx[g]] * np.minimum(1.0, huber / np.maximum(np.abs(r), 1e-12))
    c = np.average(p[g], axis=0, weights=W)
    L = float(np.sqrt(np.average(np.sum((p[g] - c) ** 2, 1), weights=W)))
    A, _, _ = p2p_system(p[g], q, nq, W, center=c, scale=L)
    ev, evec = np.linalg.eigh(A / W.sum())
    weak = evec[:, 0]
    stats = dict(
        iterations=it, converged=converged,
        last_step=dict(rot_deg=float(np.degrees(np.linalg.norm(xi[:3]))),
                       trans_mm=float(np.linalg.norm(xi[3:]) * 1000)),
        n_src_overlap=int(len(ps)), n_inliers=int(g.sum()),
        overlap_frac=float(g.sum() / len(src["P"])),
        rms_mm=float(np.sqrt(np.average(r ** 2, weights=W)) * 1000),
        rms_unweighted_mm=float(np.sqrt(np.mean(r ** 2)) * 1000),
        eigenvalues=ev.tolist(), condition=float(ev[-1] / max(ev[0], 1e-18)),
        weakest_mode=dict(rotation=weak[:3].tolist(), translation=weak[3:].tolist()),
        lever_arm_m=L)
    return T, stats


def register_pair(a, b, shots, args, kind, T_init=None, init_from=None):
    """T_init (optional) seeds ICP when the tags cannot -- see main()."""
    sa, sb = shots[a], shots[b]
    edge = dict(src=a, tgt=b, src_index=sa["index"], tgt_index=sb["index"], kind=kind,
                convention="X_tgt = T_src_to_tgt @ X_src (metres, colour-camera frames)")
    T_tag, tstats = tag_align(sa["tags"], sb["tags"], args)
    edge["tag"] = tstats
    flags = []
    if T_tag is None:
        if T_init is None or args.no_icp:
            edge.update(ok=False, flags=["no_tag_init"], quality="failed")
            return edge
        flags += ["no_tag_init", f"init_from_edge_{init_from}"]
        T0 = T_init
    else:
        edge["T_tag_src_to_tgt"] = T_tag.tolist()
        if tstats["rms_mm"] > args.tag_rms_max_mm:
            flags.append("tag_high_rms")
        T0 = T_tag

    T = T0
    if args.no_icp:
        edge["icp"] = None
    else:
        T_icp, istats = icp(sa["icp"], sb["icp"], T0, args)
        edge["icp"] = istats
        if T_icp is None:
            flags.append("icp_failed")
        elif T_tag is None and not (istats["converged"] and istats["rms_mm"] <= args.icp_rms_max_mm
                                    and istats["overlap_frac"] >= args.min_overlap):
            flags.append("icp_rejected_without_tags")   # no tag pose to fall back on
        elif T_tag is None:
            T = T_icp
            if istats["condition"] > args.cond_max:
                flags.append("icp_degenerate")
        else:
            T = T_icp
            if istats["condition"] > args.cond_max:
                flags.append("icp_degenerate")
            if istats["rms_mm"] > args.icp_rms_max_mm:
                flags.append("icp_high_residual")
            if istats["overlap_frac"] < args.min_overlap:
                flags.append("icp_low_overlap")
            if not istats["converged"]:
                flags.append("icp_not_converged")
            dr, dt = delta(T_tag, T_icp)
            istats["correction_from_tag"] = dict(rot_deg=dr, trans_mm=dt)
            if dr > args.max_correction_deg or dt > args.max_correction_mm:
                flags.append("icp_large_correction")
    if "icp_failed" in flags or "icp_rejected_without_tags" in flags:
        edge.update(ok=False, flags=flags, quality="failed")
        return edge
    edge.update(ok=True, T_src_to_tgt=T.tolist(), flags=flags,
                quality="ok" if not flags else "flagged")
    edge["info"] = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        sa["icp"]["pcd"], sb["icp"]["pcd"], args.icp_min_dist_mm / 1000.0, T)
    return edge


# ===========================================================================
# step 5: compose into shot 0's frame
# ===========================================================================
def initial_poses(n, edges):
    """BFS from node 0 over usable edges (odometry first). pose = T_node_to_ref."""
    adj = {i: [] for i in range(n)}
    for e in sorted(edges, key=lambda e: e["kind"] != "odometry"):
        T = np.asarray(e["T_src_to_tgt"])
        adj[e["src"]].append((e["tgt"], np.linalg.inv(T)))     # pose_tgt = pose_src @ T^-1
        adj[e["tgt"]].append((e["src"], T))                    # pose_src = pose_tgt @ T
    poses, q = {0: np.eye(4)}, deque([0])
    while q:
        u = q.popleft()
        for v, T in adj[u]:
            if v not in poses:
                poses[v] = poses[u] @ T
                q.append(v)
    return poses


def loop_error(edges, n):
    """Compose every odometry edge + the loop edge; identity if perfectly consistent."""
    by = {(e["src"], e["tgt"]): np.asarray(e["T_src_to_tgt"]) for e in edges}
    chain = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]
    if not all(k in by for k in chain):
        return None
    T = np.eye(4)
    for k in chain:
        T = by[k] @ T
    return dict(rot_deg=rot_deg(T[:3, :3]), trans_mm=float(np.linalg.norm(T[:3, 3]) * 1000))


def pose_graph(nodes, poses, edges, args):
    reg = o3d.pipelines.registration
    pos = {k: i for i, k in enumerate(nodes)}
    pg = reg.PoseGraph()
    for k in nodes:
        pg.nodes.append(reg.PoseGraphNode(poses[k]))
    for e in edges:
        if e["src"] in pos and e["tgt"] in pos:
            pg.edges.append(reg.PoseGraphEdge(pos[e["src"]], pos[e["tgt"]],
                                              np.asarray(e["T_src_to_tgt"]), e["info"],
                                              uncertain=(e["kind"] != "odometry")))
    n_edges = len(pg.edges)
    option = reg.GlobalOptimizationOption(
        max_correspondence_distance=args.icp_min_dist_mm / 1000.0,
        edge_prune_threshold=args.edge_prune, preference_loop_closure=args.loop_pref,
        reference_node=0)
    reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(),
                            reg.GlobalOptimizationConvergenceCriteria(), option)
    out = {k: np.asarray(pg.nodes[pos[k]].pose) for k in nodes}
    ref = np.linalg.inv(out[nodes[0]])                  # pin shot 0 to identity exactly
    return {k: ref @ T for k, T in out.items()}, n_edges, len(pg.edges)


def is_closed_loop(rows):
    deg = np.array([r["true_deg"] for r in rows])
    if len(deg) < 3:
        return False
    step = float(np.median(np.abs(np.diff(deg))))
    return bool(abs(deg[-1] - deg[0]) + step >= 360.0 - 0.5 * step)


# ===========================================================================
# output
# ===========================================================================
def jsonable(e):
    return {k: v for k, v in e.items() if k != "info"}


def write_pairs_csv(path, edges):
    cols = ["src_index", "tgt_index", "kind", "quality", "flags", "tag_mode", "tag_n_corr",
            "tag_ids_used", "tag_rms_mm", "icp_rms_mm", "icp_overlap_frac", "icp_condition",
            "icp_lambda_min", "icp_corr_rot_deg", "icp_corr_trans_mm"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in edges:
            t, i = e["tag"], e.get("icp") or {}
            c = i.get("correction_from_tag", {})
            w.writerow(dict(
                src_index=e["src_index"], tgt_index=e["tgt_index"], kind=e["kind"],
                quality=e["quality"], flags=" ".join(e["flags"]), tag_mode=t["mode"],
                tag_n_corr=t["n_corr"], tag_ids_used=" ".join(map(str, t["ids_used"])),
                tag_rms_mm=_f(t.get("rms_mm")), icp_rms_mm=_f(i.get("rms_mm")),
                icp_overlap_frac=_f(i.get("overlap_frac")), icp_condition=_f(i.get("condition")),
                icp_lambda_min=_f(i["eigenvalues"][0] if "eigenvalues" in i else None, 6),
                icp_corr_rot_deg=_f(c.get("rot_deg")), icp_corr_trans_mm=_f(c.get("trans_mm"))))


def _f(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"


def fuse(capture, rows, shots, nodes, poses, cam, args):
    Ps, Cs, Ws = [], [], []
    for k in nodes:
        color = cv2.imread(str(capture / rows[k]["color"]))
        P, C, w, uv, inb = shot_cloud(capture, rows[k], cam, args, color)
        if args.mask:
            m = in_mask(load_mask(capture, rows[k], args), uv, inb)
            P, C, w = P[m], C[m], w[m]
        if shots[k]["roi"] is not None:
            m = roi_mask(P, shots[k]["roi"], args)
            P, C, w = P[m], C[m], w[m]
        Pv, Cv, _, ws = voxel_weighted(apply(poses[k], P), C, w, args.fused_voxel)
        Ps.append(Pv), Cs.append(Cv), Ws.append(ws)
    P, C, _, _ = voxel_weighted(np.vstack(Ps), np.vstack(Cs), np.concatenate(Ws), args.fused_voxel)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    return pcd


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Tag-initialised, confidence-weighted ICP stitching of a turntable sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture", help="capture folder (sweep_capture.py + average_confidence.py)")
    ap.add_argument("--out", default="stitch", help="output sub-folder of the capture")
    ap.add_argument("--indices", default=None, help="manifest indices to use, e.g. 0-8,10")
    g = ap.add_argument_group("preprocessing")
    g.add_argument("--w-min", type=float, default=0.2, help="drop points with weight below this")
    g.add_argument("--voxel", type=float, default=0.002, help="ICP voxel size, m")
    g.add_argument("--normal-radius", type=float, default=0.006, help="normal estimation radius, m")
    g.add_argument("--refine-depth", action="store_true",
                   help="estimate one rigid depth->colour correction from this capture's tags "
                        "(depth-plane vs PnP corners), check it on held-out shots, apply it to "
                        "every depth cloud; saved as depth_correction.json")
    g.add_argument("--depth-correction", default=None,
                   help="apply a saved depth_correction.json instead of estimating one")
    g.add_argument("--mask", action="store_true",
                   help="keep only points inside each stop's SAM3 mask.png (sam3_batch_masks.py) "
                        "for ICP and fusion; tags are still measured on the unmasked data")
    g.add_argument("--mask-erode", type=int, default=3, help="px (colour) to erode the SAM3 mask")
    g.add_argument("--roi-radius", type=float, default=0.25,
                   help="m; keep points within this of the shot's tag-centre centroid, so the "
                        "static background cannot pull ICP toward identity (0 = no crop)")
    g = ap.add_argument_group("tags")
    g.add_argument("--tag-family", default="tag36h11")
    g.add_argument("--tag-ids", default=None, help="only consider these ids, e.g. 12-23")
    g.add_argument("--exclude-tags", default=None,
                   help="never register with these ids (hold-out tags for spread_analysis.py)")
    g.add_argument("--use-tag-centers", action="store_true",
                   help="one plane-fit centre per tag instead of 4 corners")
    g.add_argument("--corner-depth", choices=["pnp", "plane", "local"], default="pnp",
                   help="where tag corners come from: colour-only PnP with --tag-size-mm "
                        "(default; immune to ToF depth bias on the tag cards), colour rays on "
                        "the depth-fitted tag plane, or local median depth")
    g.add_argument("--min-corr", type=int, default=None,
                   help="min correspondences (default 16 for corners, 3 for centres)")
    g.add_argument("--tag-erode", type=int, default=3, help="px kept off the tag border")
    g.add_argument("--tag-min-pts", type=int, default=30)
    g.add_argument("--tag-size-mm", type=float, default=24.0,
                   help="black-border edge length: sets the PnP scale, and depth-plane tags whose "
                        "3D sides disagree by more than --tag-side-tol are dropped")
    g.add_argument("--tag-side-tol", type=float, default=0.10,
                   help="allowed side-length error, fraction of --tag-size-mm")
    g.add_argument("--max-obliquity", type=float, default=55.0,
                   help="deg; tags seen more edge-on than this are dropped (sides inflate beyond ~55)")
    g.add_argument("--plane-inlier-mm", type=float, default=1.5)
    g.add_argument("--ransac-iters", type=int, default=200)
    g.add_argument("--corner-win", type=float, default=6.0, help="px, --corner-depth local")
    g.add_argument("--tag-outlier-mm", type=float, default=4.0,
                   help="drop a tag whose Kabsch residual exceeds this (and 3x the median)")
    g = ap.add_argument_group("ICP")
    g.add_argument("--no-icp", action="store_true", help="tags only")
    g.add_argument("--no-neighbour-init", action="store_true",
                   help="do not seed ICP from the nearest good edge when a pair has too few "
                        "shared tags (turntable steps are equal, so neighbouring edges agree)")
    g.add_argument("--icp-max-dist-mm", type=float, default=8.0,
                   help="initial distance gate, also the overlap radius")
    g.add_argument("--icp-min-dist-mm", type=float, default=3.0, help="final distance gate")
    g.add_argument("--icp-shrink", type=float, default=0.85, help="gate shrink per iteration")
    g.add_argument("--normal-angle-max", type=float, default=35.0, help="deg")
    g.add_argument("--huber-mm", type=float, default=1.5)
    g.add_argument("--icp-iters", type=int, default=60)
    g.add_argument("--icp-min-pts", type=int, default=200)
    g = ap.add_argument_group("quality flags")
    g.add_argument("--tag-rms-max-mm", type=float, default=3.0)
    g.add_argument("--icp-rms-max-mm", type=float, default=2.0)
    g.add_argument("--cond-max", type=float, default=1000.0,
                   help="flag above this condition number; turntable scenes sit ~100-500 "
                        "because rotation about the axis is the weakest mode")
    g.add_argument("--min-overlap", type=float, default=0.2)
    g.add_argument("--max-correction-deg", type=float, default=2.0)
    g.add_argument("--max-correction-mm", type=float, default=10.0)
    g = ap.add_argument_group("composition")
    g.add_argument("--loop", choices=["auto", "yes", "no"], default="auto",
                   help="closed loop -> pose graph; auto = from manifest true_deg span")
    g.add_argument("--edge-prune", type=float, default=0.25)
    g.add_argument("--loop-pref", type=float, default=1.0, help="preference_loop_closure")
    g.add_argument("--fused-voxel", type=float, default=0.001, help="m; 0 = skip fused.ply")
    g.add_argument("--view", action="store_true", help="show the fused cloud")
    args = ap.parse_args()

    args.tag_ids = parse_ids(args.tag_ids)
    args.exclude_tags = parse_ids(args.exclude_tags)
    if args.min_corr is None:
        args.min_corr = 3 if args.use_tag_centers else 16
    capture = Path(args.capture)
    if not capture.is_dir():
        sys.exit(f"capture folder not found: {capture}")
    out = capture / args.out
    out.mkdir(parents=True, exist_ok=True)

    cam = load_camera(capture)
    rows = load_manifest(capture, parse_indices(args.indices) if args.indices else None)
    detector = Detector(families=args.tag_family, quad_decimate=1.0, refine_edges=True)
    rng = np.random.default_rng(0)

    # ---- 1 + 2a: clouds and tags per shot ---------------------------------
    def build_shots(cam, verbose):
        shots = []
        for r in rows:
            shot = build_one(r, cam)
            shots.append(shot)
            if verbose:
                print(shot.pop("line"))
        return shots

    def build_one(r, cam):
        color = cv2.imread(str(capture / r["color"]))
        if color is None or (color.shape[1], color.shape[0]) != tuple(cam["c_wh"]):
            sys.exit(f"{r['color']}: unreadable or not {cam['c_wh']} (colour intrinsics size)")
        P, C, w, uv, inb = shot_cloud(capture, r, cam, args, color)
        if len(P) < args.icp_min_pts:
            sys.exit(f"shot {r['index']}: only {len(P)} confident points -- check --w-min")
        tags = measure_tags(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), P, w, uv, inb,
                            cam, args, detector, rng)
        n_all = len(P)
        if args.mask:
            m = in_mask(load_mask(capture, r, args), uv, inb)
            P, C, w = P[m], C[m], w[m]
            if len(P) < args.icp_min_pts:
                sys.exit(f"shot {r['index']}: only {len(P)} points inside the SAM3 mask")
        roi = roi_center(tags, args)
        if roi is not None:
            m = roi_mask(P, roi, args)
            P, C, w = P[m], C[m], w[m]
        elif args.roi_radius > 0:
            print(f"[WARN] shot {r['index']:03d}: no usable tag -> no ROI crop")
        shot = dict(index=r["index"], true_deg=r["true_deg"], tags=tags, roi=roi,
                    icp=icp_cloud(P, C, w, args))
        good = [t for t in tags.values()
                if tag_usable(t, "centers" if args.use_tag_centers else args.corner_depth)]
        shot["line"] = (f"[SHOT] {r['index']:03d} {r['true_deg']:8.3f} deg  {n_all:7,d} -> {len(P):7,d} pts  "
                        f"{len(shot['icp']['P']):6,d} voxels  tags {sorted(t['id'] for t in good)}"
                        + (f"  (unusable: {sorted(i for i, t in tags.items() if not t['ok'])})"
                           if len(good) < len(tags) else ""))
        return shot

    depth_corr = None
    if args.depth_correction:
        f = Path(args.depth_correction)
        if not f.is_file():
            sys.exit(f"--depth-correction: {f} not found")
        depth_corr = dict(source=str(f), **json.loads(f.read_text()))
        cam = corrected_camera(cam, np.asarray(depth_corr["T"]))
        print(f"[CAL ] depth->colour correction loaded from {f}")
        shots = build_shots(cam, True)
    elif args.refine_depth:
        first = build_shots(cam, False)
        T_corr, st = estimate_depth_correction([s_["tags"] for s_ in first])
        if T_corr is None:
            sys.exit(f"--refine-depth: {st['error']}")
        print(f"[CAL ] depth->colour refinement from {st['n_used']} tags in {st['n_shots']} shots "
              f"({st['n_rejected']} rejected): tag mismatch {st['before_mm']:.2f} -> "
              f"{st['after_heldout_mm']:.2f} mm (held-out), distance offset "
              f"{st['range_offset_before_mm']:+.2f} -> {st['range_offset_after_mm']:+.2f} mm; "
              f"correction {st['rotation_deg']:.2f} deg, {np.round(st['translation_mm'], 1).tolist()} mm")
        if st["after_heldout_mm"] >= st["before_mm"]:
            print("[CAL ] held-out check did not improve -- correction NOT applied")
            shots = first
            for s_ in shots:
                print(s_.pop("line"))
        else:
            depth_corr = dict(T=T_corr.tolist(), stats=st,
                              note="applied as depth->colour extrinsic := T o SDK extrinsic")
            cam = corrected_camera(cam, T_corr)
            (out / "depth_correction.json").write_text(json.dumps(depth_corr, indent=2))
            shots = build_shots(cam, True)
    else:
        shots = build_shots(cam, True)

    n = len(shots)
    closed = {"yes": True, "no": False, "auto": is_closed_loop(rows)}[args.loop]
    pairs = [(i, i + 1, "odometry") for i in range(n - 1)]
    if closed:
        pairs.append((n - 1, 0, "loop"))

    # ---- 2b + 3 + 4: pairwise ---------------------------------------------
    print(f"\n  pair      kind      tags(corr)  tag_rms  icp_rms  overlap   cond    flags")
    edges = [register_pair(a, b, shots, args, kind) for a, b, kind in pairs]
    if not args.no_neighbour_init and not args.no_icp:
        seeds = [e for e in edges if e["ok"] and "T_tag_src_to_tgt" in e and e["kind"] == "odometry"
                 and e.get("icp") and "error" not in e["icp"]]
        for j, e in enumerate(edges):
            if e["ok"] or not seeds:
                continue
            s = min(seeds, key=lambda x: abs(x["src"] - e["src"]))
            edges[j] = register_pair(e["src"], e["tgt"], shots, args, e["kind"],
                                     T_init=np.asarray(s["T_src_to_tgt"]),
                                     init_from=f"{s['src_index']:03d}->{s['tgt_index']:03d}")
    for e in edges:
        kind = e["kind"]
        t, i = e["tag"], e.get("icp") or {}
        print(f"  {e['src_index']:03d}->{e['tgt_index']:03d}  {kind:<8}  "
              f"{len(t['ids_used']):2d} ({t['n_corr']:3d})   "
              f"{t.get('rms_mm', float('nan')):6.2f}   {i.get('rms_mm', float('nan')):6.2f}   "
              f"{i.get('overlap_frac', float('nan')):5.2f}  {i.get('condition', float('nan')):7.1f}  "
              f"{' '.join(e['flags']) or 'ok'}"
              + (f"   [{t['error']}]" if "error" in t else "")
              + (f"   [{i['error']}]" if "error" in i else ""))

    usable = [e for e in edges if e["ok"]]
    loop_edge_ok = any(e["kind"] == "loop" for e in usable)
    poses = initial_poses(n, usable)
    nodes = sorted(poses)
    lost = [shots[k]["index"] for k in range(n) if k not in poses]
    if lost:
        print(f"\n[WARN] not connected to shot {shots[0]['index']:03d}, dropped: {lost}")

    loop_info = dict(closed_loop_requested=closed, loop_edge_ok=loop_edge_ok)
    if closed and loop_edge_ok:
        loop_info["loop_error_before"] = loop_error(usable, n)
        poses, e_in, e_out = pose_graph(nodes, poses, usable, args)
        method = "pose_graph"
        loop_info.update(edges_in=e_in, edges_after_pruning=e_out)
        for e in usable:
            if e["src"] in poses and e["tgt"] in poses:
                pred = np.linalg.inv(poses[e["tgt"]]) @ poses[e["src"]]
                dr, dt = delta(np.asarray(e["T_src_to_tgt"]), pred)
                e["posegraph_residual"] = dict(rot_deg=dr, trans_mm=dt)
        le = loop_info["loop_error_before"]
        print(f"\n[LOOP] closure error before optimisation: "
              + (f"{le['rot_deg']:.3f} deg, {le['trans_mm']:.2f} mm" if le else "n/a (gap in chain)")
              + f"; pose graph kept {e_out}/{e_in} edges")
    else:
        method = "chain"
        if closed:
            print("\n[WARN] loop edge failed -- falling back to chaining the odometry edges")

    # ---- outputs ------------------------------------------------------------
    (out / "pairs.json").write_text(json.dumps([jsonable(e) for e in edges], indent=2))
    write_pairs_csv(out / "pairs.csv", edges)
    (out / "tags_per_shot.json").write_text(json.dumps({
        "frame": "each shot's colour-camera frame, metres",
        "shots": [dict(index=s["index"], true_deg=s["true_deg"],
                       tags={str(k): v for k, v in s["tags"].items()}) for s in shots]},
        indent=2))
    (out / "poses.json").write_text(json.dumps({
        "convention": ("T_shot_to_ref maps a point in shot k's colour-camera frame (m) into "
                       "the reference shot's colour-camera frame: X_ref = T @ X_k"),
        "reference_index": shots[0]["index"], "method": method, "loop": loop_info,
        "registration_tags_excluded": sorted(args.exclude_tags),
        "depth_correction": depth_corr,
        "shots": [dict(index=shots[k]["index"], true_deg=shots[k]["true_deg"],
                       T_shot_to_ref=poses[k].tolist()) for k in nodes],
        "unreachable_indices": lost,
        "args": {k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(args).items()}},
        indent=2))
    print(f"\n[OUT ] {method}: {len(nodes)}/{n} shots posed -> {out}")

    n_bad = sum(e["quality"] != "ok" for e in edges)
    if n_bad:
        print(f"[WARN] {n_bad}/{len(edges)} edges flagged or failed -- see pairs.csv")

    if args.fused_voxel > 0:
        fused = fuse(capture, rows, shots, nodes, poses, cam, args)
        o3d.io.write_point_cloud(str(out / "fused.ply"), fused)
        print(f"[OUT ] fused.ply  {len(fused.points):,} pts")
        if args.view:
            o3d.visualization.draw_geometries([fused], window_name="tag_stitch fused")


if __name__ == "__main__":
    main()
