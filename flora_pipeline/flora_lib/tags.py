"""
AprilTag poses from the colour image alone, and tag-pose registration of two shots.

Each tag's pose comes from PnP on its 4 corners with the known tag size (no depth: ToF
depth on paper tags carries a view-dependent bias of several mm). For two shots a, b the
per-tag transforms T = P_b @ P_a^-1 are averaged: chordal L2 mean of the rotations, mean
tag-centre translation (= tag_pose_register.py's "pose" method, equal weights).
"""

import cv2
import numpy as np
from pupil_apriltags import Detector

from .geometry import make_T, unit


def make_detector(family="tag36h11"):
    return Detector(families=family, quad_decimate=1.0, refine_edges=True)


def pnp_tag(cpx, K, dist, tag_size_mm):
    """Tag pose from its 4 pixel corners.

    pupil_apriltags gives corners as [TR, TL, BL, BR] of the upright tag; OpenCV's
    IPPE_SQUARE wants [TL, TR, BR, BL] -- permutation [1, 0, 3, 2] (its own inverse).
    Both IPPE solutions are scored, the lower reprojection error wins."""
    s = tag_size_mm / 2000.0
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
    order = [1, 0, 3, 2]
    n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, cpx[order].reshape(-1, 1, 2), K, dist,
                                                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not n_sol:
        return None
    k = int(np.argmin(np.asarray(errs).ravel()))
    R, t = cv2.Rodrigues(rvecs[k])[0], np.asarray(tvecs[k]).ravel()
    n = R[:, 2] if R[:, 2] @ t < 0 else -R[:, 2]
    return dict(corners=(obj @ R.T + t)[order], center=t,
                obliquity_deg=float(np.degrees(np.arccos(np.clip(abs(n @ unit(t)), 0, 1)))),
                reproj_px=float(np.asarray(errs).ravel()[k]))


def detect_tags(gray, K, dist, detector, tag_size_mm, max_obliquity_deg):
    """{tag id: dict(corners (4, 3) m, center, obliquity_deg, reproj_px, ok)} in the colour
    camera frame. ok = seen within max_obliquity_deg of head-on (oblique PnP is unreliable)."""
    best = {}
    for det in detector.detect(gray):
        if det.tag_id not in best or det.decision_margin > best[det.tag_id].decision_margin:
            best[det.tag_id] = det
    out = {}
    for tid, det in sorted(best.items()):
        rec = pnp_tag(np.asarray(det.corners, np.float64), K, dist, tag_size_mm)
        if rec is not None:
            rec["ok"] = rec["obliquity_deg"] <= max_obliquity_deg
            out[int(tid)] = rec
    return out


def tag_pose(corners):
    """4x4 tag->camera pose from the 3D corners (order TR, TL, BL, BR)."""
    C = np.asarray(corners)
    x = unit(C[0] - C[1])
    y = unit(C[1] - C[2])
    z = unit(np.cross(x, y))
    y = np.cross(z, x)
    return make_T(np.column_stack([x, y, z]), C.mean(0))


def pose_average(Ts, w):
    """Chordal L2 mean of the rotations of Ts (weights w): SVD projection of the weighted sum."""
    M = sum(wi * T[:3, :3] for wi, T in zip(w, Ts))
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def register_pair(tags_a, tags_b):
    """a -> b transform (X_b = T @ X_a) from the tags usable in both; (None, []) if none."""
    ids = sorted(i for i in set(tags_a) & set(tags_b) if tags_a[i]["ok"] and tags_b[i]["ok"])
    if not ids:
        return None, ids
    Ts = [tag_pose(tags_b[i]["corners"]) @ np.linalg.inv(tag_pose(tags_a[i]["corners"])) for i in ids]
    w = np.full(len(ids), 1.0 / len(ids))
    R = pose_average(Ts, w)
    ca = np.array([tags_a[i]["center"] for i in ids])
    cb = np.array([tags_b[i]["center"] for i in ids])
    return make_T(R, (w[:, None] * (cb - ca @ R.T)).sum(0)), ids
