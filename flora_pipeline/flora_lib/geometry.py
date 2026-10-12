"""Rigid transforms, camera rays / projection and voxel fusion. Metres throughout."""

import cv2
import numpy as np

UNDISTORT_CRIT = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def apply(T, P):
    """X' = T @ X for an (N, 3) array."""
    return np.asarray(P) @ T[:3, :3].T + T[:3, 3]


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def rotvec(R):
    return cv2.Rodrigues(np.asarray(R, float))[0].ravel()


def rotate_about(axis, rad):
    return cv2.Rodrigues(np.asarray(axis, float) * rad)[0]


def scale_K(K, src_wh, dst_wh):
    """Intrinsics of the same camera at another image size."""
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    out = K.copy()
    out[0] *= sx
    out[1] *= sy
    return out


def rays(uv, K, dist):
    """Pixels (N, 2) -> undistorted rays (N, 3) with z = 1."""
    xy = cv2.undistortPointsIter(np.asarray(uv, np.float64).reshape(-1, 1, 2), K, dist,
                                 None, None, UNDISTORT_CRIT).reshape(-1, 2)
    return np.column_stack([xy, np.ones(len(xy))])


def ray_grid(K, dist, w, h):
    """(h, w, 3) ray per pixel, z = 1."""
    uv = np.stack(np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)), -1)
    return rays(uv.reshape(-1, 2), K, dist).reshape(h, w, 3)


def project(P, K, dist):
    uv, _ = cv2.projectPoints(np.asarray(P, np.float64).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return uv.reshape(-1, 2)


def voxel_fuse(P, C, voxel, w=None):
    """(Weighted) voxel grid: mean point and mean colour per occupied voxel."""
    w = np.ones(len(P)) if w is None else np.maximum(w, 1e-9)
    k = np.floor(P / voxel).astype(np.int64)
    k -= k.min(0)
    n = k.max(0) + 1
    _, inv = np.unique((k[:, 0] * n[1] + k[:, 1]) * n[2] + k[:, 2], return_inverse=True)
    inv = inv.ravel()
    ws = np.bincount(inv, weights=w)
    Pm = np.column_stack([np.bincount(inv, weights=w * P[:, j]) for j in range(3)]) / ws[:, None]
    Cm = np.column_stack([np.bincount(inv, weights=w * C[:, j]) for j in range(3)]) / ws[:, None]
    return Pm, Cm
