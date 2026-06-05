#!/usr/bin/env python3
"""
Generate point clouds (PLY) from frame 00 of each angle in a capture session.

Reads a session directory produced by the capture script, loads color +
aligned depth pairs at burst index f00 for every captured angle, and writes
colored point clouds in camera coordinates.

Output naming
-------------
    <session_dir>/pointclouds_f00/
        cloud_000.ply
        cloud_010.ply
        ...

Usage
-----
    python make_pointclouds_f00.py <session_dir>

Assumes meta.json is present in <session_dir> with color_intrinsics and
depth_scale_m_per_unit. Depth is aligned to COLOR, so we back-project using
the COLOR intrinsics.
"""
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d


def load_meta(session_dir: Path) -> dict:
    meta_path = session_dir / 'meta.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'meta.json not found in {session_dir}')
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def angle_str(deg: int) -> str:
    return f'{deg:03d}'


def make_pointcloud(color_bgr: np.ndarray,
                    depth_u16: np.ndarray,
                    fx: float, fy: float, cx: float, cy: float,
                    depth_scale: float,
                    z_min: float = 0.1,
                    z_max: float = 10.0) -> o3d.geometry.PointCloud:
    """
    Back-project aligned depth + color into a colored point cloud.
    Depth is uint16 in raw units; multiply by depth_scale to get meters.
    Color is BGR (OpenCV); converted to RGB for Open3D.
    """
    h, w = depth_u16.shape

    # Sanity: depth must match color resolution (depth was aligned to color)
    if color_bgr.shape[:2] != (h, w):
        # Resize color to depth size (NEAREST to avoid blending)
        color_bgr = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_NEAREST)

    # Per-pixel grids
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    z = depth_u16.astype(np.float32) * depth_scale  # meters

    # Valid depth mask
    valid = (z > z_min) & (z < z_max)

    # Back-project: x = (u - cx) * z / fx
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    pts = np.stack([x, y, z], axis=-1)[valid]                  # (N, 3)
    colors_bgr = color_bgr[valid]                              # (N, 3) uint8
    colors_rgb = colors_bgr[:, ::-1].astype(np.float32) / 255  # BGR → RGB, [0,1]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64))
    return pc


def main():
    print("running")
    if len(sys.argv) != 2:
        print('Usage: python make_pointclouds_f00.py <session_dir>')
        sys.exit(1)

    session_dir = Path(sys.argv[1]).resolve()
    if not session_dir.is_dir():
        print(f'[error] not a directory: {session_dir}')
        sys.exit(1)

    meta = load_meta(session_dir)

    # Intrinsics — depth is aligned to COLOR, so use color intrinsics
    if 'color_intrinsics' not in meta:
        raise RuntimeError('meta.json missing color_intrinsics')
    K = meta['color_intrinsics']
    fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']
    depth_scale = meta.get('depth_scale_m_per_unit', 0.001)

    angles = meta.get('captured_angles', [])
    if not angles:
        print('[warn] no captured_angles in meta.json — nothing to do')
        return

    out_dir = session_dir / 'pointclouds_f00'
    print(f'[info] output directory: {out_dir}')
    out_dir.mkdir(exist_ok=True)

    print(f'[info] session   : {session_dir}')
    print(f'[info] intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')
    print(f'[info] depth scl : {depth_scale} m/unit')
    print(f'[info] angles    : {len(angles)}  → {out_dir}')
    print()

    n_ok, n_skip = 0, 0
    for ang in angles:
        astr = angle_str(ang)
        c_path = session_dir / f'color_{astr}_f00.png'
        d_path = session_dir / f'depth_{astr}_f00.png'
        out_path = out_dir / f'pointcloud_{astr}_f00.ply'

        if not c_path.exists() or not d_path.exists():
            print(f'  [skip] {ang:>3}° — missing file(s)')
            n_skip += 1
            continue

        color_bgr = cv2.imread(str(c_path), cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(str(d_path), cv2.IMREAD_UNCHANGED)
        if color_bgr is None or depth_u16 is None:
            print(f'  [skip] {ang:>3}° — failed to read')
            n_skip += 1
            continue
        if depth_u16.dtype != np.uint16:
            print(f'  [skip] {ang:>3}° — depth not uint16 (got {depth_u16.dtype})')
            n_skip += 1
            continue

        pc = make_pointcloud(color_bgr, depth_u16, fx, fy, cx, cy, depth_scale)
        # write_ascii=False keeps files small; works

        o3d.io.write_point_cloud(str(out_path), pc, write_ascii=False)
        print(f'  [ok]   {ang:>3}° → {out_path.name}  ({len(pc.points):>7} pts)')
        n_ok += 1

    print()
    print(f'[done] {n_ok} written, {n_skip} skipped → {out_dir}')


if __name__ == '__main__':
    main()