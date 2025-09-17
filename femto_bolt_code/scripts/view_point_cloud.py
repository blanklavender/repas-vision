# ------------------------------
# view_point_cloud_open3d.py
# ------------------------------
#!/usr/bin/env python3
"""
Open3D viewer for the color-frame point cloud (.ply) produced by the unified
capture script. Ensures true-to-scale (meters) visualization and adds helpful
reference geometry.

Usage
-----
python view_point_cloud_open3d.py \
  --ply ./captures_colorworld/capture_YYYYMMDD_HHMMSS/point_cloud_YYYYMMDD_HHMMSS.ply \
  --voxel 0.003 --z-min 0.15 --z-max 8.0 --grid 0.1 --axes 0.1

Options
-------
--ply      : Path to the saved PLY (already in COLOR frame, meters).
--voxel    : Optional voxel downsampling size in meters (e.g., 0.003 = 3 mm).
--z-min    : Cull points closer than this many meters.
--z-max    : Cull points farther than this many meters.
--axes     : Size (meters) of the origin coordinate frame gizmo (default 0.1 m).
--grid     : Draw an XY reference grid with given cell size in meters (e.g., 0.1 m). Off by default.

Notes
-----
- The PLY from the capture script is in **COLOR coordinates** and in **meters** (scale applied).
- Open3D renders geometry in whatever units the file uses; here, 1.0 = 1 meter.
- The origin (0,0,0) is at the COLOR camera center; +Z points forward.

way to run:
python view_point_cloud_open3d.py \
  --ply ./captures_colorworld/capture_YYYYMMDD_HHMMSS/point_cloud_YYYYMMDD_HHMMSS.ply \
  --voxel 0.003 --z-min 0.15 --z-max 8.0 --axes 0.1 --grid 0.1

"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d


def make_xy_grid(cell: float, n: int = 20, z: float = 1.0):
    """Create an XY grid (LineSet) centered at origin at plane Z=z.
    cell: grid cell size in meters
    n   : draws lines from -n*cell .. +n*cell (extent ≈ 2n*cell)
    z   : height of the grid plane in meters (default 1.0)
    """
    extent = n * cell
    # Lines parallel to X and Y
    pts = []
    lines = []
    colors = []
    # Y-parallel lines (vary X)
    for i, y in enumerate(np.linspace(-extent, extent, 2 * n + 1)):
        pts.append([-extent, y, z])
        pts.append([+extent, y, z])
        lines.append([len(pts) - 2, len(pts) - 1])
        colors.append([0.7, 0.7, 0.7])
    # X-parallel lines (vary Y)
    for i, x in enumerate(np.linspace(-extent, extent, 2 * n + 1)):
        pts.append([x, -extent, z])
        pts.append([x, +extent, z])
        lines.append([len(pts) - 2, len(pts) - 1])
        colors.append([0.7, 0.7, 0.7])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return ls


def describe_cloud(pcd: o3d.geometry.PointCloud):
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        print('[WARN] Empty point cloud')
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    rng = maxs - mins
    print('[INFO] Cloud stats (meters):')
    print('  Count :', pts.shape[0])
    print('  X min/max : %.3f .. %.3f (range %.3f m)' % (mins[0], maxs[0], rng[0]))
    print('  Y min/max : %.3f .. %.3f (range %.3f m)' % (mins[1], maxs[1], rng[1]))
    print('  Z min/max : %.3f .. %.3f (range %.3f m)' % (mins[2], maxs[2], rng[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', type=Path, required=True, help='Path to color-frame PLY')
    ap.add_argument('--voxel', type=float, default=0.0, help='Voxel size in meters for downsampling')
    ap.add_argument('--z-min', type=float, default=None, help='Minimum Z to keep (meters)')
    ap.add_argument('--z-max', type=float, default=None, help='Maximum Z to keep (meters)')
    ap.add_argument('--axes', type=float, default=0.1, help='Axes gizmo size in meters (default 0.1)')
    ap.add_argument('--grid', type=float, default=0.0, help='XY grid cell size in meters (0 = off)')
    args = ap.parse_args()

    p = Path(args.ply)
    if not p.exists():
        raise FileNotFoundError(p)

    pcd = o3d.io.read_point_cloud(str(p))
    if pcd.is_empty():
        raise RuntimeError('Loaded point cloud is empty')

    # Optional Z clipping (keep points in front of camera in a sensible range)
    if args.z_min is not None or args.z_max is not None:
        Z = np.asarray(pcd.points)[:, 2]
        keep = np.ones(Z.shape[0], dtype=bool)
        if args.z_min is not None:
            keep &= (Z >= float(args.z_min))
        if args.z_max is not None:
            keep &= (Z <= float(args.z_max))
        pcd = pcd.select_by_index(np.flatnonzero(keep))

    # Optional voxel downsample
    if args.voxel and args.voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel))

    describe_cloud(pcd)

    geoms = [pcd]
    if args.axes and args.axes > 0:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.axes)))
    if args.grid and args.grid > 0:
        geoms.append(make_xy_grid(cell=float(args.grid), n=20, z=0.0))  # grid at Z=0 (camera plane passes through origin)

    o3d.visualization.draw_geometries(geoms)


if __name__ == '__main__':
    main()
