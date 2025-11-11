#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d


def load_geometry_strict(p: Path):
    """
    Load geometry and classify correctly.
    Returns ("mesh", mesh) if it truly has triangles,
            ("pcd",  pcd)  if it's a point set (including triangle-mesh with 0 tris).
    """
    # Try reading as triangle mesh
    mesh = o3d.io.read_triangle_mesh(str(p))
    if mesh and len(np.asarray(mesh.vertices)) > 0:
        tris = np.asarray(mesh.triangles)
        if tris.size > 0:  # real mesh
            mesh.compute_vertex_normals()
            return "mesh", mesh
        # Vertices but no triangles: treat as point cloud
        print("[INFO] Input has mesh vertices but 0 triangles; treating as point cloud.")
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        # carry colors if any
        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors
        return "pcd", pcd

    # Fallback: read as point cloud
    pcd = o3d.io.read_point_cloud(str(p))
    if pcd and len(np.asarray(pcd.points)) > 0:
        return "pcd", pcd

    raise RuntimeError(f"Failed to load any geometry from {p}")


def ensure_normals_pcd(pcd: o3d.geometry.PointCloud, rad: float | None = None, max_nn: int = 30):
    if pcd.has_normals():
        return
    # Pick a radius if not given: ~ 3x mean nn distance
    if rad is None:
        dists = pcd.compute_nearest_neighbor_distance()
        if len(dists) == 0:
            rad = 0.02
        else:
            rad = max(1e-4, 3.0 * float(np.mean(dists)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(100)


def estimate_bpa_radii(pcd: o3d.geometry.PointCloud) -> list[float]:
    """Heuristic BPA radii from average spacing."""
    dists = pcd.compute_nearest_neighbor_distance()
    if len(dists) == 0:
        base = 0.01
    else:
        base = float(np.mean(dists))
    # A small ladder around the mean spacing
    return [0.8*base, 1.2*base, 1.6*base]


def mesh_from_pcd(
    pcd: o3d.geometry.PointCloud,
    method: str = "bpa",
    poisson_depth: int = 9,
    bpa_radii: list[float] | None = None,
) -> o3d.geometry.TriangleMesh:
    ensure_normals_pcd(pcd)

    if method == "poisson":
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
    elif method == "bpa":
        if bpa_radii is None:
            bpa_radii = estimate_bpa_radii(pcd)
            print(f"[BPA] Auto radii (units): {bpa_radii}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(bpa_radii)
        )
    else:
        raise ValueError("Unknown meshing method (use 'poisson' or 'bpa').")

    # Cleanup and normals
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    return mesh


def parse_transform(path_or_csv: str | None) -> np.ndarray:
    T = np.eye(4, dtype=float)
    if not path_or_csv:
        return T
    cand = Path(path_or_csv)
    if cand.exists():
        data = json.loads(cand.read_text())
        T = np.array(data["T"], dtype=float)
        assert T.shape == (4, 4), "Transform JSON must contain 4x4 matrix under key 'T'."
        return T
    # comma-separated row-major 4x4
    parts = [float(x) for x in path_or_csv.split(",")]
    if len(parts) != 16:
        raise ValueError("Transform must have 16 comma-separated numbers (row-major 4x4).")
    return np.array(parts, dtype=float).reshape(4, 4)


def mesh_stats(mesh: o3d.geometry.TriangleMesh) -> dict:
    verts = np.asarray(mesh.vertices)
    tri = np.asarray(mesh.triangles)
    aabb = mesh.get_axis_aligned_bounding_box()
    mn = aabb.get_min_bound(); mx = aabb.get_max_bound()
    ctr = mesh.get_center()
    return {
        "num_vertices": int(verts.shape[0]),
        "num_triangles": int(tri.shape[0]),
        "aabb_min": [float(mn[0]), float(mn[1]), float(mn[2])],
        "aabb_max": [float(mx[0]), float(mx[1]), float(mx[2])],
        "centroid": [float(ctr[0]), float(ctr[1]), float(ctr[2])],
    }


def main():
    ap = argparse.ArgumentParser(description="Robust PLY→STL with point-cloud meshing and normals.")
    ap.add_argument("input_ply", type=str, help="Path to input .ply (mesh or point cloud)")
    ap.add_argument("-o", "--output", type=str, default=None, help="Output STL path (default: input basename + .stl)")
    ap.add_argument("--units", type=str, default="meters", choices=["meters", "millimeters", "centimeters"])
    ap.add_argument("--unit-scale", type=float, default=1.0, help="Multiply all coordinates by this factor before export")
    ap.add_argument("--transform", type=str, default=None,
                    help="Path to JSON with key 'T' (4x4), or 16 comma-separated numbers (row-major 4x4)")
    ap.add_argument("--method", type=str, default="bpa", choices=["bpa", "poisson"],
                    help="Surface reconstruction for point clouds or zero-triangle meshes")
    ap.add_argument("--poisson-depth", type=int, default=9, help="Poisson octree depth")
    ap.add_argument("--bpa-radii", type=str, default="", help="Comma radii for BPA (if empty, auto-estimate)")
    ap.add_argument("--ascii", action="store_true", help="Write ASCII STL")
    args = ap.parse_args()

    in_path = Path(args.input_ply)
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".stl")

    gtype, geom = load_geometry_strict(in_path)
    print(f"[LOAD] {gtype.upper()} loaded from {in_path}")

    # unit scale
    if args.unit_scale != 1.0:
        geom.scale(args.unit_scale, center=(0, 0, 0))
        print(f"[SCALE] unit_scale={args.unit_scale}")

    # transform
    T = parse_transform(args.transform) if args.transform else np.eye(4, dtype=float)
    if args.transform:
        geom.transform(T)
        print(f"[XFORM]\n{T}")

    # If it's a point cloud (or was a zero-triangle mesh), reconstruct a surface
    if gtype == "pcd":
        bpa_radii = None
        if args.method == "bpa":
            if args.bpa_radii.strip():
                bpa_radii = [float(x) for x in args.bpa_radii.split(",")]
        mesh = mesh_from_pcd(
            geom,
            method=args.method,
            poisson_depth=args.poisson_depth,
            bpa_radii=bpa_radii,
        )
    else:
        mesh = geom
        # safety: even for meshes, ensure normals exist
        mesh.compute_vertex_normals()

    # final cleanup & write
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = o3d.io.write_triangle_mesh(
        str(out_path),
        mesh,
        write_ascii=bool(args.ascii),
        compressed=False,
        print_progress=False,
    )
    if not ok:
        raise RuntimeError(f"Failed to write STL to {out_path}")
    print(f"[WRITE] STL: {out_path} ({'ASCII' if args.ascii else 'binary'})")

    # optional sidecar metadata
    meta = {
        "source": str(in_path),
        "output_stl": str(out_path),
        "units": args.units,
        "unit_scale_applied": float(args.unit_scale),
        "transform_applied_row_major": T.reshape(-1).tolist(),
        "mesh_stats": mesh_stats(mesh),
        "note": "STL has no embedded units/coords; see this JSON for context.",
    }
    (out_path.with_suffix(out_path.suffix + ".meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"[META] {out_path.with_suffix(out_path.suffix + '.meta.json')}")


if __name__ == "__main__":
    main()
