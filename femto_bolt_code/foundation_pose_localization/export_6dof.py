#!/usr/bin/env python3
"""
Apply 6DOF transformation from a text file to a CAD model and export.

python export_6dof.py --cad ../../cad_model/StructureOnly.PLY --transform ../../6dof/20250917_164430.txt --output ../../6dof_export/20250917_164430_transformed.ply --scale 0.001 --visualize

"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2


def load_transform_matrix(txt_path: Path) -> np.ndarray:
    """Load 4x4 transformation matrix from text file."""
    if not txt_path.exists():
        raise FileNotFoundError(f"Transform file not found: {txt_path}")
    
    T = np.loadtxt(str(txt_path))
    
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {T.shape}")
    
    # Validate bottom row
    expected_bottom = np.array([0, 0, 0, 1])
    if not np.allclose(T[3, :], expected_bottom):
        print(f"[WARN] Bottom row is {T[3, :]}, expected [0, 0, 0, 1]")
    
    return T


def extract_rvec_tvec(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract rotation vector and translation vector from 4x4 transform."""
    R = T[:3, :3]
    tvec = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten(), tvec


def load_cad_geometry(path: Path):
    """Load CAD as mesh or point cloud."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh and len(np.asarray(mesh.vertices)) > 0:
        mesh.compute_vertex_normals()
        print(f"[LOAD] Loaded mesh with {len(np.asarray(mesh.vertices))} vertices")
        return mesh
    
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd and len(np.asarray(pcd.points)) > 0:
        print(f"[LOAD] Loaded point cloud with {len(np.asarray(pcd.points))} points")
        return pcd
    
    raise RuntimeError(f"Failed to load CAD from {path}")


def save_geometry(geom, path: Path) -> bool:
    """Save mesh or point cloud to PLY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(geom, o3d.geometry.TriangleMesh):
        success = o3d.io.write_triangle_mesh(str(path), geom, write_ascii=False, compressed=True)
        geom_type = "mesh"
    elif isinstance(geom, o3d.geometry.PointCloud):
        success = o3d.io.write_point_cloud(str(path), geom, write_ascii=False, compressed=True)
        geom_type = "point cloud"
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")
    
    if success:
        print(f"[EXPORT] Saved {geom_type} to: {path}")
        print(f"[EXPORT] File size: {path.stat().st_size / 1024:.2f} KB")
    else:
        print(f"[ERROR] Failed to save to: {path}")
    
    return success


def geom_centroid(g) -> np.ndarray:
    """Get centroid of mesh or point cloud."""
    if isinstance(g, o3d.geometry.TriangleMesh):
        return np.asarray(g.get_center())
    return np.asarray(g.points).mean(axis=0)


def main():
    parser = argparse.ArgumentParser(description="Apply 6DOF transform to CAD model")
    parser.add_argument("--cad", type=Path, required=True, help="Input CAD PLY file")
    parser.add_argument("--transform", type=Path, required=True, help="4x4 transform matrix text file")
    parser.add_argument("--output", type=Path, required=True, help="Output PLY file")
    parser.add_argument("--scale", type=float, default=1.0, 
                        help="Scale factor for CAD units (e.g., 0.001 for mm->m)")
    parser.add_argument("--visualize", action="store_true", help="Show result in Open3D viewer")
    args = parser.parse_args()
    
    # Load transformation matrix
    print(f"\n[INFO] Loading transform from: {args.transform}")
    T = load_transform_matrix(args.transform)
    
    # Extract and display rvec/tvec
    rvec, tvec = extract_rvec_tvec(T)
    R = T[:3, :3]
    
    print(f"\n[TRANSFORM] 4x4 Matrix:\n{T}")
    print(f"\n[TRANSFORM] Rotation matrix R:\n{R}")
    print(f"[TRANSFORM] rvec (axis-angle): {rvec}")
    print(f"[TRANSFORM] tvec: {tvec}")
    
    # Validate rotation matrix (should be orthonormal, det=1)
    det = np.linalg.det(R)
    orthogonality_error = np.linalg.norm(R @ R.T - np.eye(3))
    print(f"[VALIDATE] det(R) = {det:.6f} (should be ~1.0)")
    print(f"[VALIDATE] ||R*R^T - I|| = {orthogonality_error:.2e} (should be ~0)")
    
    # Load CAD
    print(f"\n[INFO] Loading CAD from: {args.cad}")
    cad = load_cad_geometry(args.cad)
    
    centroid_before = geom_centroid(cad)
    print(f"[CAD] Original centroid: {centroid_before}")
    
    # Apply scale if needed
    if abs(args.scale - 1.0) > 1e-9:
        cad.scale(args.scale, center=(0, 0, 0))
        print(f"[CAD] Scaled by {args.scale}")
        centroid_after_scale = geom_centroid(cad)
        print(f"[CAD] Centroid after scale: {centroid_after_scale}")
    
    # Apply 4x4 transformation
    cad.transform(T)
    print(f"[CAD] Applied 4x4 transformation")
    
    centroid_after = geom_centroid(cad)
    print(f"[CAD] Final centroid: {centroid_after}")
    
    # Save result
    print(f"\n[INFO] Saving to: {args.output}")
    save_geometry(cad, args.output)
    
    # Optional visualization
    if args.visualize:
        print("\n[VIZ] Opening viewer...")
        geoms = [cad]
        
        # Add coordinate axes at origin
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geoms.append(axes)
        
        o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()