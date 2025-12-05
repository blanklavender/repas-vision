#!/usr/bin/env python3
"""
Apply 6DOF transformation from a text file to a CAD model.
Optionally refine with ICP against a scene point cloud.

Usage:
    python apply_6dof_transform.py

The transformation matrix should be a 4x4 homogeneous matrix in a text file,
space-separated, one row per line.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import json

# =========================================================
#                     CONFIG
# =========================================================

# Input files
CAD_PLY = Path("../../cad_model/StructureOnly.PLY")           # Your CAD model (mesh or point cloud)
TRANSFORM_TXT = Path("../../captures/captures_colorworld_stable/capture_20251121_162722/foundation pose results/ob_in_cam/20251121_162722.txt")                      # 4x4 transformation matrix
SCENE_PLY = Path("../../captures/captures_colorworld_stable/capture_20251121_162722/final_crop/cropped_camframe_20251121_162722.ply")                         # Scene point cloud (for ICP, optional)

# CAD preprocessing
CAD_UNITS_TO_METERS = 0.001  # Set to 1.0 if CAD is already in meters

# ICP settings (set to False to skip ICP entirely)
USE_ICP_REFINEMENT = True
ICP_MAX_CORRESPONDENCE_DIST = 0.05   # meters
ICP_MAX_ITERATIONS = 100
ICP_RELATIVE_FITNESS = 1e-6
ICP_RELATIVE_RMSE = 1e-6
CAD_SAMPLE_POINTS = 50000            # Points to sample from mesh for ICP
SCENE_VOXEL_SIZE = 0.005             # Downsample scene for speed

# Export paths
EXPORT_DIR = Path("../../captures/captures_colorworld_stable/capture_20251121_162722/transformed_cad")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_CAD_TRANSFORMED = EXPORT_DIR / "cad_transformed_20251121_162722.ply"       # After 6DOF only
EXPORT_CAD_ICP_REFINED = EXPORT_DIR / "cad_icp_refined_20251121_162722.ply"       # After ICP (if enabled)
EXPORT_META_JSON = EXPORT_DIR / "transform_meta.json"

# Visualization
SHOW_VISUALIZATION = True


# =========================================================
#                     UTILITIES
# =========================================================

def load_transform_matrix(txt_path: Path) -> np.ndarray:
    """Load a 4x4 transformation matrix from a text file."""
    if not txt_path.exists():
        raise FileNotFoundError(f"Transform file not found: {txt_path}")
    
    T = np.loadtxt(str(txt_path))
    
    if T.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {T.shape}")
    
    # Sanity check: last row should be [0, 0, 0, 1]
    expected_last_row = np.array([0, 0, 0, 1])
    if not np.allclose(T[3, :], expected_last_row, atol=1e-6):
        print(f"[WARN] Last row is {T[3, :]}, expected [0, 0, 0, 1]")
    
    return T


def load_cad_geometry(path: Path):
    """Load CAD as mesh or point cloud."""
    if not path.exists():
        raise FileNotFoundError(f"CAD file not found: {path}")
    
    # Try mesh first
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh and len(np.asarray(mesh.vertices)) > 0:
        mesh.compute_vertex_normals()
        print(f"[CAD] Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        return mesh
    
    # Fall back to point cloud
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd and len(np.asarray(pcd.points)) > 0:
        print(f"[CAD] Loaded point cloud: {len(pcd.points)} points")
        return pcd
    
    raise RuntimeError(f"Failed to load CAD from {path}")


def load_scene_pcd(path: Path):
    """Load scene point cloud."""
    if not path.exists():
        print(f"[WARN] Scene file not found: {path}")
        return None
    
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd and len(np.asarray(pcd.points)) > 0:
        print(f"[SCENE] Loaded: {len(pcd.points)} points")
        return pcd
    
    print(f"[WARN] Failed to load scene from {path}")
    return None


def refine_with_icp(cad_geometry, scene_pcd, max_correspondence_dist, max_iterations):
    """Run point-to-plane ICP to refine CAD alignment."""
    
    # Convert mesh to point cloud if needed
    if isinstance(cad_geometry, o3d.geometry.TriangleMesh):
        print(f"[ICP] Sampling {CAD_SAMPLE_POINTS} points from CAD mesh")
        cad_pcd = cad_geometry.sample_points_uniformly(number_of_points=CAD_SAMPLE_POINTS)
    else:
        cad_pcd = cad_geometry

    # Downsample scene
    scene_down = scene_pcd.voxel_down_sample(voxel_size=SCENE_VOXEL_SIZE)
    print(f"[ICP] Scene: {len(scene_pcd.points)} → {len(scene_down.points)} points")
    print(f"[ICP] CAD: {len(cad_pcd.points)} points")

    # Estimate normals
    scene_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    cad_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )

    print(f"[ICP] Running point-to-plane ICP (max_dist={max_correspondence_dist}m, max_iter={max_iterations})")

    reg = o3d.pipelines.registration.registration_icp(
        cad_pcd, scene_down,
        max_correspondence_dist,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations,
            relative_fitness=ICP_RELATIVE_FITNESS,
            relative_rmse=ICP_RELATIVE_RMSE
        )
    )

    # Report results
    print(f"[ICP] Fitness: {reg.fitness:.4f}, Inlier RMSE: {reg.inlier_rmse:.6f} m")
    
    # Decompose correction for readability
    R_corr = reg.transformation[:3, :3]
    t_corr = reg.transformation[:3, 3]
    angle_rad = np.arccos(np.clip((np.trace(R_corr) - 1) / 2, -1, 1))
    angle_deg = np.degrees(angle_rad)
    trans_mm = np.linalg.norm(t_corr) * 1000
    print(f"[ICP] Correction: rotation={angle_deg:.2f}°, translation={trans_mm:.2f}mm")

    return reg.transformation, reg.fitness, reg.inlier_rmse


def save_geometry(geometry, path: Path):
    """Save mesh or point cloud."""
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        ok = o3d.io.write_triangle_mesh(str(path), geometry, write_vertex_normals=True)
    else:
        ok = o3d.io.write_point_cloud(str(path), geometry)
    
    if not ok:
        raise RuntimeError(f"Failed to write {path}")
    print(f"[SAVE] {path}")


def copy_geometry(geometry):
    """Deep copy a geometry object."""
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        new_geo = o3d.geometry.TriangleMesh()
        new_geo.vertices = o3d.utility.Vector3dVector(np.asarray(geometry.vertices).copy())
        new_geo.triangles = o3d.utility.Vector3iVector(np.asarray(geometry.triangles).copy())
        if geometry.has_vertex_normals():
            new_geo.vertex_normals = o3d.utility.Vector3dVector(np.asarray(geometry.vertex_normals).copy())
        if geometry.has_vertex_colors():
            new_geo.vertex_colors = o3d.utility.Vector3dVector(np.asarray(geometry.vertex_colors).copy())
        return new_geo
    else:
        new_geo = o3d.geometry.PointCloud()
        new_geo.points = o3d.utility.Vector3dVector(np.asarray(geometry.points).copy())
        if geometry.has_normals():
            new_geo.normals = o3d.utility.Vector3dVector(np.asarray(geometry.normals).copy())
        if geometry.has_colors():
            new_geo.colors = o3d.utility.Vector3dVector(np.asarray(geometry.colors).copy())
        return new_geo


# =========================================================
#                        MAIN
# =========================================================

def main():
    print("=" * 60)
    print("6DOF Transform Application")
    print("=" * 60)
    
    # ---------- Load inputs ----------
    T_6dof = load_transform_matrix(TRANSFORM_TXT)
    print(f"\n[TRANSFORM] Loaded from {TRANSFORM_TXT}:")
    print(T_6dof)
    
    # Extract rotation and translation for display
    R = T_6dof[:3, :3]
    t = T_6dof[:3, 3]
    print(f"\n  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
    
    cad = load_cad_geometry(CAD_PLY)
    
    scene_pcd = None
    if USE_ICP_REFINEMENT:
        scene_pcd = load_scene_pcd(SCENE_PLY)
    
    # ---------- Scale CAD if needed ----------
    if abs(CAD_UNITS_TO_METERS - 1.0) > 1e-9:
        print(f"\n[CAD] Scaling by {CAD_UNITS_TO_METERS} (units → meters)")
        cad.scale(CAD_UNITS_TO_METERS, center=(0, 0, 0))
    
    # ---------- Apply 6DOF transform ----------
    print("\n[CAD] Applying 6DOF transformation...")
    cad.transform(T_6dof)
    
    # Save a copy for the non-ICP version
    cad_transformed_only = copy_geometry(cad)
    
    # Export transformed CAD (before ICP)
    save_geometry(cad_transformed_only, EXPORT_CAD_TRANSFORMED)
    
    T_icp = None
    icp_fitness = None
    icp_rmse = None
    
    # ---------- Optional ICP refinement ----------
    # ============================================================
    # COMMENT OUT THIS BLOCK TO SKIP ICP
    # ============================================================
    if USE_ICP_REFINEMENT:
        if scene_pcd is not None:
            print("\n" + "=" * 60)
            print("[ICP] Starting refinement...")
            print("=" * 60)
            
            T_icp, icp_fitness, icp_rmse = refine_with_icp(
                cad, scene_pcd,
                ICP_MAX_CORRESPONDENCE_DIST,
                ICP_MAX_ITERATIONS
            )
            
            # Apply ICP correction
            cad.transform(T_icp)
            
            # Export ICP-refined CAD
            save_geometry(cad, EXPORT_CAD_ICP_REFINED)
        else:
            print("\n[ICP] Skipped: scene point cloud not available")
    else:
        print("\n[ICP] Disabled (USE_ICP_REFINEMENT = False)")
    # ============================================================
    # END OF ICP BLOCK
    # ============================================================
    
    # ---------- Export metadata ----------
    meta = {
        "inputs": {
            "cad_source": str(CAD_PLY),
            "transform_file": str(TRANSFORM_TXT),
            "scene_ply": str(SCENE_PLY) if scene_pcd else None,
        },
        "parameters": {
            "cad_units_to_meters": float(CAD_UNITS_TO_METERS),
            "use_icp": bool(USE_ICP_REFINEMENT),
            "icp_max_corr_dist_m": float(ICP_MAX_CORRESPONDENCE_DIST) if USE_ICP_REFINEMENT else None,
            "icp_max_iters": int(ICP_MAX_ITERATIONS) if USE_ICP_REFINEMENT else None,
        },
        "transforms": {
            "T_6dof_from_file": T_6dof.tolist(),
            "T_icp_correction": T_icp.tolist() if T_icp is not None else None,
            "T_final": (T_icp @ T_6dof).tolist() if T_icp is not None else T_6dof.tolist(),
        },
        "icp_results": {
            "fitness": float(icp_fitness) if icp_fitness is not None else None,
            "inlier_rmse_m": float(icp_rmse) if icp_rmse is not None else None,
        },
        "outputs": {
            "cad_transformed": str(EXPORT_CAD_TRANSFORMED),
            "cad_icp_refined": str(EXPORT_CAD_ICP_REFINED) if T_icp is not None else None,
        }
    }
    
    EXPORT_META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"[SAVE] {EXPORT_META_JSON}")
    
    # ---------- Visualization ----------
    if SHOW_VISUALIZATION:
        print("\n[VIZ] Launching viewer...")
        geoms = []
        
        # Show transformed CAD (use ICP version if available)
        geoms.append(cad)
        
        # Show scene if available
        if scene_pcd is not None:
            geoms.append(scene_pcd)
        
        # Add coordinate axes at origin
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        geoms.append(axes)
        
        o3d.visualization.draw_geometries(geoms, window_name="Transformed CAD")
    
    print("\n[DONE]")


if __name__ == "__main__":
    main()