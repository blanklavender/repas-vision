import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree

def compute_point_to_mesh_distances(point_cloud, mesh):
    """
    Compute distance from each point cloud point to nearest surface on mesh.
    Falls back to point-to-point if mesh has no faces.
    """
    points_pcd = np.asarray(point_cloud.points)
    
    # Check if mesh actually has triangles
    num_triangles = len(np.asarray(mesh.triangles))
    
    if num_triangles == 0:
        print(f"  [WARN] Mesh has no faces! Using point-to-point distance instead.")
        # Treat CAD as point cloud
        cad_points = np.asarray(mesh.vertices)
        if len(cad_points) == 0:
            raise ValueError("CAD model has no vertices!")
        
        tree = cKDTree(cad_points)
        distances, _ = tree.query(points_pcd, k=1)
        return distances
    
    # Original raycasting method for actual meshes
    try:
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_t)
        
        query_points = o3d.core.Tensor(points_pcd, dtype=o3d.core.Dtype.Float32)
        signed_distances = scene.compute_signed_distance(query_points).numpy()
        distances = np.abs(signed_distances)
        
        return distances
    except Exception as e:
        print(f"  Raycasting method failed: {e}")
        print(f"  Falling back to KDTree method...")
        
        mesh_pcd = mesh.sample_points_uniformly(number_of_points=100000)
        mesh_points = np.asarray(mesh_pcd.points)
        
        if len(mesh_points) == 0:
            raise ValueError("Failed to sample points from mesh!")
        
        tree = cKDTree(mesh_points)
        distances, _ = tree.query(points_pcd, k=1)
        
        return distances

def create_error_colormap():
    """
    Create a colormap for error visualization
    Green (good) -> Yellow (medium) -> Red (bad)
    """
    colors = ['green', 'yellow', 'orange', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('error', colors, N=n_bins)
    return cmap

def color_point_cloud_by_error(point_cloud, distances, max_error=None, percentile=95):
    """
    Color point cloud based on error distances
    
    Args:
        point_cloud: Open3D PointCloud
        distances: Array of distances
        max_error: Maximum error for color scale (None = auto from percentile)
        percentile: Percentile to use for max_error if not specified
        
    Returns:
        Colored point cloud
    """
    if max_error is None:
        max_error = np.percentile(distances, percentile)
    
    # Normalize distances to 0-1 range
    normalized_distances = np.clip(distances / max_error, 0, 1)
    
    # Create colormap
    cmap = create_error_colormap()
    
    # Map distances to colors
    colors = cmap(normalized_distances)[:, :3]  # RGB only, no alpha
    
    # Apply colors to point cloud
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return point_cloud, max_error

def print_error_statistics(distances, unit_label="mm"):
    """
    Print comprehensive error statistics
    """
    print("\n" + "=" * 80)
    print("POINT-TO-SURFACE DISTANCE STATISTICS")
    print("=" * 80)
    
    print(f"\nDistance from Point Cloud to CAD Surface:")
    print(f"  Mean:          {np.mean(distances):>10.4f} {unit_label}")
    print(f"  Median:        {np.median(distances):>10.4f} {unit_label}")
    print(f"  Std Dev:       {np.std(distances):>10.4f} {unit_label}")
    print(f"  Min:           {np.min(distances):>10.4f} {unit_label}")
    print(f"  Max:           {np.max(distances):>10.4f} {unit_label}")
    print(f"  RMSE:          {np.sqrt(np.mean(distances**2)):>10.4f} {unit_label}")
    
    print(f"\nPercentile Analysis:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(distances, p):>10.4f} {unit_label}")
    
    # Quality assessment
    mean_error = np.mean(distances)
    p95_error = np.percentile(distances, 95)
    
    print(f"\n" + "-" * 80)
    print("ALIGNMENT QUALITY ASSESSMENT")
    print("-" * 80)
    
    print(f"\n  Mean Error:   {mean_error:.4f} {unit_label}")
    print(f"  95% Error:    {p95_error:.4f} {unit_label}")
    
    if mean_error < 5:
        quality = "EXCELLENT"
        desc = "Very tight alignment"
    elif mean_error < 10:
        quality = "GOOD"
        desc = "Acceptable alignment"
    elif mean_error < 20:
        quality = "FAIR"
        desc = "Moderate alignment errors"
    else:
        quality = "POOR"
        desc = "Significant alignment errors"
    
    print(f"  Quality:      {quality} - {desc}")
    
    # Point distribution analysis
    excellent = np.sum(distances < 5)
    good = np.sum((distances >= 5) & (distances < 10))
    fair = np.sum((distances >= 10) & (distances < 20))
    poor = np.sum(distances >= 20)
    total = len(distances)
    
    print(f"\n  Point Distribution:")
    print(f"    Excellent (< 5 {unit_label}):   {excellent:>7} ({100*excellent/total:>5.1f}%)")
    print(f"    Good (5-10 {unit_label}):       {good:>7} ({100*good/total:>5.1f}%)")
    print(f"    Fair (10-20 {unit_label}):      {fair:>7} ({100*fair/total:>5.1f}%)")
    print(f"    Poor (> 20 {unit_label}):       {poor:>7} ({100*poor/total:>5.1f}%)")
    
    print("=" * 80)

def create_error_histogram(distances, unit_label="mm", output_file="error_histogram.png"):
    """
    Create and save histogram of error distribution
    """
    plt.figure(figsize=(12, 6))
    
    # Main histogram
    plt.subplot(1, 2, 1)
    plt.hist(distances, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.2f}')
    plt.axvline(np.median(distances), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(distances):.2f}')
    plt.axvline(np.percentile(distances, 95), color='orange', linestyle='--', linewidth=2, label=f'95th: {np.percentile(distances, 95):.2f}')
    plt.xlabel(f'Distance ({unit_label})')
    plt.ylabel('Number of Points')
    plt.title('Point-to-Surface Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(1, 2, 2)
    sorted_distances = np.sort(distances)
    cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
    plt.plot(sorted_distances, cumulative, linewidth=2, color='steelblue')
    plt.axhline(50, color='green', linestyle='--', alpha=0.5, label='50%')
    plt.axhline(95, color='orange', linestyle='--', alpha=0.5, label='95%')
    plt.axvline(np.median(distances), color='green', linestyle='--', alpha=0.5)
    plt.axvline(np.percentile(distances, 95), color='orange', linestyle='--', alpha=0.5)
    plt.xlabel(f'Distance ({unit_label})')
    plt.ylabel('Cumulative Percentage (%)')
    plt.title('Cumulative Distribution of Errors')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nHistogram saved to: {output_file}")
    plt.close()

def create_color_scale_reference(max_error, unit_label="mm", output_file="color_scale.png"):
    """
    Create a reference image showing the color scale
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Create gradient
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    
    cmap = create_error_colormap()
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    
    # Add labels
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.set_xticklabels([f'{0:.1f}', f'{max_error*0.25:.1f}', f'{max_error*0.5:.1f}', 
                        f'{max_error*0.75:.1f}', f'{max_error:.1f}'])
    ax.set_yticks([])
    ax.set_xlabel(f'Distance to CAD Surface ({unit_label})', fontsize=12)
    ax.set_title('Color Scale: Green (Good Alignment) → Red (Poor Alignment)', fontsize=14, pad=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Color scale reference saved to: {output_file}")
    plt.close()

def visualize_alignment(point_cloud_colored, cad_mesh, show_cad=True):
    """
    Visualize the colored point cloud with optional CAD mesh
    """
    geometries = []
    
    # Add colored point cloud
    geometries.append(point_cloud_colored)
    
    # Add CAD mesh (semi-transparent gray)
    if show_cad and cad_mesh is not None:
        cad_mesh_copy = o3d.geometry.TriangleMesh(cad_mesh)
        cad_mesh_copy.paint_uniform_color([0.7, 0.7, 0.7])
        cad_mesh_copy.compute_vertex_normals()
        geometries.append(cad_mesh_copy)
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
    geometries.append(coord_frame)
    
    print("\nVisualization Controls:")
    print("  - Rotate: Left mouse button")
    print("  - Pan: Middle mouse button or Ctrl + Left mouse")
    print("  - Zoom: Scroll wheel")
    print("  - Toggle CAD mesh: Press 'K' (if supported)")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Point Cloud Alignment Quality (Color-coded by Error)",
        width=1400,
        height=900,
        mesh_show_back_face=True
    )

def main():
    """
    Main function
    """
    print("\n" + "=" * 80)
    print("POINT CLOUD TO CAD ALIGNMENT VISUALIZATION")
    print("=" * 80 + "\n")
    
    # ============================================
    # MODIFY THESE PATHS
    # ============================================
    
    # Path to point cloud (captured data - ground truth)
    point_cloud_file = "../../captures/captures_colorworld_stable/capture_20251121_162722/ground_truth_scene/cropped_camframe_20251121_162722.ply"
    
    # # Path to transformed CAD model
    cad_mesh_file = "../../captures/captures_colorworld_stable/capture_20251121_162722/transformed_cad/cad_icp_refined_20251121_162722.ply"
    # cad_mesh_file = "../../captures/captures_colorworld_stable/capture_20251121_162722/transformed_cad/cad_transformed_20251121_162722.ply"
    
    # Unit conversion
    UNIT_SCALE = 1000.0  # meters to millimeters
    UNIT_LABEL = "mm"
    
    # Visualization settings
    MAX_ERROR_FOR_COLORMAP = None  # None = auto-compute from 95th percentile, or set value like 20.0
    PERCENTILE_FOR_MAX = 95       # Use 95th percentile for color scale
    SHOW_CAD_MESH = True          # Show CAD mesh alongside colored point cloud
    
    # Performance settings
    DOWNSAMPLE_VOXEL_SIZE = None  # None = no downsampling, or set value like 2.0 (mm) for faster computation
    MAX_POINTS = 50000            # Maximum points to sample from mesh files
    
    # ============================================
    
    # Load point cloud
    print(f"Loading point cloud: {point_cloud_file}")
    file_ext = Path(point_cloud_file).suffix.lower()
    
    if file_ext in ['.stl', '.obj']:
        # Load as mesh and convert to point cloud
        mesh_temp = o3d.io.read_triangle_mesh(point_cloud_file)
        point_cloud = mesh_temp.sample_points_uniformly(number_of_points=MAX_POINTS)
        print(f"  Converted mesh to point cloud: {len(point_cloud.points)} points")
    else:
        point_cloud = o3d.io.read_point_cloud(point_cloud_file)
        print(f"  Loaded point cloud: {len(point_cloud.points)} points")
    
    # Apply unit scaling
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) * UNIT_SCALE)
    
    # Optional downsampling for faster computation
    if DOWNSAMPLE_VOXEL_SIZE is not None:
        print(f"\nDownsampling point cloud (voxel size: {DOWNSAMPLE_VOXEL_SIZE} {UNIT_LABEL})...")
        point_cloud_original = point_cloud
        point_cloud = point_cloud.voxel_down_sample(voxel_size=DOWNSAMPLE_VOXEL_SIZE)
        print(f"  Points after downsampling: {len(point_cloud.points)} (was {len(point_cloud_original.points)})")
    
    print(f"\nFinal point cloud: {len(point_cloud.points)} points")
    
    # Load CAD mesh
    print(f"\nLoading CAD mesh: {cad_mesh_file}")
    cad_mesh = o3d.io.read_triangle_mesh(cad_mesh_file)
    print(f"  Loaded mesh: {len(cad_mesh.vertices)} vertices, {len(cad_mesh.triangles)} triangles")
    
    # Apply unit scaling
    cad_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(cad_mesh.vertices) * UNIT_SCALE)
    
    # Compute vertex normals for better rendering
    cad_mesh.compute_vertex_normals()
    
    # Compute point-to-surface distances
    print(f"\nComputing point-to-surface distances...")
    print(f"  This may take a moment for large point clouds...")
    
    distances = compute_point_to_mesh_distances(point_cloud, cad_mesh)
    
    print(f"  Computed distances for {len(distances)} points")
    
    # Print statistics
    print_error_statistics(distances, unit_label=UNIT_LABEL)
    
    # Create histogram
    create_error_histogram(distances, unit_label=UNIT_LABEL)
    
    # Color point cloud by error
    print(f"\nColoring point cloud by alignment error...")
    point_cloud_colored, max_error_used = color_point_cloud_by_error(
        point_cloud, 
        distances, 
        max_error=MAX_ERROR_FOR_COLORMAP,
        percentile=PERCENTILE_FOR_MAX
    )
    
    print(f"  Color scale range: 0 to {max_error_used:.2f} {UNIT_LABEL}")
    print(f"  Green = excellent alignment (< {max_error_used*0.25:.1f} {UNIT_LABEL})")
    print(f"  Yellow = good alignment ({max_error_used*0.25:.1f} - {max_error_used*0.5:.1f} {UNIT_LABEL})")
    print(f"  Orange = fair alignment ({max_error_used*0.5:.1f} - {max_error_used*0.75:.1f} {UNIT_LABEL})")
    print(f"  Red = poor alignment (> {max_error_used*0.75:.1f} {UNIT_LABEL})")
    
    # Create color scale reference
    create_color_scale_reference(max_error_used, unit_label=UNIT_LABEL)
    
    # Save colored point cloud
    output_pcd_file = "../../captures/captures_colorworld_stable/capture_20251121_162722/error_analysis/alignment_quality_colored.ply"
    o3d.io.write_point_cloud(output_pcd_file, point_cloud_colored)
    print(f"\nColored point cloud saved to: {output_pcd_file}")
    
    # Save error data
    output_data_file = "../../captures/captures_colorworld_stable/capture_20251121_162722/error_analysis/alignment_errors.txt"
    with open(output_data_file, 'w', encoding='utf-8') as f:
        f.write("Point-to-Surface Distance Analysis\n")
        f.write("Point Cloud (Ground Truth) to CAD Model\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Points Analyzed: {len(distances)}\n\n")
        
        f.write("Statistics:\n")
        f.write(f"  Mean:     {np.mean(distances):.6f} {UNIT_LABEL}\n")
        f.write(f"  Median:   {np.median(distances):.6f} {UNIT_LABEL}\n")
        f.write(f"  Std Dev:  {np.std(distances):.6f} {UNIT_LABEL}\n")
        f.write(f"  Min:      {np.min(distances):.6f} {UNIT_LABEL}\n")
        f.write(f"  Max:      {np.max(distances):.6f} {UNIT_LABEL}\n")
        f.write(f"  RMSE:     {np.sqrt(np.mean(distances**2)):.6f} {UNIT_LABEL}\n\n")
        
        f.write("Percentiles:\n")
        for p in [50, 75, 90, 95, 99]:
            f.write(f"  {p}th: {np.percentile(distances, p):.6f} {UNIT_LABEL}\n")
    
    print(f"Error statistics saved to: {output_data_file}")
    
    # Visualize
    visualize_choice = input("\nVisualize colored point cloud? (y/n): ").lower().strip()
    
    if visualize_choice == 'y':
        visualize_alignment(point_cloud_colored, cad_mesh, show_cad=SHOW_CAD_MESH)

if __name__ == "__main__":
    main()