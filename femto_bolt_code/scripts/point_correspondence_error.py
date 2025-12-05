import numpy as np
import xml.etree.ElementTree as ET
import open3d as o3d
from pathlib import Path

def load_picked_points(pp_file_path):
    """
    Load picked points from Open3D's .pp file format (XML)
    
    Args:
        pp_file_path: Path to the .pp file
        
    Returns:
        numpy array of picked points (N x 3)
        list of point names
    """
    tree = ET.parse(pp_file_path)
    root = tree.getroot()
    
    points = []
    names = []
    
    for point in root.findall('point'):
        x = float(point.get('x'))
        y = float(point.get('y'))
        z = float(point.get('z'))
        name = point.get('name', f'Point_{len(points)}')
        
        points.append([x, y, z])
        names.append(name)
    
    return np.array(points), names

def calculate_distances(points_a, points_b):
    """
    Calculate Euclidean and Manhattan distances between corresponding points
    
    Args:
        points_a: First set of points (N x 3)
        points_b: Second set of points (N x 3)
        
    Returns:
        euclidean_distances: Array of Euclidean distances
        manhattan_distances: Array of Manhattan distances
    """
    if len(points_a) != len(points_b):
        raise ValueError(f"Number of points must match: {len(points_a)} vs {len(points_b)}")
    
    # Calculate differences
    diff = points_a - points_b
    
    # Euclidean distance (L2 norm)
    euclidean_distances = np.linalg.norm(diff, axis=1)
    
    # Manhattan distance (L1 norm)
    manhattan_distances = np.sum(np.abs(diff), axis=1)
    
    return euclidean_distances, manhattan_distances

def print_error_analysis(points_a, points_b, labels_a="Point Cloud", labels_b="CAD Model", 
                        names_a=None, names_b=None, unit_label="mm"):
    """
    Print comprehensive error analysis with point cloud as reference
    
    Args:
        points_a: First set of points (N x 3) - REFERENCE (Point Cloud)
        points_b: Second set of points (N x 3) - TARGET (CAD Model)
        labels_a: Label for first set
        labels_b: Label for second set
        names_a: Optional list of names for points_a
        names_b: Optional list of names for points_b
        unit_label: Unit label for distances (default: mm)
    """
    euclidean_dist, manhattan_dist = calculate_distances(points_a, points_b)
    
    # Calculate displacement (error = CAD - Point Cloud)
    # Positive value means CAD is further in that direction
    displacement = points_b - points_a
    
    print("=" * 100)
    print(f"DETAILED ERROR ANALYSIS: {labels_b} Error from {labels_a} (Reference)")
    print("=" * 100)
    print(f"\nNote: {labels_a} is treated as GROUND TRUTH (accurate reference)")
    print(f"      Displacement = {labels_b} - {labels_a}")
    print(f"      Positive displacement means {labels_b} is further in that direction")
    
    print("\n" + "=" * 100)
    print("PER-POINT ERROR BREAKDOWN")
    print("=" * 100)
    
    for i in range(len(points_a)):
        point_name = names_a[i] if names_a else (names_b[i] if names_b else f"Point {i}")
        
        print(f"\n{'='*100}")
        print(f"Point {i}: {point_name}")
        print(f"{'='*100}")
        
        print(f"\n  {labels_a} Coordinates (Reference):")
        print(f"    X: {points_a[i, 0]:>10.4f} {unit_label}")
        print(f"    Y: {points_a[i, 1]:>10.4f} {unit_label}")
        print(f"    Z: {points_a[i, 2]:>10.4f} {unit_label}")
        
        print(f"\n  {labels_b} Coordinates:")
        print(f"    X: {points_b[i, 0]:>10.4f} {unit_label}")
        print(f"    Y: {points_b[i, 1]:>10.4f} {unit_label}")
        print(f"    Z: {points_b[i, 2]:>10.4f} {unit_label}")
        
        print(f"\n  Displacement ({labels_b} - {labels_a}):")
        print(f"    ΔX: {displacement[i, 0]:>10.4f} {unit_label}  {'(CAD further +X)' if displacement[i, 0] > 0 else '(CAD closer -X)' if displacement[i, 0] < 0 else '(aligned)'}")
        print(f"    ΔY: {displacement[i, 1]:>10.4f} {unit_label}  {'(CAD further +Y)' if displacement[i, 1] > 0 else '(CAD closer -Y)' if displacement[i, 1] < 0 else '(aligned)'}")
        print(f"    ΔZ: {displacement[i, 2]:>10.4f} {unit_label}  {'(CAD further +Z)' if displacement[i, 2] > 0 else '(CAD closer -Z)' if displacement[i, 2] < 0 else '(aligned)'}")
        
        print(f"\n  Error Magnitudes:")
        print(f"    Euclidean Distance: {euclidean_dist[i]:>10.4f} {unit_label}")
        print(f"    Manhattan Distance: {manhattan_dist[i]:>10.4f} {unit_label}")
        
        # Dominant error direction
        abs_disp = np.abs(displacement[i])
        dominant_axis = ['X', 'Y', 'Z'][np.argmax(abs_disp)]
        dominant_value = displacement[i, np.argmax(abs_disp)]
        print(f"\n  Dominant Error Direction: {dominant_axis}-axis ({dominant_value:+.4f} {unit_label})")
    
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS (All Points)")
    print("=" * 100)
    
    print(f"\nEuclidean Distance (L2 norm):")
    print(f"  Mean:     {np.mean(euclidean_dist):>10.4f} {unit_label}")
    print(f"  Std Dev:  {np.std(euclidean_dist):>10.4f} {unit_label}")
    print(f"  Min:      {np.min(euclidean_dist):>10.4f} {unit_label}  (Point {np.argmin(euclidean_dist)}: {names_a[np.argmin(euclidean_dist)] if names_a else ''})")
    print(f"  Max:      {np.max(euclidean_dist):>10.4f} {unit_label}  (Point {np.argmax(euclidean_dist)}: {names_a[np.argmax(euclidean_dist)] if names_a else ''})")
    print(f"  Median:   {np.median(euclidean_dist):>10.4f} {unit_label}")
    print(f"  RMSE:     {np.sqrt(np.mean(euclidean_dist**2)):>10.4f} {unit_label}")
    
    print(f"\nManhattan Distance (L1 norm):")
    print(f"  Mean:     {np.mean(manhattan_dist):>10.4f} {unit_label}")
    print(f"  Std Dev:  {np.std(manhattan_dist):>10.4f} {unit_label}")
    print(f"  Min:      {np.min(manhattan_dist):>10.4f} {unit_label}")
    print(f"  Max:      {np.max(manhattan_dist):>10.4f} {unit_label}")
    print(f"  Median:   {np.median(manhattan_dist):>10.4f} {unit_label}")
    
    print(f"\n" + "-" * 100)
    print("DISPLACEMENT STATISTICS (Signed - shows systematic bias)")
    print("-" * 100)
    
    print(f"\nX-Axis Displacement:")
    print(f"  Mean:     {np.mean(displacement[:, 0]):>10.4f} {unit_label}  {'(CAD shifted +X)' if np.mean(displacement[:, 0]) > 0 else '(CAD shifted -X)' if np.mean(displacement[:, 0]) < 0 else '(centered)'}")
    print(f"  Std Dev:  {np.std(displacement[:, 0]):>10.4f} {unit_label}")
    print(f"  Min:      {np.min(displacement[:, 0]):>10.4f} {unit_label}")
    print(f"  Max:      {np.max(displacement[:, 0]):>10.4f} {unit_label}")
    
    print(f"\nY-Axis Displacement:")
    print(f"  Mean:     {np.mean(displacement[:, 1]):>10.4f} {unit_label}  {'(CAD shifted +Y)' if np.mean(displacement[:, 1]) > 0 else '(CAD shifted -Y)' if np.mean(displacement[:, 1]) < 0 else '(centered)'}")
    print(f"  Std Dev:  {np.std(displacement[:, 1]):>10.4f} {unit_label}")
    print(f"  Min:      {np.min(displacement[:, 1]):>10.4f} {unit_label}")
    print(f"  Max:      {np.max(displacement[:, 1]):>10.4f} {unit_label}")
    
    print(f"\nZ-Axis Displacement:")
    print(f"  Mean:     {np.mean(displacement[:, 2]):>10.4f} {unit_label}  {'(CAD shifted +Z)' if np.mean(displacement[:, 2]) > 0 else '(CAD shifted -Z)' if np.mean(displacement[:, 2]) < 0 else '(centered)'}")
    print(f"  Std Dev:  {np.std(displacement[:, 2]):>10.4f} {unit_label}")
    print(f"  Min:      {np.min(displacement[:, 2]):>10.4f} {unit_label}")
    print(f"  Max:      {np.max(displacement[:, 2]):>10.4f} {unit_label}")
    
    print(f"\n" + "-" * 100)
    print("ABSOLUTE DISPLACEMENT STATISTICS (Unsigned - shows scatter)")
    print("-" * 100)
    
    abs_disp = np.abs(displacement)
    print(f"\nX-Axis (Absolute):")
    print(f"  Mean:     {np.mean(abs_disp[:, 0]):>10.4f} {unit_label}")
    print(f"  Std Dev:  {np.std(abs_disp[:, 0]):>10.4f} {unit_label}")
    print(f"  Max:      {np.max(abs_disp[:, 0]):>10.4f} {unit_label}")
    
    print(f"\nY-Axis (Absolute):")
    print(f"  Mean:     {np.mean(abs_disp[:, 1]):>10.4f} {unit_label}")
    print(f"  Std Dev:  {np.std(abs_disp[:, 1]):>10.4f} {unit_label}")
    print(f"  Max:      {np.max(abs_disp[:, 1]):>10.4f} {unit_label}")
    
    print(f"\nZ-Axis (Absolute):")
    print(f"  Mean:     {np.mean(abs_disp[:, 2]):>10.4f} {unit_label}")
    print(f"  Std Dev:  {np.std(abs_disp[:, 2]):>10.4f} {unit_label}")
    print(f"  Max:      {np.max(abs_disp[:, 2]):>10.4f} {unit_label}")
    
    # Overall assessment
    print(f"\n" + "=" * 100)
    print("ALIGNMENT ASSESSMENT")
    print("=" * 100)
    
    mean_error = np.mean(euclidean_dist)
    max_error = np.max(euclidean_dist)
    
    print(f"\n  Mean Error: {mean_error:.4f} {unit_label}")
    print(f"  Max Error:  {max_error:.4f} {unit_label}")
    
    if mean_error < 5:
        print(f"  Quality: EXCELLENT - Very tight alignment")
    elif mean_error < 10:
        print(f"  Quality: GOOD - Acceptable alignment")
    elif mean_error < 20:
        print(f"  Quality: FAIR - Moderate alignment errors")
    else:
        print(f"  Quality: POOR - Significant alignment errors")
    
    # Check for systematic bias
    mean_disp = np.mean(displacement, axis=0)
    systematic_bias = np.linalg.norm(mean_disp)
    
    print(f"\n  Systematic Bias: {systematic_bias:.4f} {unit_label}")
    print(f"    Mean shift: [{mean_disp[0]:+.4f}, {mean_disp[1]:+.4f}, {mean_disp[2]:+.4f}] {unit_label}")
    
    if systematic_bias > mean_error * 0.5:
        print(f"  NOTE: Large systematic bias detected - CAD model may be globally offset")
    
    print("=" * 100)
    
    return euclidean_dist, manhattan_dist, displacement

def load_geometry_file(file_path):
    """
    Load a geometry file (mesh or point cloud) with automatic format detection
    
    Args:
        file_path: Path to geometry file
        
    Returns:
        Open3D geometry object (TriangleMesh or PointCloud)
    """
    if not Path(file_path).exists():
        print(f"  WARNING: File not found: {file_path}")
        return None
    
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext in ['.stl', '.obj', '.off', '.gltf', '.glb']:
            # Load as mesh
            geom = o3d.io.read_triangle_mesh(file_path)
            if len(geom.vertices) == 0:
                print(f"  WARNING: Mesh has 0 vertices")
                return None
            print(f"  Loaded mesh with {len(geom.vertices)} vertices, {len(geom.triangles)} triangles")
            return geom
        elif file_ext in ['.ply', '.pcd', '.xyz', '.xyzn', '.xyzrgb', '.pts']:
            # Try as point cloud first
            try:
                geom = o3d.io.read_point_cloud(file_path)
                if len(geom.points) > 0:
                    print(f"  Loaded point cloud with {len(geom.points)} points")
                    return geom
            except:
                pass
            
            # If point cloud fails, try as mesh
            geom = o3d.io.read_triangle_mesh(file_path)
            if len(geom.vertices) == 0:
                print(f"  WARNING: File has 0 vertices/points")
                return None
            print(f"  Loaded mesh with {len(geom.vertices)} vertices, {len(geom.triangles)} triangles")
            return geom
        else:
            print(f"  WARNING: Unknown file format: {file_ext}")
            return None
    except Exception as e:
        print(f"  ERROR loading file: {e}")
        return None

def visualize_correspondences(points_a, points_b, pcd=None, cad_mesh=None):
    """
    Visualize point correspondences with lines connecting them
    
    Args:
        points_a: First set of points (N x 3)
        points_b: Second set of points (N x 3)
        pcd: Optional point cloud/mesh to display
        cad_mesh: Optional CAD mesh to display
    """
    geometries = []
    
    # Add point cloud/mesh if provided - paint it GRAY
    if pcd is not None:
        # Remove any existing colors and paint gray
        if isinstance(pcd, o3d.geometry.TriangleMesh):
            pcd.vertex_colors = o3d.utility.Vector3dVector([])  # Clear vertex colors
            pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
        else:
            pcd.colors = o3d.utility.Vector3dVector([])  # Clear colors
            pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
        geometries.append(pcd)
    
    # Add CAD mesh if provided - paint it GRAY
    if cad_mesh is not None:
        # Remove any existing colors and paint gray
        if isinstance(cad_mesh, o3d.geometry.TriangleMesh):
            cad_mesh.vertex_colors = o3d.utility.Vector3dVector([])  # Clear vertex colors
            cad_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
        else:
            cad_mesh.colors = o3d.utility.Vector3dVector([])  # Clear colors
            cad_mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
        geometries.append(cad_mesh)
    
    # Create SPHERES for picked points (more visible than point clouds)
    sphere_radius = 5.0  # 5mm spheres - adjust if needed
    
    # Red spheres for point cloud picks
    for point in points_a:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(point)
        sphere.paint_uniform_color([1, 0, 0])  # Red
        sphere.compute_vertex_normals()
        geometries.append(sphere)
    
    # Blue spheres for CAD picks
    for point in points_b:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(point)
        sphere.paint_uniform_color([0, 0, 1])  # Blue
        sphere.compute_vertex_normals()
        geometries.append(sphere)
    
    # Create lines connecting correspondences
    lines = [[i, i + len(points_a)] for i in range(len(points_a))]
    colors = [[0, 1, 0] for _ in range(len(lines))]  # Green lines
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack([points_a, points_b]))
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    geometries.append(line_set)
    
    # Small coordinate frame at origin (optional - comment out if not needed)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20.0)
    geometries.append(coord_frame)
    
    print("\nVisualization Legend:")
    print("  Gray surface: Point cloud/CAD geometry")
    print("  Red spheres: Point cloud picks")
    print("  Blue spheres: CAD model picks")
    print("  Green lines: Correspondences (error vectors)")
    print(f"  Sphere radius: {sphere_radius} mm")
    
    o3d.visualization.draw_geometries(geometries, 
                                       window_name="Point Correspondences",
                                       width=1200, height=900)

def main():
    """
    Main function - modify paths as needed
    """
    print("\n" + "=" * 80)
    print("POINT CORRESPONDENCE ERROR ANALYSIS")
    print("=" * 80 + "\n")
    
    # ============================================
    # MODIFY THESE PATHS
    # ============================================
    
    # Path to picked points from point cloud
    pcd_picked_points_file = "../../captures/captures_colorworld_stable/capture_20251121_162722/ground_truth_scene/cropped_camframe_with_faces_20251121_162722_picked_points.pp"
    
    # Option 1: Load CAD points from another .pp file
    cad_picked_points_file = "../../captures/captures_colorworld_stable/capture_20251121_162722/transformed_cad/cad_icp_refined_20251121_162722_picked_points.pp"
    
    # Optional: Load full point cloud and CAD mesh for visualization
    point_cloud_file = "../../captures/captures_colorworld_stable/capture_20251121_162722/ground_truth_scene/cropped_camframe_with_faces_20251121_162722.ply"
    cad_mesh_file = "../../captures/captures_colorworld_stable/capture_20251121_162722/transformed_cad/cad_icp_refined_20251121_162722.ply"
    
    # Unit conversion: Set to 1000 if coordinates are in meters and you want mm
    # Set to 1 if already in mm
    UNIT_SCALE = 1000.0  # Convert meters to millimeters
    UNIT_LABEL = "mm"     # Label for output
    
    # ============================================
    
    # Load point cloud picked points
    print(f"Loading point cloud picks from: {pcd_picked_points_file}")
    pcd_points, pcd_names = load_picked_points(pcd_picked_points_file)
    
    # Apply unit conversion
    pcd_points = pcd_points * UNIT_SCALE
    
    print(f"  Loaded {len(pcd_points)} points (converted to {UNIT_LABEL})")
    for i, name in enumerate(pcd_names):
        print(f"    {i}: {name} - [{pcd_points[i, 0]:.2f}, {pcd_points[i, 1]:.2f}, {pcd_points[i, 2]:.2f}]")
    
    # Load CAD picked points
    print(f"\nLoading CAD picks from: {cad_picked_points_file}")
    cad_points, cad_names = load_picked_points(cad_picked_points_file)
    
    # Apply unit conversion
    cad_points = cad_points * UNIT_SCALE
    
    print(f"  Loaded {len(cad_points)} points (converted to {UNIT_LABEL})")
    for i, name in enumerate(cad_names):
        print(f"    {i}: {name} - [{cad_points[i, 0]:.2f}, {cad_points[i, 1]:.2f}, {cad_points[i, 2]:.2f}]")
    
    # Verify same number of points
    if len(pcd_points) != len(cad_points):
        print(f"\nERROR: Number of points don't match!")
        print(f"  Point cloud: {len(pcd_points)} points")
        print(f"  CAD model: {len(cad_points)} points")
        return
    
    print(f"\nAnalyzing {len(pcd_points)} point correspondences...\n")
    
    # Calculate and print error analysis
    euclidean_dist, manhattan_dist, displacement = print_error_analysis(
        pcd_points, cad_points, 
        labels_a="Point Cloud", 
        labels_b="CAD Model",
        names_a=pcd_names,
        names_b=cad_names,
        unit_label=UNIT_LABEL
    )
    
    # Save results to file
    output_file = "correspondence_errors.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Point Correspondence Error Analysis\n")
        f.write("Point Cloud (Captured Data) as Reference Ground Truth\n")
        f.write("=" * 100 + "\n\n")
        
        # Detailed per-point breakdown
        f.write("DETAILED PER-POINT ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        
        for i in range(len(euclidean_dist)):
            name = pcd_names[i] if pcd_names else (cad_names[i] if cad_names else f"Point {i}")
            f.write(f"\nPoint {i}: {name}\n")
            f.write("-" * 100 + "\n")
            f.write(f"Point Cloud (Reference): [{pcd_points[i, 0]:.4f}, {pcd_points[i, 1]:.4f}, {pcd_points[i, 2]:.4f}] {UNIT_LABEL}\n")
            f.write(f"CAD Model:              [{cad_points[i, 0]:.4f}, {cad_points[i, 1]:.4f}, {cad_points[i, 2]:.4f}] {UNIT_LABEL}\n")
            f.write(f"Displacement (ΔX, ΔY, ΔZ): [{displacement[i, 0]:+.4f}, {displacement[i, 1]:+.4f}, {displacement[i, 2]:+.4f}] {UNIT_LABEL}\n")
            f.write(f"Euclidean Distance: {euclidean_dist[i]:.6f} {UNIT_LABEL}\n")
            f.write(f"Manhattan Distance: {manhattan_dist[i]:.6f} {UNIT_LABEL}\n")
        
        # Summary table
        f.write("\n" + "=" * 100 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Index':<6} | {'Point Name':<30} | {'ΔX':<12} | {'ΔY':<12} | {'ΔZ':<12} | {'Euclidean':<12} | {'Manhattan':<12}\n")
        f.write("-" * 100 + "\n")
        
        for i in range(len(euclidean_dist)):
            name = pcd_names[i] if pcd_names else (cad_names[i] if cad_names else f"Point {i}")
            f.write(f"{i:<6} | {name:<30} | {displacement[i, 0]:>11.4f} | {displacement[i, 1]:>11.4f} | {displacement[i, 2]:>11.4f} | {euclidean_dist[i]:>11.4f} | {manhattan_dist[i]:>11.4f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Euclidean Distance:\n")
        f.write(f"  Mean: {np.mean(euclidean_dist):.6f} {UNIT_LABEL}\n")
        f.write(f"  RMSE: {np.sqrt(np.mean(euclidean_dist**2)):.6f} {UNIT_LABEL}\n")
        f.write(f"  Std:  {np.std(euclidean_dist):.6f} {UNIT_LABEL}\n")
        f.write(f"  Min:  {np.min(euclidean_dist):.6f} {UNIT_LABEL}\n")
        f.write(f"  Max:  {np.max(euclidean_dist):.6f} {UNIT_LABEL}\n\n")
        
        f.write(f"Manhattan Distance:\n")
        f.write(f"  Mean: {np.mean(manhattan_dist):.6f} {UNIT_LABEL}\n")
        f.write(f"  Std:  {np.std(manhattan_dist):.6f} {UNIT_LABEL}\n")
        f.write(f"  Min:  {np.min(manhattan_dist):.6f} {UNIT_LABEL}\n")
        f.write(f"  Max:  {np.max(manhattan_dist):.6f} {UNIT_LABEL}\n\n")
        
        f.write(f"Displacement Statistics (CAD - Point Cloud):\n")
        f.write(f"  X-Axis: mean={np.mean(displacement[:, 0]):+.6f}, std={np.std(displacement[:, 0]):.6f} {UNIT_LABEL}\n")
        f.write(f"  Y-Axis: mean={np.mean(displacement[:, 1]):+.6f}, std={np.std(displacement[:, 1]):.6f} {UNIT_LABEL}\n")
        f.write(f"  Z-Axis: mean={np.mean(displacement[:, 2]):+.6f}, std={np.std(displacement[:, 2]):.6f} {UNIT_LABEL}\n\n")
        
        mean_disp = np.mean(displacement, axis=0)
        systematic_bias = np.linalg.norm(mean_disp)
        f.write(f"Systematic Bias: {systematic_bias:.6f} {UNIT_LABEL}\n")
        f.write(f"  Mean shift vector: [{mean_disp[0]:+.6f}, {mean_disp[1]:+.6f}, {mean_disp[2]:+.6f}] {UNIT_LABEL}\n")
    
    print(f"\nResults saved to: {output_file}")
    
    # Also save as CSV for easy analysis in Excel
    csv_file = "correspondence_errors.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Index,Point_Name,PC_X,PC_Y,PC_Z,CAD_X,CAD_Y,CAD_Z,Delta_X,Delta_Y,Delta_Z,Euclidean_Dist,Manhattan_Dist\n")
        for i in range(len(euclidean_dist)):
            name = pcd_names[i] if pcd_names else (cad_names[i] if cad_names else f"Point_{i}")
            name = name.replace(',', '_')  # Remove commas from names for CSV
            f.write(f"{i},{name},{pcd_points[i,0]:.6f},{pcd_points[i,1]:.6f},{pcd_points[i,2]:.6f},")
            f.write(f"{cad_points[i,0]:.6f},{cad_points[i,1]:.6f},{cad_points[i,2]:.6f},")
            f.write(f"{displacement[i,0]:.6f},{displacement[i,1]:.6f},{displacement[i,2]:.6f},")
            f.write(f"{euclidean_dist[i]:.6f},{manhattan_dist[i]:.6f}\n")
    
    print(f"CSV export saved to: {csv_file}")
    
    # Visualization
    visualize = input("\nVisualize correspondences? (y/n): ").lower().strip() == 'y'
    
    if visualize:
        # Load optional geometries
        pcd = None
        cad_mesh = None
        
        if point_cloud_file and Path(point_cloud_file).exists():
            print(f"\nLoading point cloud/mesh: {point_cloud_file}")
            pcd = load_geometry_file(point_cloud_file)
            if pcd is not None:
                # Scale the geometry too
                if isinstance(pcd, o3d.geometry.TriangleMesh):
                    pcd.vertices = o3d.utility.Vector3dVector(np.asarray(pcd.vertices) * UNIT_SCALE)
                else:
                    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * UNIT_SCALE)
        
        if cad_mesh_file and Path(cad_mesh_file).exists():
            print(f"\nLoading CAD mesh: {cad_mesh_file}")
            cad_mesh = load_geometry_file(cad_mesh_file)
            if cad_mesh is not None:
                # Scale the geometry too
                if isinstance(cad_mesh, o3d.geometry.TriangleMesh):
                    cad_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(cad_mesh.vertices) * UNIT_SCALE)
                else:
                    cad_mesh.points = o3d.utility.Vector3dVector(np.asarray(cad_mesh.points) * UNIT_SCALE)
        
        if pcd is None and cad_mesh is None:
            print("\nWARNING: Could not load any geometry files for visualization")
            print("Visualizing picked points only...")
        
        visualize_correspondences(pcd_points, cad_points, pcd, cad_mesh)

if __name__ == "__main__":
    main()