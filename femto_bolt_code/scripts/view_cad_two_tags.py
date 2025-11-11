#!/usr/bin/env python3
"""
Visualize CAD model with origin axes and specific marker points
"""
from pathlib import Path
import numpy as np
import open3d as o3d

# =========================================================
#                     CONFIG
# =========================================================
CAD_PLY = Path(r"../../cad_model/StructureTotal-v2.PLY")
CAD_UNITS_TO_METERS =  1.0 # Set to 1.0 if you want to keep CAD units
# Points to mark (in CAD units - will be converted automatically)
# Points to mark (in CAD units - will be converted automatically)
MARKER_POINTS = {
    "Top Left": np.array([-2.8601, -633.1330, 639.400]),
    "Top Right": np.array([-2.8601, -629.633, 639.400]),
    "Bottom Left": np.array([-2.8601, -633.1330, 635.9]),
    "Bottom Right": np.array([-2.8601, -629.633, 635.9]),
    "Center": np.array([-2.8601, -631.383, 637.65])
}

# *** NEW: Apply rotation transformation ***
APPLY_ROTATION = True  # Set to False to disable
ROTATION_AXIS = 'Y'    # 'X', 'Y', or 'Z'
ROTATION_DEGREES = -90  # Rotation angle in degrees

def create_rotation_matrix(axis: str, degrees: float) -> np.ndarray:
    """
    Create a rotation matrix for rotation around X, Y, or Z axis.
    
    Args:
        axis: 'X', 'Y', or 'Z'
        degrees: Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        3x3 rotation matrix
    """
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    
    if axis.upper() == 'X':
        return np.array([
            [1,  0,  0],
            [0,  c, -s],
            [0,  s,  c]
        ])
    elif axis.upper() == 'Y':
        return np.array([
            [ c,  0,  s],
            [ 0,  1,  0],
            [-s,  0,  c]
        ])
    elif axis.upper() == 'Z':
        return np.array([
            [c, -s,  0],
            [s,  c,  0],
            [0,  0,  1]
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'X', 'Y', or 'Z'")

# Apply rotation to all marker points if enabled
if APPLY_ROTATION:
    R = create_rotation_matrix(ROTATION_AXIS, ROTATION_DEGREES)
    print(f"\nApplying {ROTATION_DEGREES}° rotation around {ROTATION_AXIS}-axis")
    print(f"Rotation matrix:\n{R}\n")
    
    MARKER_POINTS_ROTATED = {}
    for label, point in MARKER_POINTS.items():
        rotated_point = R @ point  # Matrix multiplication
        MARKER_POINTS_ROTATED[label] = rotated_point
        print(f"{label:12s}: {point} → {rotated_point}")
    
    # Replace original points with rotated ones
    MARKER_POINTS = MARKER_POINTS_ROTATED

# Visualization settings
AXES_SIZE = 100.0  # Size of coordinate axes (in CAD units before conversion)
SPHERE_RADIUS = 5.0  # Radius of marker spheres (in CAD units before conversion)
DRAW_CONNECTING_LINES = True  # Draw lines between corners to show the rectangle

# =========================================================
#                     HELPER FUNCTIONS
# =========================================================
def load_cad_geometry(path: Path):
    """Load CAD model (mesh or point cloud)"""
    if not path.exists():
        raise FileNotFoundError(f"CAD file not found: {path}")
    
    # Try loading as mesh first
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh and len(np.asarray(mesh.vertices)) > 0:
        mesh.compute_vertex_normals()
        return mesh
    
    # Try as point cloud
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd and len(np.asarray(pcd.points)) > 0:
        return pcd
    
    raise RuntimeError(f"Failed to load CAD from {path}")

def create_coordinate_axes(size: float = 1.0, origin: np.ndarray = None):
    """
    Create RGB coordinate axes (X=red, Y=green, Z=blue)
    
    Args:
        size: Length of each axis
        origin: Origin position (default: [0, 0, 0])
    """
    if origin is None:
        origin = np.array([0.0, 0.0, 0.0])
    
    radius = max(size * 0.01, 0.001)  # 1% of axis length
    
    # Create cylinders for each axis
    x_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size)
    y_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size)
    z_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=size)
    
    # Color: X=red, Y=green, Z=blue
    x_cyl.paint_uniform_color([1.0, 0.0, 0.0])
    y_cyl.paint_uniform_color([0.0, 1.0, 0.0])
    z_cyl.paint_uniform_color([0.0, 0.0, 1.0])
    
    # Rotate and position cylinders
    # X-axis: rotate around Y by -90°
    x_cyl.rotate(
        x_cyl.get_rotation_matrix_from_xyz((0.0, -np.pi/2.0, 0.0)),
        center=(0, 0, 0)
    )
    x_cyl.translate((size/2.0, 0.0, 0.0))
    
    # Y-axis: rotate around X by 90°
    y_cyl.rotate(
        y_cyl.get_rotation_matrix_from_xyz((np.pi/2.0, 0.0, 0.0)),
        center=(0, 0, 0)
    )
    y_cyl.translate((0.0, size/2.0, 0.0))
    
    # Z-axis: no rotation needed (already points up)
    z_cyl.translate((0.0, 0.0, size/2.0))
    
    # Create cones for arrow tips
    cone_radius = radius * 2.5
    cone_height = size * 0.15
    
    x_cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
    y_cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
    z_cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)
    
    x_cone.paint_uniform_color([1.0, 0.0, 0.0])
    y_cone.paint_uniform_color([0.0, 1.0, 0.0])
    z_cone.paint_uniform_color([0.0, 0.0, 1.0])
    
    # Position cones at ends of axes
    x_cone.rotate(
        x_cone.get_rotation_matrix_from_xyz((0.0, 0.0, -np.pi/2.0)),
        center=(0, 0, 0)
    )
    x_cone.translate((size, 0.0, 0.0))
    
    y_cone.rotate(
        y_cone.get_rotation_matrix_from_xyz((0.0, 0.0, 0.0)),
        center=(0, 0, 0)
    )
    y_cone.translate((0.0, size, 0.0))
    
    z_cone.rotate(
        z_cone.get_rotation_matrix_from_xyz((np.pi, 0.0, 0.0)),
        center=(0, 0, 0)
    )
    z_cone.translate((0.0, 0.0, size))
    
    # Combine all axis components
    axes = x_cyl + y_cyl + z_cyl + x_cone + y_cone + z_cone
    axes.compute_vertex_normals()
    
    # Translate to origin if specified
    if not np.allclose(origin, 0.0):
        axes.translate(origin)
    
    return axes

def create_marker_sphere(position: np.ndarray, radius: float, color: list):
    """Create a colored sphere at the given position"""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=20)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    sphere.translate(position)
    return sphere

def create_connecting_lines(points: dict, line_color: list = None):
    """
    Create lines connecting the corner points in rectangle order
    
    Args:
        points: Dictionary with keys containing "Top Left", "Top Right", etc.
        line_color: RGB color for lines (default: yellow)
    """
    if line_color is None:
        line_color = [1.0, 1.0, 0.0]  # Yellow
    
    # Extract corner points in order
    try:
        tl = points["Top Left"]
        tr = points["Top Right"]
        bl = points["Bottom Left"]
        br = points["Bottom Right"]
    except KeyError as e:
        print(f"Warning: Missing point for connecting lines: {e}")
        return None
    
    # Create line segments: TL->TR, TR->BR, BR->BL, BL->TL
    line_points = [tl, tr, br, bl, tl]  # Close the loop
    
    lines = []
    for i in range(len(line_points) - 1):
        lines.append([i, i + 1])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([line_color for _ in lines])
    
    # Also add diagonals
    diagonal_points = line_points[:-1]  # Remove duplicate last point
    diagonal_lines = [[0, 2], [1, 3]]  # TL->BR, TR->BL
    
    diagonal_set = o3d.geometry.LineSet()
    diagonal_set.points = o3d.utility.Vector3dVector(diagonal_points)
    diagonal_set.lines = o3d.utility.Vector2iVector(diagonal_lines)
    diagonal_set.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.0] for _ in diagonal_lines])
    
    return line_set, diagonal_set

# =========================================================
#                        MAIN
# =========================================================
def main():
    print("=" * 60)
    print("CAD Model Visualization with Markers")
    print("=" * 60)
    
    if APPLY_ROTATION:
        print(f"\n[NOTE] Marker points rotated {ROTATION_DEGREES}° around {ROTATION_AXIS}-axis")
    

    # Load CAD model
    print(f"\n[1/4] Loading CAD model from: {CAD_PLY}")
    cad = load_cad_geometry(CAD_PLY)
    
    # Get original bounding box info
    if isinstance(cad, o3d.geometry.TriangleMesh):
        vertices = np.asarray(cad.vertices)
        print(f"      Loaded mesh with {len(vertices)} vertices")
    else:
        vertices = np.asarray(cad.points)
        print(f"      Loaded point cloud with {len(vertices)} points")
    
    bbox = cad.get_axis_aligned_bounding_box()
    bbox_center = bbox.get_center()
    bbox_extent = bbox.get_extent()
    
    print(f"      Bounding box center: [{bbox_center[0]:.2f}, {bbox_center[1]:.2f}, {bbox_center[2]:.2f}]")
    print(f"      Bounding box extent: [{bbox_extent[0]:.2f}, {bbox_extent[1]:.2f}, {bbox_extent[2]:.2f}]")
    
    # Apply unit conversion
    print(f"\n[2/4] Applying unit conversion: {CAD_UNITS_TO_METERS}")
    scale_factor = float(CAD_UNITS_TO_METERS)
    cad.scale(scale_factor, center=cad.get_center())
    
    # Create visualization geometries
    print(f"\n[3/4] Creating visualization markers...")
    geoms = []
    
    # Add CAD model
    geoms.append(cad)
    
    # Create coordinate axes at origin
    axes_size_scaled = AXES_SIZE * scale_factor
    print(f"      - Coordinate axes at origin (size: {axes_size_scaled:.3f}m)")
    axes = create_coordinate_axes(size=axes_size_scaled, origin=np.array([0.0, 0.0, 0.0]))
    geoms.append(axes)
    
    # Create marker spheres for each point
    sphere_radius_scaled = SPHERE_RADIUS * scale_factor
    converted_points = {}
    
    # Color scheme for markers
    colors = {
        "Top Left": [1.0, 0.0, 0.0],      # Red
        "Top Right": [0.0, 1.0, 0.0],     # Green
        "Bottom Left": [0.0, 0.0, 1.0],   # Blue
        "Bottom Right": [1.0, 0.0, 1.0],  # Magenta
        "Center": [1.0, 1.0, 0.0]         # Yellow
    }
    
    for label, point_cad_units in MARKER_POINTS.items():
        # Convert to meters (or keep in CAD units if scale_factor = 1.0)
        point_converted = point_cad_units * scale_factor
        converted_points[label] = point_converted
        
        color = colors.get(label, [0.5, 0.5, 0.5])  # Default gray
        sphere = create_marker_sphere(point_converted, sphere_radius_scaled, color)
        geoms.append(sphere)
        
        print(f"      - {label:12s}: [{point_converted[0]:8.4f}, {point_converted[1]:8.4f}, {point_converted[2]:8.4f}]")
    
    # Add connecting lines between corners
    if DRAW_CONNECTING_LINES:
        print(f"      - Drawing connecting lines between corners")
        result = create_connecting_lines(converted_points)
        if result:
            edge_lines, diagonal_lines = result
            geoms.append(edge_lines)
            geoms.append(diagonal_lines)
    
    # Display
    print(f"\n[4/4] Launching visualization...")
    print("\nViewer controls:")
    print("  - Left mouse: Rotate")
    print("  - Right mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - Press 'H' in viewer for more help")
    print("=" * 60)
    
    o3d.visualization.draw_geometries(
        geoms,
        window_name="CAD Model with Coordinate Axes and Markers",
        width=1280,
        height=720
    )

if __name__ == "__main__":
    main()