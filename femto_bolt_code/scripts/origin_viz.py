import open3d as o3d
import numpy as np

CAD_PATH = r"../../cad_model/Structure2.PLY"

mesh = o3d.io.read_triangle_mesh(CAD_PATH)
if not mesh.has_triangles():
    raise RuntimeError("Expected a triangle mesh.")
mesh.compute_vertex_normals()

# ---- Measurements for sizing helpers ----
aabb = mesh.get_axis_aligned_bounding_box()
extent = aabb.get_max_bound() - aabb.get_min_bound()
diag = float(np.linalg.norm(extent))
axis_size = max(0.05 * diag, 1e-3)   # 5% of model diagonal
pt_radius = max(0.01 * diag, 1e-3)   # small sphere size
line_width = 3                        # for visualization backends that support it

# ---- Origins / centers ----
origin = np.array([0.0, 0.0, 0.0])     # CAD/world origin
centroid = mesh.get_center()           # vertex-mean centroid (CAD units)
aabb_center = aabb.get_center()        # AABB center (CAD units)

# ---- Geometries to draw ----
# 1) World/CAD origin axes at (0,0,0)
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)

# 2) AABB (red) for scale/context
aabb.color = (1.0, 0.0, 0.0)

# 3) Centroid marker (blue sphere)
sphere_centroid = o3d.geometry.TriangleMesh.create_sphere(radius=pt_radius)
sphere_centroid.translate(centroid)
sphere_centroid.paint_uniform_color([0.1, 0.3, 1.0])

# 4) AABB center marker (green sphere)
sphere_aabbc = o3d.geometry.TriangleMesh.create_sphere(radius=pt_radius * 0.8)
sphere_aabbc.translate(aabb_center)
sphere_aabbc.paint_uniform_color([0.1, 0.8, 0.1])

# 5) Line from origin → centroid (purple)
line_pts = o3d.utility.Vector3dVector(np.vstack([origin, centroid]))
line_segs = o3d.utility.Vector2iVector([[0, 1]])
colors = o3d.utility.Vector3dVector([[0.6, 0.2, 0.8]])
line = o3d.geometry.LineSet(points=line_pts, lines=line_segs)
line.colors = colors

# ---- Debug print ----
np.set_printoptions(precision=3, suppress=True)
print("=== DEBUG ===")
print("CAD origin (0,0,0):            ", origin)
print("Vertex centroid (mesh.get_center()):", centroid)
print("AABB center:                       ", aabb_center)
print("Axis size:", round(axis_size, 3), "| Sphere radius:", round(pt_radius, 3))
print("Note: Units are whatever the CAD file uses (no scaling applied).")

# ---- Visualize ----
# Tip: Press 'R' to reset view; zoom/rotate to find the axes & markers.
o3d.visualization.draw_geometries(
    [mesh, frame, aabb, sphere_centroid, sphere_aabbc, line],
    window_name="CAD origin vs centroid (Open3D)",
    width=1100, height=800
)
