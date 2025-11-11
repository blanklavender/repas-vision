#!/usr/bin/env python3
import open3d as o3d
import numpy as np

# ---------- CONFIG ----------
CAD_PATH = r"../../cad_model/Structure2.PLY"
SCALE_MM_TO_M = 1  # mm -> m
R_cad = np.array([
    [ 0.00268905, -0.99738914,  0.07216419],
    [ 0.99985527,  0.00389394,  0.01656102],
    [-0.01679878,  0.07210921,  0.99725526]
], dtype=float)

# ---------- LOAD ----------
mesh = o3d.io.read_triangle_mesh(CAD_PATH)
mesh.compute_vertex_normals()

# ---------- SCALE ABOUT CENTROID ----------
c_before = mesh.get_center().copy()                  # centroid in original units
mesh.scale(SCALE_MM_TO_M, center=c_before)           # convert mm->m about centroid
c_after = mesh.get_center().copy()                   # same numbers, now "meters"

# ---------- ROTATE ABOUT CENTROID ----------
mesh.rotate(R_cad, center=c_after)

# ---------- WHERE DID THE CAD ORIGIN GO? ----------
# Start with p0 = (0,0,0). Apply SAME transforms (scale about c, then rotate about c):
p0 = np.zeros(3)
p0_after_scale = c_after + SCALE_MM_TO_M * (p0 - c_after)         # scale about centroid
p0_after_rot   = c_after + R_cad @ (p0_after_scale - c_after)      # rotate about centroid
cad_origin_world = p0_after_rot  # final world position of the CAD's original (0,0,0)

# ---------- VIS HELPERS ----------
aabb = mesh.get_axis_aligned_bounding_box()
extent = aabb.get_max_bound() - aabb.get_min_bound()
diag = float(np.linalg.norm(extent)) if np.linalg.norm(extent) > 0 else 1.0

axis_world = max(0.05 * diag, 1e-3)
axis_small = max(0.04 * diag, 1e-3)
pt_radius  = max(0.02 * diag, 5e-4)

# World origin frame
frame_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_world)

# Centroid marker + frame
centroid_marker = o3d.geometry.TriangleMesh.create_sphere(radius=pt_radius)
centroid_marker.translate(c_after)
centroid_marker.paint_uniform_color([0.1, 0.3, 1.0])  # blue
frame_centroid = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_small)
frame_centroid.translate(c_after)

# CAD-origin-after-transform marker + frame
origin_marker = o3d.geometry.TriangleMesh.create_sphere(radius=pt_radius * 0.9)
origin_marker.translate(cad_origin_world)
origin_marker.paint_uniform_color([1.0, 0.4, 0.1])     # orange
frame_cadorig = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_small)
frame_cadorig.translate(cad_origin_world)

# Color the mesh
mesh.paint_uniform_color([0.8, 0.8, 0.8])

# ---------- DEBUG ----------
np.set_printoptions(precision=6, suppress=True)
print("=== DEBUG (meters) ===")
print("World origin:                      [0. 0. 0.]")
print("Centroid after scaling:            ", c_after)
print("CAD origin after scale+rotate:     ", cad_origin_world)
print("AABB extent (m):                   ", extent)

# ---------- SHOW ----------
o3d.visualization.draw_geometries(
    [mesh, frame_world, centroid_marker, frame_centroid, origin_marker, frame_cadorig],
    window_name="World origin vs Centroid vs CAD-origin (after transforms)",
    width=1200, height=900
)
