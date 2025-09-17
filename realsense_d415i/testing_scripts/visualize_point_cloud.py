# visualize_point_cloud.py

import open3d as o3d
import numpy as np

# 2. Load your point‐cloud
pcd = o3d.io.read_point_cloud("../femto_point_cloud_files/output/pointcloud/807693/DepthPoints_807760.ply")

# Quick info
print(f"Loaded point cloud with {len(pcd.points)} points")
points = np.asarray(pcd.points)
print("First 5 points:\n", points[:5])

# 3. (Optional) Add coordinate axes at the world origin
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=max(points.max(axis=0) - points.min(axis=0)) * 0.1,
    origin=[0, 0, 0]
)

# 4. Visualize
o3d.visualization.draw_geometries(
    [pcd, axes],
    window_name="3D Point Cloud",
    width=1024,
    height=768,
    left=50,
    top=50,
    point_show_normal=False
)
