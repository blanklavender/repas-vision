import open3d as o3d
import numpy as np

# 1. Load PLY file using Open3D
input_file = "./3D_Final_Ply_Files/decent.ply"
pcd = o3d.io.read_point_cloud(input_file)

# 2. Convert to NumPy array to process distances
points = np.asarray(pcd.points)

# 3. Compute distances from the origin (0,0,0)
distances = np.linalg.norm(points, axis=1)

# 4. Create a mask to keep only points within 1.0 meter
max_distance = 1.0
mask = distances < max_distance

# 5. Filter out background (far) points
filtered_points = points[mask]

# 6. Create a new point cloud with the filtered points
pcd_filtered = o3d.geometry.PointCloud()
pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)

# 7. (Optional) If you have color or normals in your PLY, filter those similarly
#    E.g., if you also have colors:
#    colors = np.asarray(pcd.colors)
#    pcd_filtered.colors = o3d.utility.Vector3dVector(colors[mask])

# 8. Save the filtered point cloud to a new PLY file
output_file = "filtered_pointcloud.ply"
o3d.io.write_point_cloud(output_file, pcd_filtered)

print(f"Filtered PLY saved as: {output_file}")
