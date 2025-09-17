import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the PLY file
def load_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)

# Plot the point cloud

def plot_point_cloud(points):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract X, Y, Z coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ax.scatter(x, y, z, s=1, c=z, cmap='viridis', marker='o')

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("3D Point Cloud Visualization")

    plt.show()

# Main function
if __name__ == "__main__":
    ply_file = "captured_pointcloud.ply" 
    points = load_ply(ply_file)

    if points.shape[0] > 0:
        plot_point_cloud(points)
    else:
        print("No points found in the PLY file.")