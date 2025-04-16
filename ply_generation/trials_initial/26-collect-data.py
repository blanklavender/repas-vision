import cv2

import numpy as np

import pyrealsense2 as rs

import open3d as o3d  # Import Open3D

 

# Initialize RealSense pipeline and configuration

pipeline = rs.pipeline()

config = rs.config()

config.enable_stream(rs.stream.depth, rs.format.z16, 30)

config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

 

# Start streaming

pipeline.start(config)

 

# Wait for a coherent pair of frames: depth and color

frames = pipeline.wait_for_frames()

depth_frame = frames.get_depth_frame()

color_frame = frames.get_color_frame()

 

# Create a point cloud

pc = rs.pointcloud()

pc.map_to(color_frame)

 

# Generate the point cloud with color

points = pc.calculate(depth_frame)

v, t = points.get_vertices(), points.get_texture_coordinates()

verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)

colors = np.asanyarray(color_frame.get_data()).reshape(-1, 3)  # RGB

 

# Create an Open3D point cloud

pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(verts)

pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

 

# Create an Open3D visualizer and add the point cloud

visualizer = o3d.visualization.Visualizer()

visualizer.create_window()

visualizer.add_geometry(pcd)

 

# Set the view and visualize the point cloud

visualizer.get_render_option().point_size = 1.0  # Adjust point size if needed

visualizer.run()

 

# Stop streaming

pipeline.stop()