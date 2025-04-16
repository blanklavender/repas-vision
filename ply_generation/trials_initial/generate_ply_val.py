import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    # Wait for a coherent pair of frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        raise RuntimeError("Could not acquire depth or color frame")

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Get intrinsics
    profile = pipeline.get_active_profile()
    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    # Create point cloud
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)

    # Convert point cloud to numpy array
    vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # XYZ
    tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)  # UV

    # Convert UV to pixel indices
    u = (tex[:, 0] * color_image.shape[1]).astype(np.int32)
    v = (tex[:, 1] * color_image.shape[0]).astype(np.int32)
    u = np.clip(u, 0, color_image.shape[1] - 1)
    v = np.clip(v, 0, color_image.shape[0] - 1)

    # Extract RGB values
    colors = color_image[v, u] / 255.0  # Normalize to range [0,1]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vtx)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name="RealSense Point Cloud")

finally:
    pipeline.stop()
