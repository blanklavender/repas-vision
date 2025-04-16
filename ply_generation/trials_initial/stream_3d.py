import pyrealsense2 as rs

import numpy as np

import open3d as o3d

 

def capture_ply(output_file="output.ply"):

    # Configure depth and color streams

    pipeline = rs.pipeline()

    config = rs.config()

   

    # Enable depth stream

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

 

    # Start the pipeline

    pipeline.start(config)

   

    try:

        # Wait for frames (allow camera to stabilize)

        for _ in range(30): 

            frames = pipeline.wait_for_frames()

 

        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()

        color_frame = frames.get_color_frame()

       

        if not depth_frame or not color_frame:

            print("No frames received. Check camera connection.")

            return

 

        # Create point cloud object

        pc = rs.pointcloud()

        points = pc.calculate(depth_frame)  # Compute 3D points

        pc.map_to(color_frame)  # Map color to the point cloud

 

        # Export the point cloud as a PLY file

        points.export_to_ply(output_file, color_frame)

        print(f"Point cloud saved to {output_file}")

 

    finally:

        # Stop the pipeline

        pipeline.stop()

 

# Run the function to capture and save the point cloud

capture_ply("captured_pointcloud.ply")