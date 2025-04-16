import pyrealsense2 as rs
import numpy as np
import cv2

# Configure the pipeline to stream depth and color data
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

try:
    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        raise RuntimeError("Could not acquire both depth and color frames.")
    
    # Convert the images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Save the color image as a PNG file
    cv2.imwrite("color_snapshot.png", color_image)
    print("Saved color snapshot as 'color_snapshot.png'")

    # Normalize the depth image to an 8-bit scale (0-255)
    depth_image_scaled = cv2.convertScaleAbs(depth_image, alpha=0.03)
    
    # Optionally apply a color map to better visualize the depth differences
    depth_colormap = cv2.applyColorMap(depth_image_scaled, cv2.COLORMAP_JET)
    cv2.imwrite("depth_snapshot_colormap.png", depth_colormap)
    print("Saved depth snapshot with colormap as 'depth_snapshot_colormap.png'")
    
    # Save the raw normalized depth image (if you prefer grayscale)
    cv2.imwrite("depth_snapshot_normalized.png", depth_image_scaled)
    print("Saved normalized depth snapshot as 'depth_snapshot_normalized.png'")
    
    # Create a point cloud object and map the color frame to it
    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    
    # Export the point cloud to a PLY file (including color information)
    points.export_to_ply("output.ply", color_frame)
    print("Saved point cloud to 'output.ply'")
    
finally:
    # Stop streaming
    pipeline.stop()
