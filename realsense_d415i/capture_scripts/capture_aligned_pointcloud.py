import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# Enable depth stream (stereo module) at 640x480 @ 30 FPS
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# Enable color stream at 640x480 @ 30 FPS
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Create an align object so that depth is aligned to color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Create a pointcloud object to generate the point cloud data
pc = rs.pointcloud()

try:
    print("Streaming started. Press 's' to save the PLY file, or ESC to exit.")
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Generate the point cloud from the depth frame and map it to the color frame
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)
        
        # Visualize the 2D color and depth images
        cv2.imshow('Color Image', color_image)
        # Convert depth image to a viewable format using a colormap
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth Image', depth_colormap)
        
        key = cv2.waitKey(1)
        # Press 's' key to save the point cloud to a PLY file
        if key & 0xFF == ord('s'):
            ply_filename = "output.ply"
            points.export_to_ply(ply_filename, color_frame)
            print(f"Saved point cloud to {ply_filename}")
        # Press ESC key to exit
        elif key & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
