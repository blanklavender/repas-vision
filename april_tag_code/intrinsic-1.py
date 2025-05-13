import pyrealsense2 as rs
import numpy as np

# Start pipeline and stream configuration
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start the stream
pipeline.start(config)

# Extract intrinsics from color stream
profile = pipeline.get_active_profile()
video_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = video_stream.get_intrinsics()

# Define variables for OpenCV usage
fx = intr.fx
fy = intr.fy
cx = intr.ppx
cy = intr.ppy
dist_coeffs = np.array(intr.coeffs, dtype=np.float32)

camera_matrix = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float32)

# Print to verify
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# Optional: stop if you only need intrinsics once
pipeline.stop()


