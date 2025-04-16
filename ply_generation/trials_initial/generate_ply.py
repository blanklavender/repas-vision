import pyrealsense2 as rs

# Configure pipeline

pipe = rs.pipeline()

config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipe.start(config)



# Create pointcloud object

pc = rs.pointcloud()

align = rs.align(rs.stream.color)

rs.log_to_console(rs.log_severity.debug)

# Capture frames

while True:

    frames = pipe.wait_for_frames()

    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()

    color_frame = aligned_frames.get_color_frame()



    # Calculate point cloud

    points = pc.calculate(depth_frame)



    # Save to PLY file

    ply = rs.save_to_ply("output.ply")

    ply.set_option(rs.save_to_ply.option_ply_binary, True)  # Set to binary format

    ply.process(color_frame)  # Use color data for texture



    # ... (Optional: add loop control or other processing)



pipe.stop()
