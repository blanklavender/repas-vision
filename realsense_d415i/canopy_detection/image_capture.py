import pyrealsense2 as rs
import numpy as np
import cv2
import datetime


## Depth Image is not properly color mapped, so the snapshot is bad
## Output: RGB PNG and Bag File (storing depth information)

def get_timestamp():
    """Return the current timestamp in YYYY-MM-DDTHHMMSS format."""
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")

def capture_snapshot():
    # Create a RealSense pipeline and configuration.
    pipeline = rs.pipeline()
    config = rs.config()

    # Set the highest available resolutions:
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Generate a timestamp for unique file names.
    timestamp = get_timestamp()

    # Set up recording to a bag file with the timestamp in its name.
    bag_filename = f"new-captures/depth_capture_{timestamp}.bag"
    config.enable_record_to_file(bag_filename)

    # Start streaming.
    pipeline.start(config)

    try:
        # Wait for a coherent pair of frames: depth and color.
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            raise RuntimeError("Could not acquire both depth and color frames.")

        # Convert frames to numpy arrays.
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Create filenames with the timestamp.
        color_filename = f"new-captures/plane_image_capture_{timestamp}.png"
        depth_filename = f"new-captures/depth_snapshot_{timestamp}.png"

        # Save the color image (in its native BGR format) as a PNG file.
        cv2.imwrite(color_filename, color_image)
        print(f"Saved color snapshot as '{color_filename}'")

        # Save the raw depth image as a PNG file.
        # The depth image is a 16-bit single-channel image (z16 format).
        cv2.imwrite(depth_filename, depth_image)
        print(f"Saved raw depth snapshot as '{depth_filename}'")
        print(f"Recording bag file to '{bag_filename}'")

    finally:
        # Stop streaming (finalizes the bag file).
        pipeline.stop()

def main():
    capture_snapshot()

if __name__ == "__main__":
    main()
