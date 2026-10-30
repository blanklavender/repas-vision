import argparse

import pyrealsense2 as rs
import numpy as np
import cv2

# Color resolutions supported by the D415 RGB sensor that we use in this repo
RESOLUTIONS = {
    "640x480": (640, 480),
    "1280x720": (1280, 720),
}

parser = argparse.ArgumentParser(description="Live RealSense color stream preview")
parser.add_argument("--res", choices=RESOLUTIONS.keys(), default="640x480",
                    help="color stream resolution (default: 640x480)")
parser.add_argument("--fps", type=int, default=30, choices=[6, 15, 30],
                    help="frame rate (default: 30; use 15 or 6 on USB 2 ports)")
args = parser.parse_args()

width, height = RESOLUTIONS[args.res]

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, args.fps)

if not config.can_resolve(rs.pipeline_wrapper(pipeline)):
    raise SystemExit(
        f"Camera cannot stream color {width}x{height} @ {args.fps}fps. "
        "Check the USB 3 connection, or try a lower --fps. "
        "Run supported_stream_list.py to see what the camera offers."
    )

# Start pipeline
profile = pipeline.start(config)
usb = profile.get_device().get_info(rs.camera_info.usb_type_descriptor)
print(f"Streaming color {width}x{height} @ {args.fps}fps over USB {usb}. Press 'q' to quit.")

window = f"{width}x{height} Color Stream"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)  # resizable, so 1280x720 fits small screens

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow(window, color_image)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
