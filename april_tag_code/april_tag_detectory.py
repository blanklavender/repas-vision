import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the camera
pipeline.start(config)

# Initialize the AprilTag detector
detector = Detector(families='tag36h11', nthreads=4, quad_decimate=1.0,
quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)

# Create an align object
align = rs.align(rs.stream.color)

# Get the intrinsic parameters of the aligned color camera
profile = pipeline.get_active_profile()
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
intrinsics = color_profile.get_intrinsics()

# Define init_pos correctly as a 1D array
init_pos = np.array([2.88055450e-01, 1.34160019e-18, 8.83400061e-02])

print("Waiting for 5 seconds to stabilize...")
time.sleep(5)
print("Starting frame processing...")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(gray_image)

        for tag in tags:
            cv2.polylines(
                color_image, [tag.corners.astype(int)], True, (0, 255, 0), 2)
            cv2.circle(color_image, tuple(int(i)
                    for i in tag.center), 5, (0, 0, 255), -1)

            camera_z = depth_frame.get_distance(
                int(tag.center[0]), int(tag.center[1]))

            if camera_z > 0.15 and camera_z < 1:
                arm_x = camera_z
                arm_y = -((camera_z / intrinsics.fx) *
                        (tag.center[0] - intrinsics.ppx))
                arm_z = -((camera_z / intrinsics.fy) *
                        (tag.center[1] - intrinsics.ppy))
                camera_position = np.array([arm_x, arm_y, arm_z])

                goal_x = init_pos[0] + arm_x - 0.025
                goal_y = init_pos[1] + arm_y + 0.04
                goal_z = init_pos[2] + arm_z - 0.005

                print(f'Pixel: {int(tag.center[0]), int(tag.center[1])}')
                print(f'Position: {goal_x, goal_y, goal_z}')

                with open("coordinates.txt", 'w') as file:
                    file.write(f"{goal_x}\n")
                    file.write(f"{goal_y}\n")
                    file.write(f"{goal_z}\n")

                cv2.putText(color_image, f"3D coordinates: X={goal_x:.4f}, Y={goal_y:.4f}, Z={goal_z:.4f}", (25, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(color_image, f'Pixel: {int(tag.center[0]), int(tag.center[1]), camera_z}', (
                    25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('AprilTag Detection with Depth', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()