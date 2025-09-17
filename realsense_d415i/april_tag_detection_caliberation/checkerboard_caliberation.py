import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import os

# === CONFIGURATION ===
CHECKERBOARD = (21, 20)  # 22x21 squares → 21x20 inner corners
square_size = 25.4  # 1 inch in mm
num_required_captures = 15

# === OBJECT POINTS SETUP ===
objp = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")

def calibrate_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    print("[INFO] Starting camera stream...")
    pipeline.start(config)
    os.makedirs("debug_frames", exist_ok=True)

    detected_last_frame = False  # Track detection state to avoid spam

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

            if ret and not detected_last_frame:
                print("[INFO] Checkerboard detected in frame.")
                detected_last_frame = True
            elif not ret:
                detected_last_frame = False
                failed_name = f"debug_frames/frame_failed_{len(objpoints)}.png"
                cv2.imwrite(failed_name, color_image)

            if ret:
                cv2.drawChessboardCorners(color_image, CHECKERBOARD, corners, ret)

            cv2.putText(color_image, f"Captures: {len(objpoints)}/{num_required_captures}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Calibration View", color_image)

            key = cv2.waitKey(1)

            if key == ord('c'):
                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    print(f"[INFO] Captured frame {len(objpoints)}/{num_required_captures}")
                else:
                    print("[WARN] Checkerboard not detected. Frame NOT captured.")

                if len(objpoints) >= num_required_captures:
                    print("[INFO] Sufficient frames captured. Starting calibration...")
                    break

            elif key == ord('q'):
                print("[INFO] Quit signal received. Ending session.")
                break

        cv2.destroyAllWindows()

        if len(objpoints) < 3:
            print("[ERROR] Not enough valid captures. Calibration aborted.")
            return

        # Calibration
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        print("\n=== Calibration Complete ===")
        print("Camera Matrix (K):\n", K)
        print("Distortion Coefficients:\n", dist.ravel())

        timestamp = get_timestamp()
        os.makedirs("calibration-results", exist_ok=True)
        np.savez(f"calibration-results/camera_intrinsics_{timestamp}.npz", K=K, dist=dist)

    finally:
        pipeline.stop()
        print("[INFO] Camera stream stopped.")

def main():
    calibrate_camera()

if __name__ == "__main__":
    main()
