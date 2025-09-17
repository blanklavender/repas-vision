import cv2
import numpy as np
import pyrealsense2 as rs
from pupil_apriltags import Detector
import time
import os

# === Camera intrinsics & distortion ===
# Checkerboard calibration values
K = np.array([
    [605.2845686,   0.0,        309.95995203],
    [0.0,           605.44233933, 229.79166863],
    [0.0,             0.0,          1.0     ]
], dtype=np.float32)

dist_coeffs = np.array([
    0.04344582,  0.32076285, -0.00060687, -0.0004814, -1.40593456
], dtype=np.float32)

TAG_SIZE = 0.0303   # updated tag size in meters

# —– Start RealSense (color only)
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.color, 640,  480, rs.format.bgr8, 30)
pipeline.start(cfg)

# AprilTag detector
detector = Detector(
    families='tag36h11',
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=1.0,
    refine_edges=1,
    decode_sharpening=0.5,
    debug=0
)

print("Stabilizing camera auto-settings…")
time.sleep(5)
print("Running color-only AprilTag pose estimation…")

# ensure screenshots folder exists
os.makedirs("screenshots", exist_ok=True)

try:
    while True:
        frame = pipeline.wait_for_frames().get_color_frame()
        if not frame:
            continue

        img = np.asanyarray(frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        tags = detector.detect(gray)

        for tag in tags:
            corners = tag.corners.astype(np.float32)
            cv2.polylines(img, [corners.astype(int)], True, (0,255,0), 2)
            cx, cy = map(int, tag.center)
            cv2.circle(img, (cx,cy), 5, (0,0,255), -1)

            h = TAG_SIZE / 2.0
            obj_pts = np.array([
                [-h, -h, 0],
                [ h, -h, 0],
                [ h,  h, 0],
                [-h,  h, 0],
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_pts, corners, K, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                continue

            cv2.drawFrameAxes(
                img, K, dist_coeffs,
                rvec, tvec,
                TAG_SIZE * 0.5,
                2
            )

            R_mat, _ = cv2.Rodrigues(rvec)  # natural to obtain different looking R_mat values for the same rvec
            t = tvec.ravel()  # [X, Y, Z] in meters

            coord_text = f"X:{t[0]:.3f}m Y:{t[1]:.3f}m Z:{t[2]:.3f}m"
            cv2.putText(img, coord_text, (cx + 10, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(img, coord_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            print(f"Tag {tag.tag_id} → rvec={rvec.ravel()}, tvec={t}")
            print(f"R:\n{R_mat}\nT: {t}\n")

        cv2.imshow("AprilTag Pose (Color-Only)", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # quit on 'q'
            break
        elif key == ord('e'):
            # screenshot on 'e'
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshots/screenshot_{timestamp}.png"
            cv2.imwrite(filename, img)
            print(f"[+] Screenshot saved to {filename}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
