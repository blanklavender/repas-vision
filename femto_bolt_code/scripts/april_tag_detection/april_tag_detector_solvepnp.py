#!/usr/bin/env python3
"""
Femto Bolt (pyorbbecsdk) — Color-only AprilTag Z-distance via PnP

What it does:
- Opens COLOR stream only.
- Detects AprilTag corners in the COLOR image.
- Runs cv2.solvePnP (IPPE square if available) using your calibrated intrinsics.
- Displays and logs:
    Z (optical-axis distance)  ~ tvec[2]  [meters]
    Euclidean distance        ~ ||tvec||  [meters]

Keys:
  q / ESC : quit
  e       : save a screenshot to ./screenshots

Requirements:
  pip install pupil-apriltags opencv-python
"""

import os
import json
import time
import cv2
import numpy as np

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, VideoStreamProfile, OBError, FrameSet
)

try:
    from utils import frame_to_bgr_image as ob_frame_to_bgr  
except Exception:
    ob_frame_to_bgr = None

# =========================
# USER CONFIG
# =========================
CALIB_JSON_PATH = r"./calibration_parameters/checkerboard_color_intrinsics_2025-08-26T183535.json" 
TAG_SIZE_M      = 0.0303             # adjusted tag size based on validation checks from viz tool

ESC_KEY = 27

# =========================
# Helpers
# =========================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def load_intrinsics_json(path: str):
    """Expect keys: fx, fy, cx, cy, width, height, dist_coeffs (list of 0..8)."""
    with open(path, "r") as f:
        J = json.load(f)
    fx, fy = float(J["fx"]), float(J["fy"])
    cx, cy = float(J["cx"]), float(J["cy"])
    w, h  = int(J["width"]), int(J["height"])
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    dist = np.array(J.get("dist_coeffs", [0,0,0,0,0]), dtype=np.float32)
    if dist.size < 5:
        dist = np.pad(dist, (0, 5 - dist.size)).astype(np.float32)
    elif dist.size > 5:
        dist = dist[:5].astype(np.float32)
    return K, dist, w, h

def scale_K(K: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    K2 = K.copy()
    K2[0,0] *= sx  # fx
    K2[1,1] *= sy  # fy
    K2[0,2] *= sx  # cx
    K2[1,2] *= sy  # cy
    return K2

def pick_color_profile(pipeline: Pipeline, w: int, h: int, fps: int) -> VideoStreamProfile:
    """Try to request BGR at (w,h,fps); fall back to default."""
    plist = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        return plist.get_video_stream_profile(w, h, OBFormat.BGR, fps)
    except OBError:
        pass
    try:
        return plist.get_video_stream_profile(w, h, OBFormat.RGB, fps)
    except OBError:
        pass
    try:
        return plist.get_video_stream_profile(w, h, OBFormat.MJPG, fps)
    except OBError:
        return plist.get_default_video_stream_profile()

def frame_to_bgr(color_frame):
    """Convert Orbbec color frame to BGR (simple paths)."""
    if color_frame is None:
        return None
    if ob_frame_to_bgr is not None:
        try:
            return ob_frame_to_bgr(color_frame)
        except Exception:
            pass

    w = color_frame.get_width()
    h = color_frame.get_height()
    fmt = color_frame.get_format()
    buf = color_frame.get_data()
    data = np.frombuffer(buf, dtype=np.uint8)

    try:
        if fmt == OBFormat.BGR:
            return data.reshape((h, w, 3))
        elif fmt == OBFormat.RGB:
            rgb = data.reshape((h, w, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif fmt in (OBFormat.MJPG, getattr(OBFormat, "MJPEG", OBFormat.MJPG)):
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        else:
            # Minimal: unsupported format here (e.g., NV12) → bail
            return None
    except Exception:
        return None

def tag_object_corners(size_m: float) -> np.ndarray:
    """TL, TR, BR, BL order (matches pupil_apriltags corners)."""
    s = size_m / 2.0
    return np.array([
        [-s,  s, 0.0],  # TL
        [ s,  s, 0.0],  # TR
        [ s, -s, 0.0],  # BR
        [-s, -s, 0.0],  # BL
    ], dtype=np.float32)

# =========================
# Main
# =========================
def main():
    # Load intrinsics
    K_cal, dist_cal, req_w, req_h = load_intrinsics_json(CALIB_JSON_PATH)
    req_fps = 30

    # Start Orbbec COLOR-only pipeline
    pipe = Pipeline()
    cfg  = Config()
    try:
        color_prof = pick_color_profile(pipe, req_w, req_h, req_fps)
        cfg.enable_stream(color_prof)
        pipe.start(cfg)
    except Exception as e:
        print("Failed to start color stream:", e)
        return

    # AprilTag detector
    from pupil_apriltags import Detector
    detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    obj_pts = tag_object_corners(TAG_SIZE_M)
    K_active = K_cal.copy()
    dist_active = dist_cal.copy()
    have_dims = False

    screenshots_dir = ensure_dir("./screenshots")
    print("Running… press 'q' or ESC to quit, 'e' to screenshot.")

    try:
        while True:
            frames: FrameSet = pipe.wait_for_frames(100)
            if frames is None:
                continue
            cframe = frames.get_color_frame()
            if cframe is None:
                continue

            w_act, h_act = cframe.get_width(), cframe.get_height()
            if not have_dims:
                print(f"Active color stream: {w_act}x{h_act} (requested {req_w}x{req_h}@{req_fps})")
                if (w_act, h_act) != (req_w, req_h):
                    K_active = scale_K(K_cal, req_w, req_h, w_act, h_act)
                    print("Scaled intrinsics K to match active frame size.")
                have_dims = True

            img = frame_to_bgr(cframe)
            if img is None:
                print("Unsupported color format for this minimal example.")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray, estimate_tag_pose=False)

            for t in tags:
                corners = t.corners.astype(np.float32)  # TL, TR, BR, BL
                # Solve PnP (prefer IPPE for square planar targets)
                flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
                ok, rvec, tvec = cv2.solvePnP(obj_pts, corners, K_active, dist_active, flags=flag)
                if not ok:
                    continue

                # Distances
                z_m = float(tvec[2,0])             # optical-axis distance (meters)
                r_m = float(np.linalg.norm(tvec))  # true Euclidean distance (meters)

                # Draw
                cx, cy = map(int, t.center)
                cv2.polylines(img, [corners.astype(int)], True, (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
                cv2.drawFrameAxes(img, K_active, dist_active, rvec, tvec, TAG_SIZE_M*0.5, 2)

                cv2.putText(img, f"Z ~ {z_m:.3f} m", (cx+8, cy-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, f"||t|| ~ {r_m:.3f} m", (cx+8, cy+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)

            cv2.imshow("AprilTag Z via PnP (Color-only)", img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ESC_KEY):
                break
            if k == ord('e'):
                fn = os.path.join(screenshots_dir, time.strftime("tag_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(fn, img)
                print("Saved:", fn)
    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Stopped.")

if __name__ == "__main__":
    main()
