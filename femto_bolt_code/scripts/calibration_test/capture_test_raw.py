"""
Femto Bolt raw capture: RGB + uncalibrated depth (NO D2C alignment).

- Color saved as PNG (BGR) at native color resolution.
- Depth saved as 16-bit PNG (raw uint16, mm values) at native depth resolution.
- Streams run independently; no AlignFilter, no spatial registration.

Keys: SPACE = capture frame pair, q = quit.
"""
import os
import cv2
import numpy as np
from datetime import datetime
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

# ---- config ----
OUT_DIR = "raw_captures"
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
# Femto Bolt depth modes: 320x288, 640x576 (NFOV unbinned),
# 512x512 (WFOV 2x2 binned), 1024x1024 (WFOV unbinned).
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 576, 30

def frame_to_bgr_image(color_frame):
    """Decode a pyorbbecsdk color frame to a BGR numpy image."""
    w = color_frame.get_width()
    h = color_frame.get_height()
    fmt = color_frame.get_format()
    data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    if fmt == OBFormat.NV21:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV21)
    if fmt == OBFormat.NV12:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    if fmt == OBFormat.I420:
        return cv2.cvtColor(data.reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_I420)
    return None

def make_session_dir():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUT_DIR, f"session_{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def build_config(pipeline):
    cfg = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_video_stream_profile(
        COLOR_W, COLOR_H, OBFormat.MJPG, COLOR_FPS
    )
    cfg.enable_stream(color_profile)

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_video_stream_profile(
        DEPTH_W, DEPTH_H, OBFormat.Y16, DEPTH_FPS
    )
    cfg.enable_stream(depth_profile)

    # NOTE: no cfg.set_align_mode(...) and no AlignFilter is constructed.
    return cfg


def main():
    session_dir = make_session_dir()
    print(f"Saving to: {session_dir}")
    print("SPACE = capture | q = quit")

    pipeline = Pipeline()
    cfg = build_config(pipeline)
    pipeline.start(cfg)

    frame_idx = 0
    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            # color (MJPG -> BGR)
            color_bgr = frame_to_bgr_image(color_frame)
            if color_bgr is None:
                continue

            # depth (raw uint16, native depth-sensor frame, untouched)
            d_w = depth_frame.get_width()
            d_h = depth_frame.get_height()
            depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(d_h, d_w)

            # preview only (not saved)
            depth_vis = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            preview = np.hstack([
                cv2.resize(color_bgr, (640, 360)),
                cv2.resize(depth_vis, (640, 360)),
            ])
            cv2.imshow("Femto Bolt  |  color (left)  |  depth (right, unaligned)", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                color_path = os.path.join(session_dir, f"color_f{frame_idx:03d}.png")
                depth_path = os.path.join(session_dir, f"depth_f{frame_idx:03d}.png")
                cv2.imwrite(color_path, color_bgr)
                cv2.imwrite(depth_path, depth_raw)  # 16-bit PNG preserves mm losslessly
                print(f"[{frame_idx:03d}] color {color_bgr.shape[1]}x{color_bgr.shape[0]}  "
                      f"depth {d_w}x{d_h}")
                frame_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()