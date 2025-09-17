#!/usr/bin/env python3
"""
Femto Bolt — AprilTag Z (optical-axis) via ToF depth lookup @ 1280x720

- Requests COLOR 1280x720 and DEPTH 1280x720 (falls back cleanly if unsupported)
- Aligns DEPTH -> COLOR
- Detects AprilTag in COLOR (pupil_apriltags)
- Z = median depth (meters) in a small window around tag center
- Shows helpful debug info to diagnose "Z invalid" issues

Keys:
  q / ESC : quit
  e       : save screenshot to ./screenshots
"""

import os
import time
import cv2
import numpy as np
from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, VideoStreamProfile, OBError,
    AlignFilter, OBStreamType, FrameSet
)
from pupil_apriltags import Detector

ESC_KEY = 27

# -------------------------
# User-tunable params
# -------------------------
REQ_W, REQ_H, REQ_FPS = 1280, 720, 30
WINDOW        = 5        # 3, 5, 7... median window
MIN_DEPTH_M   = 0.25     # meters (WFOV ~0.25m; use 0.50 if NFOV)
MAX_DEPTH_M   = 8.00     # meters
FAMILY        = "tag36h11"
PRINT_ONCE    = True     # print one-time debug info

# If you have Orbbec's utils helper
try:
    from utils import frame_to_bgr_image as ob_frame_to_bgr
except Exception:
    ob_frame_to_bgr = None


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def pick_color_profile(pipeline: Pipeline, w: int, h: int, fps: int) -> VideoStreamProfile:
    plist = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    for fmt in (getattr(OBFormat, "BGR", None), getattr(OBFormat, "RGB", None),
                getattr(OBFormat, "NV12", None), getattr(OBFormat, "MJPG", None)):
        if fmt is None:
            continue
        try:
            return plist.get_video_stream_profile(w, h, fmt, fps)
        except OBError:
            pass
    return plist.get_default_video_stream_profile()


def pick_depth_profile(pipeline: Pipeline, w: int, h: int, fps: int) -> VideoStreamProfile:
    plist = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    # Try 1280x720 Y16 first
    try:
        return plist.get_video_stream_profile(w, h, OBFormat.Y16, fps)
    except OBError:
        pass
    # Common fallback for ToF is 640x576
    for (fw, fh) in ((640, 576), (640, 480), (512, 512)):
        try:
            print("depth fallback is being implemented")
            return plist.get_video_stream_profile(fw, fh, OBFormat.Y16, fps)
        except OBError:
            continue
    return plist.get_default_video_stream_profile()


def frame_to_bgr(color_frame):
    """Robust-ish conversion to BGR."""
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
        if fmt == getattr(OBFormat, "BGR", None):
            return data.reshape((h, w, 3))
        elif fmt == getattr(OBFormat, "RGB", None):
            rgb = data.reshape((h, w, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif fmt in (getattr(OBFormat, "MJPG", None), getattr(OBFormat, "MJPEG", None)):
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif fmt == getattr(OBFormat, "NV12", None):
            nv12 = data.reshape((h * 3 // 2, w))
            return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        elif fmt == getattr(OBFormat, "YUYV", None):
            yuy2 = data.reshape((h, w, 2))
            return cv2.cvtColor(yuy2, cv2.COLOR_YUV2BGR_YUY2)
        else:
            return None
    except Exception:
        return None


def median_depth_u16(depth_u16: np.ndarray, u: int, v: int, window: int = 5) -> int:
    """Median of non-zero depth (uint16) around (u,v). Returns 0 if none."""
    h, w = depth_u16.shape
    r = window // 2
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    patch = depth_u16[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return 0
    return int(np.median(valid))


def u16_to_meters(z_u16: int, scale: float) -> float:
    """
    Convert raw uint16 depth to meters.

    - If scale looks like meters/unit (typical ~0.001), use it.
    - If scale is 1.0 (common when units are mm), treat u16 as millimeters.
    """
    if scale is not None and 0.0 < scale <= 0.01:
        return z_u16 * scale
    # Heuristic: if scale is >= 0.1 or ~1.0, assume raw is millimeters.
    return z_u16 * 0.001


def main():
    screenshots_dir = ensure_dir("./screenshots")

    # --- Start pipeline with explicit 1280x720 requests (safe fallbacks)
    pipe = Pipeline()
    cfg  = Config()
    try:
        cprof = pick_color_profile(pipe, REQ_W, REQ_H, REQ_FPS)
        dprof = pick_depth_profile(pipe, REQ_W, REQ_H, REQ_FPS)
        cfg.enable_stream(cprof)
        cfg.enable_stream(dprof)
    except Exception as e:
        print("Failed to configure streams:", e)
        return

    try:
        pipe.enable_frame_sync()  # temporal sync (optional)
    except Exception:
        pass

    try:
        pipe.start(cfg)
    except Exception as e:
        print("Failed to start pipeline:", e)
        return

    # Align DEPTH -> COLOR
    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    # AprilTag detector
    detector = Detector(families=FAMILY, nthreads=2, quad_decimate=1.0,
                        quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)

    printed = False
    print("Running… press 'q' or ESC to quit, 'e' to screenshot.")
    try:
        while True:
            frames: FrameSet = pipe.wait_for_frames(100)
            if not frames:
                continue

            frames = align.process(frames)
            if not frames:
                continue
            frames = frames.as_frame_set()

            cframe = frames.get_color_frame()
            dframe = frames.get_depth_frame()
            if not cframe or not dframe:
                continue

            img = frame_to_bgr(cframe)
            if img is None:
                print("Unsupported color format. Try BGR/RGB/MJPG/NV12.")
                continue

            w_c, h_c = cframe.get_width(), cframe.get_height()
            w_d, h_d = dframe.get_width(), dframe.get_height()

            # if not printed or PRINT_ONCE:
            #     print(f"COLOR active: {w_c}x{h_c}  DEPTH(aligned): {w_d}x{h_d}")
            #     try:
            #         print("Depth scale (reported):", dframe.get_depth_scale())
            #     except Exception:
            #         pass
            #     printed = True

            if (w_c != w_d) or (h_c != h_d):
                print("Alignment failed (sizes differ).")
                continue

            depth_u16 = np.frombuffer(dframe.get_data(), dtype=np.uint16).reshape(h_d, w_d)
            scale = dframe.get_depth_scale()  # may be 0.001 (m/unit) or 1.0 (mm/unit)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray, estimate_tag_pose=False)

            for t in tags:
                corners = t.corners.astype(np.int32)
                cx, cy = map(int, t.center)

                cv2.polylines(img, [corners], True, (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

                z_u16 = median_depth_u16(depth_u16, cx, cy, window=WINDOW)
                Z_m = u16_to_meters(z_u16, scale)

                # Draw ROI box
                r = WINDOW // 2
                cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), (255, 200, 0), 1)

                valid = (z_u16 > 0) and (MIN_DEPTH_M <= Z_m <= MAX_DEPTH_M)
                if valid:
                    cv2.putText(img, f"Z ~ {Z_m:.3f} m", (cx + 8, cy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # Helpful debug text
                    cv2.putText(img, f"Z invalid (raw={z_u16}, scale={scale:.6f})",
                                (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

            cv2.imshow("AprilTag Z via ToF (Depth aligned to Color)", img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), ESC_KEY):
                break
            if k == ord('e'):
                fn = os.path.join(screenshots_dir, time.strftime("tof_z_%Y%m%d_%H%M%S.png"))
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
