#!/usr/bin/env python3
"""
Capture paired IR + RGB images for stereo calibration on Orbbec Femto Bolt.

Streams (raw, NOT D2C-aligned — we are CALIBRATING the alignment):
  Color : 1280x720 @ 30 fps, MJPG decoded to BGR
  IR    : 640x576  @ 30 fps, Y16 (saved as 16-bit PNG, no normalization)
  Depth : enabled because IR + depth share the ToF sensor on Femto Bolt;
          we don't save depth.

Output:
  <out>/rgb/rgb_000.png   8-bit BGR
  <out>/ir/ir_000.png     16-bit Y16

Keys:
  SPACE   save current pair (saves regardless of detection; status logged)
  a       toggle auto-capture (saves when both corners detected + cooldown)
  d       delete most recent pair
  q / ESC quit
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from pyorbbecsdk import (
    Config,
    OBFormat,
    OBSensorType,
    Pipeline,
)

# ---------- Defaults for your setup ----------
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
IR_W, IR_H, IR_FPS = 640, 576, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 576, 30
DEFAULT_BOARD_INNER = (21, 20)        # 20x20-square board -> 19x19 inner corners
AUTO_COOLDOWN_S = 1.5                 # min seconds between auto saves


# ---------- Frame decoding ----------
def color_to_bgr(frame):
    fmt = frame.get_format()
    w, h = frame.get_width(), frame.get_height()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3).copy()
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    if fmt == OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    raise RuntimeError(f"Unsupported color format: {fmt}")


def ir_to_u16(frame):
    w, h = frame.get_width(), frame.get_height()
    data = np.frombuffer(frame.get_data(), dtype=np.uint16)
    return data.reshape(h, w).copy()


def ir_for_display(ir_u16):
    lo, hi = np.percentile(ir_u16, (1, 99))
    if hi <= lo:
        hi = lo + 1
    return np.clip((ir_u16.astype(np.float32) - lo) * 255.0 / (hi - lo),
                   0, 255).astype(np.uint8)


# ---------- Corner detection (fast, for live feedback only) ----------
def quick_corners(gray, pattern_size):
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        ok, corners = cv2.findChessboardCorners(255 - gray, pattern_size, flags)
    return ok, corners


# ---------- Capture session ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="captures")
    ap.add_argument("--rows", type=int, default=DEFAULT_BOARD_INNER[1],
                    help="inner-corner rows (squares-1 along board height)")
    ap.add_argument("--cols", type=int, default=DEFAULT_BOARD_INNER[0],
                    help="inner-corner cols (squares-1 along board width)")
    ap.add_argument("--start-idx", type=int, default=0)
    args = ap.parse_args()

    pattern_size = (args.cols, args.rows)
    out_root = Path(args.out)
    ir_dir = out_root / "ir"
    rgb_dir = out_root / "rgb"
    ir_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    # ---- Pipeline ----
    pipeline = Pipeline()
    config = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_video_stream_profile(
        COLOR_W, COLOR_H, OBFormat.MJPG, COLOR_FPS)
    config.enable_stream(color_profile)

    ir_profiles = pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
    ir_profile = ir_profiles.get_video_stream_profile(
        IR_W, IR_H, OBFormat.Y16, IR_FPS)
    config.enable_stream(ir_profile)

    # On Femto Bolt the depth and IR share the ToF sensor; enabling depth
    # makes IR delivery reliable. We just don't read/save depth.
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_video_stream_profile(
        DEPTH_W, DEPTH_H, OBFormat.Y16, DEPTH_FPS)
    config.enable_stream(depth_profile)

    # Do NOT enable D2C alignment / hardware sync here — we want raw streams.

    pipeline.start(config)
    print(f"Streaming. Pattern (cols x rows) = {pattern_size}")
    print(f"Saving to {out_root.resolve()}")
    print("SPACE save | a auto-toggle | d delete-last | q/ESC quit")

    idx = args.start_idx
    saved_paths = []          # list of (ir_path, rgb_path) for delete-last
    auto_capture = False
    last_auto_save = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames(200)
            if frames is None:
                continue
            cf = frames.get_color_frame()
            ir = frames.get_ir_frame()
            if cf is None or ir is None:
                continue

            rgb = color_to_bgr(cf)
            if rgb is None:
                continue
            ir_u16 = ir_to_u16(ir)
            ir_disp = ir_for_display(ir_u16)

            rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            ok_rgb, c_rgb = quick_corners(rgb_gray, pattern_size)
            ok_ir, c_ir = quick_corners(ir_disp, pattern_size)
            both_ok = ok_rgb and ok_ir

            # Build display views
            rgb_view = rgb.copy()
            ir_view = cv2.cvtColor(ir_disp, cv2.COLOR_GRAY2BGR)
            if ok_rgb:
                cv2.drawChessboardCorners(rgb_view, pattern_size, c_rgb, True)
            if ok_ir:
                cv2.drawChessboardCorners(ir_view, pattern_size, c_ir, True)

            color_ok = (0, 255, 0) if both_ok else (0, 165, 255)
            mode = "AUTO" if auto_capture else "MANUAL"
            label = (f"[{idx:03d}]  RGB={'OK' if ok_rgb else '..'}"
                     f"  IR={'OK' if ok_ir else '..'}  mode={mode}")
            for img in (rgb_view, ir_view):
                cv2.putText(img, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_ok, 2)

            cv2.imshow("RGB (1280x720)", rgb_view)
            cv2.imshow("IR (640x576)", ir_view)

            # ---- Save logic ----
            now = time.time()
            do_save = False
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            if key == ord(' '):
                do_save = True
            elif key == ord('a'):
                auto_capture = not auto_capture
                print(f"  auto-capture {'ON' if auto_capture else 'OFF'}")
            elif key == ord('d'):
                if saved_paths:
                    last_ir, last_rgb = saved_paths.pop()
                    for p in (last_ir, last_rgb):
                        try:
                            p.unlink()
                        except FileNotFoundError:
                            pass
                    idx -= 1
                    print(f"  deleted pair {idx:03d}")
                else:
                    print("  nothing to delete")
            elif auto_capture and both_ok and (now - last_auto_save) >= AUTO_COOLDOWN_S:
                do_save = True
                last_auto_save = now

            if do_save:
                rgb_path = rgb_dir / f"rgb_{idx:03d}.png"
                ir_path = ir_dir / f"ir_{idx:03d}.png"
                cv2.imwrite(str(rgb_path), rgb)
                cv2.imwrite(str(ir_path), ir_u16)        # 16-bit PNG
                tag = "OK" if both_ok else "no-corners"
                print(f"  saved [{idx:03d}] {tag}  rgb={rgb_path.name}  ir={ir_path.name}")
                saved_paths.append((ir_path, rgb_path))
                idx += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Done. {idx - args.start_idx} pair(s) saved to {out_root.resolve()}")


if __name__ == "__main__":
    main()