#!/usr/bin/env python3
"""
Femto Bolt — Capture synchronized COLOR + IR pairs for stereo calibration.
- COLOR: BGR 1280x720 @ 30
- IR:    Y16 (native) — laser DISABLED so checkerboard is visible
- Right-click to save a pair when checkerboard is detected in BOTH streams.
"""
import os, cv2, datetime
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBPropertyID

CHECKERBOARD = (19, 19)
NUM_CAPTURES = 25
REQ_CW, REQ_CH, REQ_FPS = 1280, 720, 30
OUT_DIR = "callibration_captures"

capture_requested = False
def mouse_callback(event, x, y, flags, param):
    global capture_requested
    if event == cv2.EVENT_RBUTTONDOWN:
        capture_requested = True

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p

def pick_profile(pipeline, sensor, w, h, fmt, fps):
    plist = pipeline.get_stream_profile_list(sensor)
    return plist.get_video_stream_profile(w, h, fmt, fps)

def pick_ir_profile(pipeline):
    """Pick the highest-res IR profile available."""
    plist = pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
    # Try common Femto Bolt IR resolutions in order of preference
    for (w, h, fps) in [(640, 576, 30), (512, 512, 30), (320, 288, 30)]:
        try:
            return plist.get_video_stream_profile(w, h, OBFormat.Y16, fps)
        except Exception:
            continue
    raise RuntimeError("No suitable IR profile found")

def color_frame_to_bgr(cf):
    w, h = cf.get_width(), cf.get_height()
    buf = cf.get_data()
    data = np.frombuffer(buf, dtype=np.uint8)
    stride = None
    for name in ("get_stride_in_bytes", "get_line_stride_in_bytes"):
        if hasattr(cf, name):
            try: stride = int(getattr(cf, name)()); break
            except Exception: pass
    if stride is None: stride = w * 3
    row_cols = stride // 3
    return data[:stride*h].reshape(h, row_cols, 3)[:, :w, :].copy()

def ir_frame_to_u16(irf):
    w, h = irf.get_width(), irf.get_height()
    buf = irf.get_data()
    data = np.frombuffer(buf, dtype=np.uint16)
    return data[:w*h].reshape(h, w).copy()

def detect(gray):
    try:
        found, corners = cv2.findChessboardCornersSB(
            gray, CHECKERBOARD,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if found: return True, corners
    except Exception:
        pass
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    return cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

def main():
    ensure_dir(OUT_DIR)
    pipeline, cfg = Pipeline(), Config()

    cprof  = pick_profile(pipeline, OBSensorType.COLOR_SENSOR, REQ_CW, REQ_CH, OBFormat.BGR, REQ_FPS)
    irprof = pick_ir_profile(pipeline)
    cfg.enable_stream(cprof)
    cfg.enable_stream(irprof)

    from pyorbbecsdk import OBFrameAggregateOutputMode
    cfg.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipeline.enable_frame_sync()

    pipeline.start(cfg) 

    # Disable IR laser so checkerboard is visible in IR
    try:
        device = pipeline.get_device()
        device.set_bool_property(OBPropertyID.OB_PROP_LASER_BOOL, False)
        print("✅ IR laser disabled")
    except Exception as e:
        print(f"⚠️  Could not disable laser: {e}")
        print("    You may need to manually find the right property name.")

    print(f"📡 Color {cprof.get_width()}x{cprof.get_height()}  IR {irprof.get_width()}x{irprof.get_height()}")

    cv2.namedWindow("Capture")
    cv2.setMouseCallback("Capture", mouse_callback)
    saved = 0
    global capture_requested

    frame_idx = 0
    last_found_c, last_found_i = False, False

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27): break
                continue

            cf, irf = frames.get_color_frame(), frames.get_ir_frame()
            if cf is None or irf is None:
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27): break
                continue

            color = color_frame_to_bgr(cf)
            ir16  = ir_frame_to_u16(irf)
            ir8   = cv2.normalize(ir16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            frame_idx += 1
            frame_idx += 1
            if frame_idx % 5 == 0:
                gray_c = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                last_found_c, _ = detect(gray_c)
                last_found_i = detect(ir8)[0] if last_found_c else False
            found_c, found_i = last_found_c, last_found_i

            both = found_c and found_i
            disp_c = color.copy()
            disp_i = cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)
            border = (0,200,0) if both else (0,0,255)
            cv2.rectangle(disp_c, (0,0), (disp_c.shape[1]-1, disp_c.shape[0]-1), border, 6)
            cv2.rectangle(disp_i, (0,0), (disp_i.shape[1]-1, disp_i.shape[0]-1), border, 6)

            # Resize IR to match color height for side-by-side
            scale = disp_c.shape[0] / disp_i.shape[0]
            disp_i_r = cv2.resize(disp_i, (int(disp_i.shape[1]*scale), disp_c.shape[0]))
            view = np.hstack([disp_c, disp_i_r])
            txt = f"Color:{found_c}  IR:{found_i}  Saved:{saved}/{NUM_CAPTURES}  (right-click to save)"
            cv2.putText(view, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Capture", view)

            if capture_requested:
                capture_requested = False
                if both:
                    cv2.imwrite(f"{OUT_DIR}/color_{saved:02d}.png", color)
                    cv2.imwrite(f"{OUT_DIR}/ir_{saved:02d}.png", ir16)  # 16-bit PNG
                    saved += 1
                    print(f"  saved pair {saved}")
                    if saved >= NUM_CAPTURES: break
                else:
                    print("  skipped (board not in both)")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Done. {saved} pairs in {OUT_DIR}/")

if __name__ == "__main__":
    main()