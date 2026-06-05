#!/usr/bin/env python3
"""
Femto Bolt (Orbbec SDK) — COLOR intrinsics calibration with a checkerboard.
Basic, robust version:
- Forces a single stream: BGR @ 1280x720 @ 30 FPS (no format fallbacks).
- Handles stride safely; no NV12/YUYV/MJPG used.
- Gracefully waits for frames (no timeout crashes).
"""

import os, cv2, json, datetime
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBError

# ---------- Config ----------
CHECKERBOARD = (19, 19)      # inner corners (cols, rows)
SQUARE_SIZE_MM = 12.7        # mm per square
NUM_CAPTURES = 20
REQ_W, REQ_H, REQ_FPS = 1280, 720, 30
WAIT_TIMEOUT_MS = 1500       # generous per-frame wait
WARMUP_FRAMES = 8

# Optional beep on Windows (no-op elsewhere)
def _noop(*args, **kwargs): pass
try:
    import winsound
    def beep_detect():
        try: winsound.Beep(1200, 80)
        except Exception: pass
except Exception:
    beep_detect = _noop

# Mouse capture flag
capture_requested = False
def mouse_callback(event, x, y, flags, param):
    global capture_requested
    if event == cv2.EVENT_RBUTTONDOWN:
        capture_requested = True

# Fixed 3D object points (checkerboard grid in mm)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= float(SQUARE_SIZE_MM)

def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def timestamp():  return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")

def pick_color_profile_basic(pipeline):
    """Return exactly BGR @ REQ_W×REQ_H @ REQ_FPS. No fallbacks."""
    plist = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        return plist.get_video_stream_profile(REQ_W, REQ_H, OBFormat.BGR, REQ_FPS)
    except OBError as e:
        raise RuntimeError(
            f"BGR {REQ_W}x{REQ_H}@{REQ_FPS} not available on this device.\n"
            f"Please pick a supported BGR mode (see your printed profile list) or adjust REQ_*."
        ) from e

def frame_to_bgr_stride_safe(color_frame):
    """Read BGR with stride cropping (no conversions)."""
    if color_frame is None: return None
    w = color_frame.get_width(); h = color_frame.get_height()
    fmt = color_frame.get_format()
    if fmt != OBFormat.BGR:  # we configured BGR only
        return None
    buf = color_frame.get_data()
    data = np.frombuffer(buf if isinstance(buf, (bytes, bytearray)) else memoryview(buf), dtype=np.uint8)

    # Try to get per-line stride; fallback to tight if unknown
    stride = None
    for name in ("get_stride_in_bytes", "get_line_stride_in_bytes", "get_stride_bytes", "get_line_stride"):
        if hasattr(color_frame, name):
            try:
                stride = int(getattr(color_frame, name)())
                break
            except Exception:
                pass
    if stride is None: stride = w * 3
    if data.size < stride * h: return None

    row_cols = stride // 3
    bgr = data[:stride*h].reshape(h, row_cols, 3)[:, :w, :]  # crop to width
    return bgr

def draw_banner(img, text, ok=True):
    h, w = img.shape[:2]
    bar_h, pad = 48, 10
    bg = (40,120,40) if ok else (40,40,160)
    cv2.rectangle(img, (0,0), (w, bar_h+2*pad), bg, -1)
    led = (0,255,0) if ok else (0,0,255)
    cv2.circle(img, (pad+18, pad+bar_h//2), 12, (0,0,0), -1)
    cv2.circle(img, (pad+18, pad+bar_h//2), 10, led, -1)
    cv2.putText(img, text, (pad+44, pad+int(bar_h*0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

def draw_progress(img, n, total):
    h, w = img.shape[:2]
    margin, bar_h = 16, 14
    x0, y0 = margin, h - margin - bar_h
    bar_w = w - 2*margin
    cv2.rectangle(img, (x0, y0), (x0+bar_w, y0+bar_h), (200,200,200), 2)
    filled = int(bar_w * min(n, total) / max(1,total))
    cv2.rectangle(img, (x0+2, y0+2), (x0+filled-2, y0+bar_h-2), (0,200,0), -1)
    cv2.putText(img, f"{n}/{total} captures", (x0, y0-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

def draw_border(img, ok=True, thickness=8):
    h, w = img.shape[:2]
    color = (0,200,0) if ok else (0,0,255)
    cv2.rectangle(img, (0,0), (w-1,h-1), color, thickness)

def detect_corners(gray):
    """Prefer SB; fallback to classic (still robust, no FAST_CHECK)."""
    # SB is more accurate; some OpenCV builds might not have it—hence try/except
    try:
        found, corners = cv2.findChessboardCornersSB(
            gray, CHECKERBOARD,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if found:
            return True, corners
    except Exception:
        pass
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    if found:
        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
        return True, corners
    return False, None

def wait_for_valid_frame(pipeline, timeout_ms=WAIT_TIMEOUT_MS):
    """Loop until a valid color frame arrives; never raises on timeout."""
    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms)
        except Exception:
            cv2.waitKey(1)
            continue
        if frames is None:
            cv2.waitKey(1)
            continue
        cf = frames.get_color_frame()
        if cf is None:
            cv2.waitKey(1)
            continue
        return cf  # valid color frame

def main():
    out_dir = ensure_dir("calibration-results")
    pipeline, cfg = Pipeline(), Config()

    # Configure and start a single, basic stream
    vp = pick_color_profile_basic(pipeline)
    cfg.enable_stream(vp)
    pipeline.start(cfg)
    actual_w, actual_h, actual_fps = vp.get_width(), vp.get_height(), vp.get_fps()
    print(f"📡 Started color stream: {actual_w}x{actual_h}@{actual_fps} (BGR)")

    # Warm-up: let exposure settle; wait only for valid frames (no crashes)
    print("⏳ Warming up…")
    for _ in range(WARMUP_FRAMES):
        _ = wait_for_valid_frame(pipeline)

    objpoints, imgpoints = [], []
    cv2.namedWindow("Femto COLOR Calibration")
    cv2.setMouseCallback("Femto COLOR Calibration", mouse_callback)

    try:
        last_state = False
        while True:
            # Always get a valid frame or keep waiting
            color_frame = wait_for_valid_frame(pipeline)
            img = frame_to_bgr_stride_safe(color_frame)
            if img is None:  # extremely unlikely given we forced BGR
                cv2.waitKey(1); continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = detect_corners(gray)

            if found and not last_state:
                print("[INFO] Checkerboard detected. Right-click to capture.")
                beep_detect()
            elif not found and last_state:
                print("[INFO] Checkerboard lost.")
            last_state = found

            # HUD
            if found:
                cv2.drawChessboardCorners(img, CHECKERBOARD, corners, True)
                draw_banner(img, "Checkerboard: FOUND  ✓   Right-click to capture", ok=True)
            else:
                draw_banner(img, "Checkerboard: NOT FOUND", ok=False)
            draw_progress(img, len(objpoints), NUM_CAPTURES)
            draw_border(img, ok=found, thickness=8)
            cv2.putText(img, f"{actual_w}x{actual_h} @ {actual_fps}  BGR",
                        (18, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(img, "Right-click: capture | q/ESC: quit", (18, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow("Femto COLOR Calibration", img)

            # Capture
            global capture_requested
            if capture_requested:
                capture_requested = False
                if found:
                    objpoints.append(objp.copy())
                    imgpoints.append(corners.copy())
                    print(f"[INFO] Captured {len(objpoints)}/{NUM_CAPTURES}")
                else:
                    print("[WARN] Not captured (no checkerboard).")
                if len(objpoints) >= NUM_CAPTURES:
                    print("[INFO] Collected enough views. Calibrating…")
                    break

            # Quit keys
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                if len(objpoints) < 3:
                    print("[INFO] Quit. Not enough captures to calibrate.")
                    return
                break

        cv2.destroyAllWindows()

        if len(objpoints) < 3:
            print("[ERROR] Not enough valid captures. Aborting.")
            return

        # Calibrate
        image_size = (img.shape[1], img.shape[0])
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None,
            flags=0,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )

        print("\n=== Calibration Complete ===")
        print(f"RMS reprojection error: {rms:.4f} px")
        print("K:\n", K)
        print("Distortion (k1 k2 p1 p2 k3):", dist.ravel())

        ts = timestamp()
        base = os.path.join(out_dir, f"femto_color_intrinsics_{ts}")
        np.savez(base + ".npz", K=K, dist=dist, image_size=image_size,
                 checkerboard=CHECKERBOARD, square_size_mm=SQUARE_SIZE_MM, rms=rms)
        with open(base + ".json", "w", encoding="utf-8") as f:
            json.dump({
                "width": image_size[0], "height": image_size[1],
                "fx": float(K[0,0]), "fy": float(K[1,1]),
                "cx": float(K[0,2]), "cy": float(K[1,2]),
                "dist_coeffs": dist.ravel().tolist(),
                "checkerboard_inner_corners": {"cols": CHECKERBOARD[0], "rows": CHECKERBOARD[1]},
                "square_size_mm": float(SQUARE_SIZE_MM),
                "rms_px": float(rms)
            }, f, indent=2)
        print(f"Saved: {base+'.npz'}\nSaved: {base+'.json'}")

    finally:
        try: pipeline.stop()
        except Exception: pass
        cv2.destroyAllWindows()
        print("✅ Stopped cleanly.")

if __name__ == "__main__":
    main()
