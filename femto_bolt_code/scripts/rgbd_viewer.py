
#!/usr/bin/env python3
# RGB (UVC) + Depth (Orbbec SDK) viewer with:
# - separate Depth window (JET)
# - loud depth stats each second
# - auto-enable emitter/laser (best-effort)
# - rescue mode: multiple fallbacks to start depth
# Keys: q = quit, s = save RGB & Depth snapshots

import os, time, datetime
import numpy as np
import cv2

# ---------- Orbbec SDK (depth) ----------
try:
    from pyorbbecsdk import *
    OB_AVAILABLE = True
except Exception as e:
    print(f"⚠️  pyorbbecsdk not available: {e}")
    OB_AVAILABLE = False

# ---------- Config ----------
RGB_DEVICE = "/dev/video0"        # your RGB UVC node
RGB_W, RGB_H, RGB_FPS = 1280, 720, 30  # you can try 1920x1080 if your machine handles it

# ---------- Utilities ----------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def now_stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------- RGB via UVC ----------
def open_rgb_capture(dev=RGB_DEVICE, w=RGB_W, h=RGB_H, fps=RGB_FPS):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open {dev}")

    # Prefer YUYV (no JPEG decode)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y','U','Y','V'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)

    # Let settings settle
    for _ in range(5):
        cap.read()

    ok, frame = cap.read()
    if not ok or frame is None:
        print("⚠️  YUYV read failed; trying MJPG …")
        cap.release()
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to reopen {dev}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        for _ in range(5):
            cap.read()

    print(f"🎥 RGB: {dev} requested {w}x{h}@{fps}")
    return cap

# ---------- Depth helpers (SDK) ----------
def _fmt_name(x):
    s = str(x)
    return s.split('.')[-1] if '.' in s else s

def _try_set_bool(sensor, pid_name, value=True):
    if not hasattr(OBPropertyID, pid_name):
        return False
    try:
        sensor.set_bool_property(getattr(OBPropertyID, pid_name), bool(value))
        print(f"✔ set {pid_name} = {value}")
        return True
    except Exception:
        return False

def _try_set_int(sensor, pid_name, value):
    if not hasattr(OBPropertyID, pid_name):
        return False
    try:
        sensor.set_int_property(getattr(OBPropertyID, pid_name), int(value))
        print(f"✔ set {pid_name} = {value}")
        return True
    except Exception:
        return False

def _enable_emitters(device):
    """Best-effort: enable laser/projector/flood; silently skips unsupported props."""
    try:
        ds = device.get_sensor(OBSensorType.DEPTH_SENSOR)
    except Exception:
        return
    # Booleans
    for pid in ["OB_PROP_LASER_ENABLE_BOOL", "OB_PROP_LASER_CONTROL_BOOL", "OB_PROP_IR_FLOOD_BOOL"]:
        _try_set_bool(ds, pid, True)
    # Ints (tune if needed)
    _try_set_int(ds, "OB_PROP_LASER_POWER_INT", 100)
    # Optional IR exposure/gain (if available)
    _try_set_int(ds, "OB_PROP_IR_GAIN_INT", 1)
    _try_set_int(ds, "OB_PROP_IR_EXP_TIME_INT", 3000)

def _pick_depth_profile_from_device(device):
    """Enumerate via device sensor (more reliable on some builds)."""
    ds = device.get_sensor(OBSensorType.DEPTH_SENSOR)
    plist = ds.get_stream_profile_list()
    best = None
    i = 0
    while True:
        try:
            p = plist.get_profile(i)
        except Exception:
            break
        try:
            vp = p.as_video_stream_profile() if hasattr(p, "as_video_stream_profile") else p
            intr = vp.get_intrinsic()
            try: fps = vp.get_fps()
            except: fps = 30
            try: fmt = vp.get_format()
            except: fmt = "?"
            name = _fmt_name(fmt).upper()
            # Prefer Y16-ish formats at ~30fps and common sizes
            if name in ("Y16", "Y_U16", "DEPTH16"):
                if intr.width in (640, 576, 512, 480, 320) and int(fps) in (30, 25, 20, 15):
                    return vp
            if best is None:
                best = vp
        except Exception:
            pass
        i += 1
    return best  # may be None

# ---------- Depth pipeline (rescue mode) ----------
def start_depth_pipeline():
    if not OB_AVAILABLE:
        return None, None
    try:
        # More verbose SDK logs can help debugging
        os.environ.setdefault("OB_LOG_LEVEL", "debug")

        ctx = Context()
        devs = ctx.query_devices()
        if len(devs) == 0:
            print("⚠️  No Orbbec device for depth.")
            return None, None
        dev = devs[0]

        # 1) Try to enable emitters first (silent if not supported)
        _enable_emitters(dev)

        pipe = Pipeline(dev)
        cfg = Config()

        # 2) Device-sensor enumeration (more reliable on some builds)
        try:
            chosen = _pick_depth_profile_from_device(dev)
            if chosen is not None:
                cfg.enable_stream(chosen)
                try:
                    intr = chosen.get_intrinsic()
                    try: fps = chosen.get_fps()
                    except: fps = "?"
                    try: fmt = chosen.get_format()
                    except: fmt = "?"
                    print(f"🟦 Depth profile (device): {intr.width}x{intr.height} @ {fps} {fmt}")
                except Exception:
                    intr = None
                    print("🟦 Depth profile (device) chosen (intrinsics unavailable)")
                pipe.start(cfg)
                print("📡 Depth pipeline started (device)")
                return pipe, intr
            else:
                print("ℹ️  Device API returned 0 depth profiles.")
        except Exception as e:
            print(f"ℹ️  Device API enumeration failed: {e}")

        # 3) Pipeline API default
        try:
            plist = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            try:
                dp = plist.get_default_video_stream_profile()
                cfg = Config(); cfg.enable_stream(dp)
                intr = dp.get_intrinsic()
                try: fps = dp.get_fps()
                except: fps = "?"
                try: fmt = dp.get_format()
                except: fmt = "?"
                print(f"🟦 Depth profile (pipeline default): {intr.width}x{intr.height} @ {fps} {fmt}")
                pipe.start(cfg)
                print("📡 Depth pipeline started (pipeline default)")
                return pipe, intr
            except Exception as e:
                print(f"ℹ️  Pipeline default selection failed: {e}")

            # 4) Old wildcard trick (0,0,Y16,0)
            try:
                dp = plist.get_video_stream_profile(0, 0, OBFormat.Y16, 0)
                cfg = Config(); cfg.enable_stream(dp)
                intr = dp.get_intrinsic()
                print(f"🟦 Depth profile (wildcard): {intr.width}x{intr.height}")
                pipe.start(cfg)
                print("📡 Depth pipeline started (wildcard)")
                return pipe, intr
            except Exception as e:
                print(f"ℹ️  Pipeline wildcard failed: {e}")
        except Exception as e:
            print(f"ℹ️  Pipeline enumeration failed altogether: {e}")

        print("⚠️  Could not start depth (no profiles via device or pipeline).")
        print("   Tip: open OrbbecViewer → enable Depth & Emitter, then close it and re-run.")
        return None, None

    except Exception as e:
        print(f"⚠️  Failed to start depth pipeline: {e}")
        return None, None

# ---------- Depth frame conversion/visual ----------
def depth_to_numpy_u16(depth_frame, width, height):
    """C-contiguous uint16 (H,W) from the frame buffer, robust to SDK variants."""
    buf = depth_frame.get_data()
    n = width * height
    if isinstance(buf, np.ndarray):
        arr = np.asarray(buf, dtype=np.uint16)
    else:
        try:
            mv = memoryview(buf)
            arr = np.frombuffer(mv, dtype=np.uint16, count=n)
        except TypeError:
            arr = np.frombuffer(bytes(buf), dtype=np.uint16, count=n)
    if arr.size < n:
        return None
    return np.ascontiguousarray(arr[:n]).reshape((height, width))

def normalize_depth_for_display(depth_u16):
    """Auto-contrast depth to 8-bit + stats for visualization."""
    nz = depth_u16[depth_u16 > 0]
    if nz.size == 0:
        return np.zeros_like(depth_u16, dtype=np.uint8), 0.0, 0, 0, 0, 0
    p50 = float(np.percentile(nz, 50))
    p95 = float(np.percentile(nz, 95))
    dmin = int(nz.min()); dmax = int(nz.max())
    lo = max(int(np.percentile(nz, 1)), dmin)
    hi = max(int(np.percentile(nz, 99)), lo + 1)
    depth_8u = np.clip((depth_u16.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    pct_nz = 100.0 * nz.size / depth_u16.size
    return depth_8u, pct_nz, dmin, dmax, int(p50), int(p95)

# ---------- Main ----------
def main():
    # RGB
    cap = open_rgb_capture()
    # Depth
    pipe, dintr = start_depth_pipeline()

    out_dir = ensure_dir("captures")
    t0 = time.time()
    n_rgb = n_depth = 0
    last_stat = time.time()

    print("✅ Running…  (q=quit, s=save snapshots)")
    try:
        while True:
            # --- RGB ---
            ok_rgb, rgb_bgr = cap.read()
            if not ok_rgb or rgb_bgr is None:
                ok_rgb, rgb_bgr = cap.read()
            if ok_rgb and rgb_bgr is not None:
                n_rgb += 1

            # --- Depth ---
            depth_vis = None
            depth_u16 = None
            d_stats_text = "depth: --"
            d_stats_tuple = None

            if pipe is not None:
                try:
                    fs = pipe.wait_for_frames(100)  # small timeout
                except Exception:
                    fs = None
                if fs is not None:
                    d = fs.get_depth_frame()
                    if d is not None:
                        if dintr is not None:
                            w, h = dintr.width, dintr.height
                        else:
                            # some builds expose getters on frame
                            try:
                                w, h = d.get_width(), d.get_height()
                            except Exception:
                                w = h = None
                        if w and h:
                            depth_u16 = depth_to_numpy_u16(d, w, h)
                            if depth_u16 is not None:
                                n_depth += 1
                                depth_8u, pct_nz, dmin, dmax, p50, p95 = normalize_depth_for_display(depth_u16)
                                depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                                d_stats_text = f"depth: %NZ={pct_nz:.1f} min={dmin} max={dmax} p50={p50} p95={p95}"
                                d_stats_tuple = (pct_nz, dmin, dmax, p50, p95)

            # --- Compose side-by-side panel ---
            if ok_rgb and rgb_bgr is not None:
                left = rgb_bgr
            else:
                left = np.zeros((RGB_H, RGB_W, 3), dtype=np.uint8)

            if depth_vis is not None:
                h_target = left.shape[0]
                scale = h_target / depth_vis.shape[0]
                depth_rs = cv2.resize(depth_vis, (int(depth_vis.shape[1]*scale), h_target), interpolation=cv2.INTER_NEAREST)
            else:
                depth_rs = np.zeros((left.shape[0], left.shape[0]//2, 3), dtype=np.uint8)

            rgb_show = left.copy()
            cv2.putText(rgb_show, f"RGB {rgb_show.shape[1]}x{rgb_show.shape[0]}", (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            if depth_vis is not None:
                cv2.putText(depth_rs, d_stats_text, (10,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            else:
                cv2.putText(depth_rs, "depth: (not available)", (10,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

            panel = np.hstack([rgb_show, depth_rs])
            cv2.imshow("RGB | Depth", panel)

            # --- Separate Depth window if we have it ---
            if depth_vis is not None:
                cv2.imshow("Depth (JET)", depth_vis)

            # --- Loud console stats every ~1s ---
            now = time.time()
            if now - last_stat >= 1.0:
                fps_rgb   = n_rgb   / (now - t0)
                fps_depth = n_depth / (now - t0) if n_depth else 0.0
                if d_stats_tuple is not None:
                    pct_nz, dmin, dmax, p50, p95 = d_stats_tuple
                    print(f"FPS RGB {fps_rgb:.1f} | Depth {fps_depth:.1f} | %NZ {pct_nz:.1f} | min {dmin} max {dmax} | p50 {p50} p95 {p95}")
                else:
                    print(f"FPS RGB {fps_rgb:.1f} | Depth {fps_depth:.1f} | (no depth stats)")
                last_stat = now

            # --- Keys ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = now_stamp()
                if ok_rgb and rgb_bgr is not None:
                    p_rgb = os.path.join(out_dir, f"rgb_{ts}.png")
                    cv2.imwrite(p_rgb, rgb_bgr)
                    print(f"💾 saved {p_rgb}")
                if depth_u16 is not None and depth_vis is not None:
                    p_d8  = os.path.join(out_dir, f"depth_vis_{ts}.png")
                    p_d16 = os.path.join(out_dir, f"depth_u16_{ts}.npy")
                    cv2.imwrite(p_d8, depth_vis)
                    np.save(p_d16, depth_u16)
                    print(f"💾 saved {p_d8} and {p_d16}")

    finally:
        cap.release()
        if pipe is not None:
            try: pipe.stop()
            except Exception: pass
        cv2.destroyAllWindows()
        print("✅ Stopped cleanly.")

if __name__ == "__main__":
    main()
