#!/usr/bin/env python3
"""
Flora Inspection Station -- automated turntable sweep with the Femto Bolt.

Rotates the turntable in fixed increments and, at every stop, saves:

    color_000_ang000.png    8-bit BGR, 1280x720
    depth_000_ang000.png    16-bit, D2C-aligned to the colour frame
    cloud_000_ang000.ply    binary xyzrgb, from the SDK's PointCloudFilter
    manifest.csv            index / commanded angle / true angle / files / time
    intrinsics.json         fx fy cx cy, depth scale, units -- written once

Matches the Orbbec Viewer settings: D2C(SW) alignment, colour 1280x720 30fps
MJPG, depth 640x576 30fps Y16, frame sync on, xyzrgb clouds.

    python sweep_capture.py COM5 --step 5 --speed 5 --out captures/plant_01

    python sweep_capture.py COM7 --step 10 --speed 5 --out captures/yellow_pot_08-25_deg_10_speed_5 --settle 3

Keys (with a preview window focused):  q / ESC = abort.

Requires turntable.py in the same folder.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pyorbbecsdk import (
    Pipeline, Config, Context, OBSensorType, OBFormat, OBError,
    AlignFilter, OBStreamType, OBFrameAggregateOutputMode,
    PointCloudFilter, save_point_cloud_to_ply,
)

from turntable import Turntable, TurntableError

ESC = 27
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 576, 30

# Firmware truncates degrees->steps, so a commanded 5 deg is really 4.966 deg.
# Deterministic, so we log the real number for downstream registration.
STEPS_PER_DEG = (2048 * 4) / 360.0


# --------------------------------------------------------------------------
# camera plumbing (from the working Femto preview script)
# --------------------------------------------------------------------------

def select_profile(profile_list, w, h, fps, formats):
    for fmt in formats:
        try:
            p = profile_list.get_video_stream_profile(w, h, fmt, fps)
            if p is not None:
                return p
        except OBError:
            pass
    return profile_list.get_default_video_stream_profile()


def color_to_bgr(frame):
    fmt = frame.get_format()
    w, h = frame.get_width(), frame.get_height()
    data = frame.get_data()
    if fmt in (OBFormat.BGR, OBFormat.RGB, OBFormat.BGRA, OBFormat.RGBA):
        ch = 4 if fmt in (OBFormat.BGRA, OBFormat.RGBA) else 3
        img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, ch)).copy()
        if fmt in (OBFormat.RGB, OBFormat.RGBA):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img[:, :, :3]
    if fmt == OBFormat.NV12:
        y = np.frombuffer(data, dtype=np.uint8, count=w * h).reshape((h, w))
        uv = np.frombuffer(data, dtype=np.uint8, offset=w * h).reshape((h // 2, w))
        return cv2.cvtColor(np.vstack((y, uv)), cv2.COLOR_YUV2BGR_NV12)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    raise RuntimeError(f'Unsupported color format: {fmt}')


def depth_to_u16(frame):
    w, h = frame.get_width(), frame.get_height()
    return np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((h, w))


def get_intrinsics(depth_frame):
    """Post-align, the depth frame carries the colour intrinsics."""
    try:
        intr = depth_frame.get_stream_profile().as_video_stream_profile().get_intrinsic()
        return dict(fx=float(intr.fx), fy=float(intr.fy),
                    cx=float(intr.cx), cy=float(intr.cy),
                    width=int(intr.width), height=int(intr.height))
    except Exception as e:
        return {"error": f"could not read intrinsics: {e}"}


def get_depth_scale(depth_frame):
    """mm per raw depth unit. 1.0 on the Femto Bolt, but ask rather than assume."""
    try:
        return float(depth_frame.get_depth_scale())
    except Exception:
        return 1.0


# --------------------------------------------------------------------------
# point cloud -- SDK PointCloudFilter, same path the Orbbec Viewer uses
# --------------------------------------------------------------------------

def count_valid_points(pc_frame):
    """RGB_POINT is 6 x float32 per point (x y z r g b). Count the non-zero ones."""
    try:
        data = np.frombuffer(pc_frame.as_points_frame().get_data(), dtype=np.float32)
        pts = data.reshape(-1, 6)
        return int(np.count_nonzero(pts[:, 2] > 0)), int(pts.shape[0])
    except Exception:
        return -1, -1


# --------------------------------------------------------------------------
# frame acquisition
# --------------------------------------------------------------------------

def grab_settled(pipe, align, seconds, preview=True):
    """Drain the queue for `seconds`, return the newest complete aligned set.

    This matters: the pipeline keeps buffering while the table is turning, so
    without draining you save a frame shot mid-motion.
    """
    deadline = time.time() + seconds
    last = None
    while time.time() < deadline or last is None:
        frames = pipe.wait_for_frames(100)
        if frames is None:
            if time.time() > deadline + 5.0:
                raise RuntimeError("no frames from the camera for 5 s")
            continue
        aligned = align.process(frames)
        if not aligned:
            continue
        fs = aligned.as_frame_set()
        c, d = fs.get_color_frame(), fs.get_depth_frame()
        if c is None or d is None:
            continue
        last = (color_to_bgr(c), depth_to_u16(d), d, aligned)

        if preview:
            show(last[0], last[1], "settling")
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), ESC):
                raise KeyboardInterrupt
    return last


def show(color, depth, label):
    vis = cv2.resize(color, (640, 360))
    cv2.putText(vis, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('color', vis)
    dvis = cv2.convertScaleAbs(depth, alpha=0.05)
    cv2.imshow('depth', cv2.applyColorMap(dvis, cv2.COLORMAP_JET))


# --------------------------------------------------------------------------

def true_deg(step, count):
    per_move = int(abs(step) * STEPS_PER_DEG) / STEPS_PER_DEG
    return per_move * count * (1 if step > 0 else -1)


def main():
    p = argparse.ArgumentParser(description="Turntable sweep capture on the Femto Bolt")
    p.add_argument("port", help="COM5 on Windows, /dev/ttyACM0 on Ubuntu")
    p.add_argument("--step", type=int, default=5, help="degrees between captures")
    p.add_argument("--speed", type=int, default=5, help="turntable speed 1-9")
    p.add_argument("--out", default=None, help="output directory")
    p.add_argument("--total", type=int, default=360, help="degrees to sweep")
    p.add_argument("--settle", type=float, default=2.0,
                   help="seconds to let the leaves stop moving after each step")
    p.add_argument("--warmup", type=float, default=3.0,
                   help="seconds of auto-exposure warm-up before the first shot")
    p.add_argument("--decimate", type=int, default=1,
                   help="PointCloudFilter decimation factor; 2 quarters the point count")
    p.add_argument("--units", choices=["mm", "m"], default="mm",
                   help="point cloud units. mm is the SDK/Viewer default")
    p.add_argument("--ascii-ply", action="store_true",
                   help="ASCII PLY instead of binary (much larger, human readable)")
    p.add_argument("--no-ply", action="store_true", help="skip point clouds")
    p.add_argument("--no-preview", action="store_true", help="headless")
    args = p.parse_args()

    preview = not args.no_preview
    n_pos = args.total // args.step
    out = Path(args.out or f"captures/sweep_{datetime.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)

    if Context().query_devices().get_count() == 0:
        sys.exit("No Orbbec camera found. Check the USB connection.")

    # ---- camera ----------------------------------------------------------
    pipe = Pipeline()
    cfg = Config()
    cfg.enable_stream(select_profile(
        pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR),
        DEPTH_W, DEPTH_H, DEPTH_FPS, [OBFormat.Y16]))
    cfg.enable_stream(select_profile(
        pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR),
        COLOR_W, COLOR_H, COLOR_FPS, [OBFormat.MJPG, OBFormat.NV12, OBFormat.BGR]))
    cfg.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipe.enable_frame_sync()
    pipe.start(cfg)

    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)   # D2C(SW)

    pcf = None
    if not args.no_ply:
        pcf = PointCloudFilter()
        pcf.set_create_point_format(OBFormat.RGB_POINT)   # xyzrgb, as in the Viewer
        if args.decimate > 1:
            pcf.set_decimation_factor(args.decimate)

    # ---- turntable (opening the port resets the board -> it homes) -------
    print(f"Connecting to the turntable on {args.port} -- it homes, ~15 s ...")
    try:
        tt = Turntable(args.port)
        tt.set_speed(args.speed)
        tt.home()
    except TurntableError as e:
        pipe.stop()
        sys.exit(f"Turntable: {e}")
    except Exception as e:
        pipe.stop()
        sys.exit(f"Could not open {args.port}: {e}\n"
                 f"Is the Arduino IDE Serial Monitor or another script holding it?")

    print(f"Warming up the camera for {args.warmup:.0f} s ...")
    rows, intr, scale, units_applied = [], None, None, "mm"

    try:
        color, depth, dframe, _ = grab_settled(pipe, align, args.warmup, preview)

        # Units. The SDK emits millimetres by default, which is what the Viewer
        # exports. set_position_data_scaled() rescales that; we ask the depth
        # frame for its own scale rather than assuming 1.0.
        scale = get_depth_scale(dframe)
        if pcf is not None and args.units == "m":
            try:
                pcf.set_position_data_scaled(scale / 1000.0)
                units_applied = "m"
            except Exception as e:
                print(f"[WARN] could not set metre scaling ({e}); staying in mm")
        elif pcf is not None:
            try:
                pcf.set_position_data_scaled(scale)
            except Exception:
                pass   # default is already mm

        print(f"\n{n_pos} positions, {args.step} deg apart "
              f"({true_deg(args.step, 1):.4f} deg true). "
              f"Clouds in {units_applied}. Writing to {out}\n")

        for i in range(n_pos):
            commanded = (i * args.step) % 360
            real = true_deg(args.step, i)

            color, depth, dframe, aligned = grab_settled(pipe, align, args.settle, preview)

            if intr is None:
                intr = get_intrinsics(dframe)
                (out / "intrinsics.json").write_text(json.dumps({
                    "note": "colour intrinsics; depth is D2C-aligned to colour",
                    "intrinsics": intr,
                    "depth_scale_mm_per_unit": scale,
                    "point_cloud_units": units_applied,
                    "point_cloud_source": "pyorbbecsdk PointCloudFilter, RGB_POINT",
                    "decimation_factor": args.decimate,
                    "color": f"{COLOR_W}x{COLOR_H}@{COLOR_FPS} MJPG",
                    "depth": f"{DEPTH_W}x{DEPTH_H}@{DEPTH_FPS} Y16",
                    "align_mode": "D2C software (AlignFilter -> COLOR_STREAM)",
                }, indent=2))

            stem = f"{i:03d}_ang{commanded:03d}"
            cname, dname = f"color_{stem}.png", f"depth_{stem}.png"
            cv2.imwrite(str(out / cname), color)
            cv2.imwrite(str(out / dname), depth)

            pname, npts, ntotal = "", 0, 0
            if pcf is not None:
                pc_frame = pcf.process(aligned)
                if pc_frame is None:
                    print(f"[WARN] no point cloud at {commanded} deg")
                else:
                    pname = f"cloud_{stem}.ply"
                    save_point_cloud_to_ply(str(out / pname), pc_frame,
                                            save_binary=not args.ascii_ply)
                    npts, ntotal = count_valid_points(pc_frame)

            rows.append({
                "index": i,
                "commanded_deg": commanded,
                "true_deg": round(real, 4),
                "color": cname, "depth": dname, "cloud": pname,
                "valid_points": npts, "total_points": ntotal,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            print(f"[{i + 1:3d}/{n_pos}] {commanded:3d} deg "
                  f"(true {real:7.3f})  {npts:>8d} valid pts")

            if preview:
                show(color, depth, f"saved {commanded} deg")
                cv2.waitKey(1)

            if i + 1 < n_pos:
                tt.move(args.step)

    except KeyboardInterrupt:
        print("\naborted")
    except Exception as e:
        print(f"\nstopped: {e}")
    finally:
        if rows:
            with open(out / "manifest.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"\n{len(rows)} positions -> {out}")
            print(f"total true rotation: {rows[-1]['true_deg']:.3f} deg")
        try:
            tt.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        pipe.stop()


if __name__ == '__main__':
    main()