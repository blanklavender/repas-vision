#!/usr/bin/env python3
"""
Flora Inspection Station -- automated turntable sweep with the Femto Bolt, ONE shot per stop.

Rotates the turntable in fixed increments; at every stop it waits --settle seconds (the
camera queue is drained meanwhile, so nothing shot mid-motion is kept), then saves the
newest synchronised frame:

    color_000_ang000.png    8-bit BGR, 1280x720
    depth_000_ang000.png    16-bit, D2C-aligned to the colour frame (depth_scale mm per unit)
    cloud_000_ang000.ply    binary xyzrgb from the SDK's PointCloudFilter (skip with --no-ply)
    manifest.csv            index / commanded angle / true angle / files / time, rewritten
                            after every stop so an aborted sweep still has a valid one
    intrinsics.json         colour intrinsics, depth scale, units -- written once
    camera_params.json      native depth + colour intrinsics/distortion and the
                            depth->colour extrinsic -- written once

(Earlier versions also saved a burst of raw depth + IR frames per stop for averaging;
averaging made no real difference, so that is gone. Old captures keep working downstream.)

Matches the Orbbec Viewer settings: D2C(SW) alignment, colour 1280x720 30fps MJPG,
depth 640x576 30fps Y16, frame sync on, xyzrgb clouds.

    python sweep_capture.py COM7 --step 10 --speed 5 --settle 3 --out captures/yellow_pot_10-06_deg_10_speed_5
    python sweep_capture.py /dev/ttyACM0 --step 10 --out <path> --no-ply --no-preview --home-after

Keys (with a preview window focused):  q / ESC = abort.

Requires turntable.py in the same folder and flora_pipeline/flora_lib (repository root).
"""

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flora_pipeline"))
from flora_lib import dataset  # noqa: E402  (shared capture layout)

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


def _intr(intr):
    return dict(fx=float(intr.fx), fy=float(intr.fy), cx=float(intr.cx), cy=float(intr.cy),
                width=int(intr.width), height=int(intr.height))


def _dist(d):
    """OBCameraDistortion -> dict plus OpenCV order [k1 k2 p1 p2 k3 k4 k5 k6]."""
    out = {k: float(getattr(d, k)) for k in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")}
    out["opencv_order"] = [out[k] for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")]
    return out


def get_camera_params(raw_depth_frame, color_frame, depth_scale):
    """Native (un-aligned) depth + colour calibration, straight from the stream
    profiles (same calls as the SDK's coordinate_transform example)."""
    dp = raw_depth_frame.get_stream_profile().as_video_stream_profile()
    cp = color_frame.get_stream_profile().as_video_stream_profile()
    ext = dp.get_extrinsic_to(cp)
    return {
        "note": ("depth_intrinsic/distortion describe the native (un-aligned) depth "
                 "sensor. X_color = R @ X_depth + t, t in mm."),
        "depth_intrinsic": _intr(dp.get_intrinsic()),
        "depth_distortion": _dist(dp.get_distortion()),
        "color_intrinsic": _intr(cp.get_intrinsic()),
        "color_distortion": _dist(cp.get_distortion()),
        "depth_to_color": {"R": np.array(ext.rot, dtype=float).reshape(3, 3).tolist(),
                           "t_mm": np.array(ext.transform, dtype=float).reshape(3).tolist()},
        "depth_scale_mm_per_unit": float(depth_scale),
    }


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
    """Drain the queue for `seconds`, return the newest complete aligned set as
    (colour BGR, aligned depth u16, aligned depth frame, aligned frameset,
    raw depth frame, raw colour frame).

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
        raw_d, raw_c = frames.get_depth_frame(), frames.get_color_frame()
        aligned = align.process(frames)
        if not aligned or raw_d is None or raw_c is None:
            continue
        fs = aligned.as_frame_set()
        c, d = fs.get_color_frame(), fs.get_depth_frame()
        if c is None or d is None:
            continue
        last = (color_to_bgr(c), depth_to_u16(d).copy(), d, aligned, raw_d, raw_c)

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
    p = argparse.ArgumentParser(description="Turntable sweep capture on the Femto Bolt, one shot per stop",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("port", help="COM5 on Windows, /dev/ttyACM0 on Ubuntu")
    p.add_argument("--step", type=int, default=10, help="degrees between captures")
    p.add_argument("--speed", type=int, default=5, help="turntable speed 1-9")
    p.add_argument("--out", default=None, help="output directory (default captures/sweep_<time>)")
    p.add_argument("--settle", type=float, default=3.0,
                   help="seconds to let the leaves stop moving after each step")
    p.add_argument("--total", type=int, default=360, help="degrees to sweep")
    p.add_argument("--warmup", type=float, default=3.0,
                   help="seconds of auto-exposure warm-up before the first shot")
    p.add_argument("--home-after", action="store_true",
                   help="re-home the turntable after the sweep, so the plant stands as in the first shot")
    p.add_argument("--decimate", type=int, default=1,
                   help="PointCloudFilter decimation factor; 2 quarters the point count")
    p.add_argument("--units", choices=["mm", "m"], default="mm",
                   help="SDK point cloud units. mm is the SDK/Viewer default")
    p.add_argument("--ascii-ply", action="store_true",
                   help="ASCII PLY instead of binary (much larger, human readable)")
    p.add_argument("--no-ply", action="store_true", help="skip the SDK point clouds (not used downstream)")
    p.add_argument("--no-preview", action="store_true", help="headless")
    args = p.parse_args()
    if not 1 <= args.speed <= 9:
        p.error("--speed must be 1-9")
    if args.step == 0:
        p.error("--step must not be 0")

    preview = not args.no_preview
    n_pos = args.total // abs(args.step)
    out = Path(args.out or f"captures/sweep_{datetime.now():%Y%m%d_%H%M%S}")
    if (out / dataset.MANIFEST).exists():
        sys.exit(f"{out} already holds a capture (manifest.csv) -- choose another --out")
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
    rows, scale, units_applied, status = [], None, "mm", 0

    try:
        _, _, dframe, *_ = grab_settled(pipe, align, args.warmup, preview)

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
              f"({true_deg(args.step, 1):.4f} deg true), settle {args.settle:g} s. Writing to {out}\n")

        for i in range(n_pos):
            commanded = (i * args.step) % 360
            real = true_deg(args.step, i)

            # one shot: the newest frame once the table has been still for --settle s
            color, depth, dframe, aligned, raw_d, raw_c = grab_settled(pipe, align, args.settle, preview)

            if i == 0:
                cam = get_camera_params(raw_d, raw_c, scale)
                (out / dataset.CAMERA_PARAMS).write_text(json.dumps(cam, indent=2))
                (out / "intrinsics.json").write_text(json.dumps({
                    "note": "colour intrinsics; depth is D2C-aligned to colour",
                    "intrinsics": get_intrinsics(dframe),
                    "depth_scale_mm_per_unit": scale,
                    "point_cloud_units": units_applied,
                    "point_cloud_source": "pyorbbecsdk PointCloudFilter, RGB_POINT",
                    "decimation_factor": args.decimate,
                    "color": f"{COLOR_W}x{COLOR_H}@{COLOR_FPS} MJPG",
                    "depth": f"{DEPTH_W}x{DEPTH_H}@{DEPTH_FPS} Y16",
                    "align_mode": "D2C software (AlignFilter -> COLOR_STREAM)",
                }, indent=2))

            stem = dataset.stem(i, commanded)
            cname, dname = dataset.color_name(stem), dataset.depth_name(stem)
            if not (cv2.imwrite(str(out / cname), color) and cv2.imwrite(str(out / dname), depth)):
                raise RuntimeError(f"could not write {out / cname} / {dname}")

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
            dataset.write_manifest(out, rows)
            print(f"[{i + 1:3d}/{n_pos}] {commanded:3d} deg (true {real:7.3f})  "
                  f"{int((depth > 0).sum()):>8d} valid depth px" + (f", {npts} cloud pts" if pcf else ""))

            if preview:
                show(color, depth, f"saved {commanded} deg")
                cv2.waitKey(1)

            if i + 1 < n_pos:
                tt.move(args.step)

        if args.home_after:
            print("\nre-homing the turntable ...")
            tt.home()

    except KeyboardInterrupt:
        print("\naborted")
        status = 130
    except Exception as e:
        print(f"\nstopped: {e}")
        status = 1
    finally:
        if rows:
            print(f"\n{len(rows)}/{n_pos} positions -> {out}")
            print(f"total true rotation: {rows[-1]['true_deg']:.3f} deg")
        try:
            tt.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        pipe.stop()
    sys.exit(status)


if __name__ == '__main__':
    main()
