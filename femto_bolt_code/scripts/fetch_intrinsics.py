#!/usr/bin/env python3
"""
Femto Bolt — Factory intrinsics/extrinsics extractor

- COLOR profile: EXACT (COLOR_W x COLOR_H @ FPS, BGR)
- DEPTH profile: EXACT (DEPTH_W x DEPTH_H @ FPS, Y16)
- Starts the pipeline with both streams enabled so get_camera_param()
  returns parameters matched to the active resolutions.
- Writes both:
    • per-profile intrinsics from stream_profile.get_intrinsic()
    • factory bundle from pipeline.get_camera_param()
      (rgb_intrinsic, depth_intrinsic, rgb_distortion, depth_distortion,
       transform = depth->color extrinsic)
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# ----- USER SETTINGS -----
COLOR_W, COLOR_H = 1280, 720      # color stream
DEPTH_W, DEPTH_H = 640, 576       # NFOV unbinned — best depth quality on Femto Bolt
FPS = 30
OUT_DIR = Path("./calibration-results")

# ----- SDK IMPORT -----
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBStreamType, OBFormat
except Exception as e:
    raise SystemExit(
        "ERROR: pyorbbecsdk not found or failed to import."
    ) from e


# ----- Helpers -----
def _require_exact_video_profile(pipe, sensor_type, w, h, fmt, fps):
    plist = pipe.get_stream_profile_list(sensor_type)
    try:
        return plist.get_video_stream_profile(w, h, fmt, fps)
    except Exception:
        sname = "COLOR" if sensor_type == OBSensorType.COLOR_SENSOR else "DEPTH"
        fmt_name = getattr(fmt, "name", str(fmt))
        raise SystemExit(
            f"ERROR: Required {sname} profile not available: "
            f"{w}x{h} @ {fps} {fmt_name}."
        )


def _intrinsic_to_dict(intr):
    """Parse OBCameraIntrinsic (factory) — fields: fx, fy, cx, cy, width, height."""
    fx = float(intr.fx)
    fy = float(intr.fy)
    cx = float(intr.cx)
    cy = float(intr.cy)
    width  = int(intr.width)
    height = int(intr.height)
    return {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": width, "height": height,
        "K": [[fx, 0.0, cx],
              [0.0, fy, cy],
              [0.0, 0.0, 1.0]],
    }


def _distortion_to_dict(dist):
    """Parse OBCameraDistortion — Brown-Conrady: k1..k6, p1, p2."""
    keys = ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")
    out = {}
    for k in keys:
        if hasattr(dist, k):
            out[k] = float(getattr(dist, k))
    # OpenCV-style ordering: [k1, k2, p1, p2, k3, k4, k5, k6]
    out["opencv_order"] = [
        out.get("k1", 0.0), out.get("k2", 0.0),
        out.get("p1", 0.0), out.get("p2", 0.0),
        out.get("k3", 0.0), out.get("k4", 0.0),
        out.get("k5", 0.0), out.get("k6", 0.0),
    ]
    return out


def _transform_to_dict(xform, translation_in_mm=True):
    """Parse OBD2CTransform / OBExtrinsic — fields: rot (9 floats), trans (3 floats)."""
    R = np.array(xform.rot, dtype=float).reshape(3, 3)
    t = np.array(xform.transform, dtype=float).reshape(3)
    if translation_in_mm:
        t_m = t * 1e-3
    else:
        t_m = t.copy()
    return {
        "R": R.tolist(),
        "t_mm": t.tolist(),
        "t_m": t_m.tolist(),
        "note": "X_color = R @ X_depth + t  (applied in matching units)",
    }


# ----- Main -----
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")

    if not hasattr(OBFormat, "BGR"):
        raise SystemExit("ERROR: This SDK build has no OBFormat.BGR.")

    pipe = Pipeline()

    # cam = pipe.get_camera_param()
    # print(dir(cam))                    # look for transform / d2c_transform / extrinsic
    # print(dir(cam.transform))          # look for rot/rotation, trans/translation
    # print(dir(cam.rgb_intrinsic))      # confirm fx, fy, cx, cy, width, height
    # print(dir(cam.rgb_distortion))     # confirm k1..k6, p1, p2

    # 1. Resolve EXACT profiles
    cprof = _require_exact_video_profile(
        pipe, OBSensorType.COLOR_SENSOR, COLOR_W, COLOR_H, OBFormat.BGR, FPS
    )
    dprof = _require_exact_video_profile(
        pipe, OBSensorType.DEPTH_SENSOR, DEPTH_W, DEPTH_H, OBFormat.Y16, FPS
    )

    # 2. Per-stream intrinsics (these come straight from the profile)
    cintr_profile = _intrinsic_to_dict(cprof.get_intrinsic())
    dintr_profile = _intrinsic_to_dict(dprof.get_intrinsic())
    d2c_profile   = _transform_to_dict(dprof.get_extrinsic_to(cprof))

    # 3. Start pipeline with BOTH streams enabled so get_camera_param()
    #    returns parameters matched to the active resolution pair.
    cfg = Config()
    cfg.enable_stream(cprof)
    cfg.enable_stream(dprof)
    pipe.start(cfg)
    try:
        cam = pipe.get_camera_param()

        rgb_intr_factory   = _intrinsic_to_dict(cam.rgb_intrinsic)
        depth_intr_factory = _intrinsic_to_dict(cam.depth_intrinsic)
        rgb_dist_factory   = _distortion_to_dict(cam.rgb_distortion)
        depth_dist_factory = _distortion_to_dict(cam.depth_distortion)
        d2c_factory        = _transform_to_dict(cam.transform)
    finally:
        pipe.stop()

    bundle = {
        "device": "Orbbec Femto Bolt",
        "timestamp": ts,
        "streams": {
            "color": {"width": COLOR_W, "height": COLOR_H, "fps": FPS, "format": "BGR"},
            "depth": {"width": DEPTH_W, "height": DEPTH_H, "fps": FPS, "format": "Y16"},
        },

        # From the individual stream profiles
        "from_stream_profile": {
            "color_intrinsics": cintr_profile,
            "depth_intrinsics": dintr_profile,
            "depth_to_color":   d2c_profile,
        },

        # From pipeline.get_camera_param() — preferred for alignment math
        "from_camera_param": {
            "color_intrinsics":  rgb_intr_factory,
            "depth_intrinsics":  depth_intr_factory,
            "color_distortion":  rgb_dist_factory,
            "depth_distortion":  depth_dist_factory,
            "depth_to_color":    d2c_factory,
        },
    }

    out_path = OUT_DIR / f"factory_calib_{COLOR_W}x{COLOR_H}_d{DEPTH_W}x{DEPTH_H}_{ts}.json"
    out_path.write_text(json.dumps(bundle, indent=2))

    print("\n[OK] Saved factory calibration bundle:")
    print(f"  Color : {COLOR_W} x {COLOR_H} @ {FPS} BGR")
    print(f"  Depth : {DEPTH_W} x {DEPTH_H} @ {FPS} Y16")
    print(f"  File  : {out_path}\n")

    # Quick sanity print
    print("RGB  intrinsics (factory):", rgb_intr_factory)
    print("Depth intrinsics (factory):", depth_intr_factory)
    print("Depth->Color t (meters):", d2c_factory["t_m"])


if __name__ == "__main__":
    main()