#!/usr/bin/env python3
"""
Femto Bolt — Factory intrinsics/extrinsics for EXACT WxH (BGR color only)

- COLOR profile: EXACT (WIDTH x HEIGHT @ FPS, format=BGR) — errors if missing
- DEPTH profile: EXACT (WIDTH x HEIGHT @ FPS, format=Y16) — errors if missing
- Fetches:
    • COLOR intrinsics (fx, fy, cx, cy, width, height, K, distortion if present)
    • DEPTH intrinsics (same fields)
    • DEPTH -> COLOR extrinsics (R, t) in meters
- Writes a single bundle JSON to ./calibration-results:
    factory_calib_exact_{WIDTH}x{HEIGHT}_{timestamp}.json
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# ----- USER SETTINGS -----
WIDTH, HEIGHT, FPS = 1280, 720, 30
OUT_DIR = Path("./calibration-results")

# ----- SDK IMPORT -----
try:
    from pyorbbecsdk import Pipeline, OBSensorType, OBFormat
except Exception as e:
    raise SystemExit(
        "ERROR: pyorbbecsdk not found or failed to import. Install Orbbec SDK Python bindings."
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
            f"ERROR: Required {sname} profile not available: {w}x{h} @ {fps} {fmt_name}."
            "\nNo fallbacks are allowed in this script. "
            "Change WIDTH/HEIGHT/FPS or enable this mode on the device."
        )

def _intrinsic_to_dict(intr):
    fx = float(getattr(intr, 'fx', 0.0))
    fy = float(getattr(intr, 'fy', 0.0))
    cx = float(getattr(intr, 'cx', 0.0))
    cy = float(getattr(intr, 'cy', 0.0))
    width  = int(getattr(intr, 'width',  0))
    height = int(getattr(intr, 'height', 0))

    # Best-effort distortion read (SDK variants differ)
    dist_model = getattr(intr, 'distortion_model', None)
    coeffs = None
    for key in ('distortion_coeffs', 'coeffs', 'k'):
        if hasattr(intr, key):
            try:
                coeffs = np.array(getattr(intr, key), dtype=float).ravel().tolist()
                break
            except Exception:
                pass

    out = {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "width": width, "height": height,
        "K": [[fx, 0.0, cx],
              [0.0, fy, cy],
              [0.0, 0.0, 1.0]]
    }
    if dist_model is not None:
        out["distortion"] = {
            "model": str(dist_model),
            "coeffs": coeffs if coeffs is not None else []
        }
    return out

def _extrinsic_to_dict(extr):
    if hasattr(extr, 'rotation'):
        R = np.array(extr.rotation, dtype=float).reshape(3, 3)
    elif hasattr(extr, 'get_rotation'):
        R = np.array(extr.get_rotation(), dtype=float).reshape(3, 3)
    else:
        R = np.eye(3, dtype=float)

    if hasattr(extr, 'translation'):
        t = np.array(extr.translation, dtype=float).reshape(3)
    elif hasattr(extr, 'get_translation'):
        t = np.array(extr.get_translation(), dtype=float).reshape(3)
    else:
        t = np.zeros(3, dtype=float)

    return {"R": R.tolist(), "t": t.tolist()}


# ----- Main -----
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")

    pipe = Pipeline()

    # EXACT profiles only
    # COLOR=BGR (hard requirement)
    if not hasattr(OBFormat, "BGR"):
        raise SystemExit("ERROR: This SDK build has no OBFormat.BGR. Install a build that exposes BGR.")

    cprof = _require_exact_video_profile(
        pipe, OBSensorType.COLOR_SENSOR, WIDTH, HEIGHT, OBFormat.BGR, FPS
    )
    dprof = _require_exact_video_profile(
        pipe, OBSensorType.DEPTH_SENSOR, WIDTH, HEIGHT, OBFormat.Y16, FPS
    )

    # Fetch intrinsics and extrinsics (DEPTH -> COLOR)
    cintr = _intrinsic_to_dict(cprof.get_intrinsic())
    dintr = _intrinsic_to_dict(dprof.get_intrinsic())
    d2c   = _extrinsic_to_dict(dprof.get_extrinsic_to(cprof))

    bundle = {
        "device": "Orbbec Femto Bolt",
        "timestamp": ts,
        "requested_exact": {"width": WIDTH, "height": HEIGHT, "fps": FPS, "color_format": "BGR", "depth_format": "Y16"},
        "color_intrinsics": cintr,
        "depth_intrinsics": dintr,
        "extrinsics": {
            "depth_to_color": d2c  # X_c = R * X_d + t
        }
    }

    out_path = OUT_DIR / f"factory_calib_exact_{WIDTH}x{HEIGHT}_{ts}.json"
    out_path.write_text(json.dumps(bundle, indent=2))

    print("\n[OK] Saved exact-resolution calibration (BGR color only):")
    print(f"  Resolution : {WIDTH} x {HEIGHT} @ {FPS}")
    print(f"  Bundle JSON: {out_path}\n")

if __name__ == "__main__":
    main()
