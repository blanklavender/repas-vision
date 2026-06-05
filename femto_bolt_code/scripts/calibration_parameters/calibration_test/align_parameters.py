#!/usr/bin/env python3
"""
Stereo calibration: Orbbec Femto Bolt IR (depth) <-> RGB.

Pipeline
--------
1. Load paired IR + RGB images of a checkerboard (one capture per view,
   filenames sorted in matching order).
2. Detect inner-corner grid in both modalities for each pair.
3. Run cv2.calibrateCamera per camera using the factory intrinsics as the
   initial guess (CALIB_USE_INTRINSIC_GUESS) and the 8-coef rational model
   (CALIB_RATIONAL_MODEL) — matches Orbbec's opencv_order layout.
4. Run cv2.stereoCalibrate with CALIB_FIX_INTRINSIC to solve only for the
   rigid transform R, T (depth -> color, units = square_size units = mm).
5. Save results in the same JSON layout as the factory file.

Expected directory layout (override with --ir-dir / --rgb-dir):
    captures/
        ir/   ir_000.png  ir_001.png  ...    (Y16 16-bit or 8-bit grayscale)
        rgb/  rgb_000.png rgb_001.png ...    (BGR color)

Pairs are matched positionally after sorted globbing, so ir_000 <-> rgb_000.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------- Defaults for your setup ----------
DEFAULT_BOARD_INNER = (21, 20)         # inner corners for a 20x20-square board
DEFAULT_SQUARE_SIZE_MM = 25.4          # 10 inches / 20 squares = 0.5" = 25.4 mm
DEFAULT_FACTORY_JSON = "../calibration-results/factory_calib_1280x720_d640x576_2026-04-13T164910.json"
DEFAULT_IR_DIR = "captures/ir"
DEFAULT_RGB_DIR = "captures/rgb"
DEFAULT_OUTPUT = "femto_bolt_user_calib.json"

COLOR_SIZE = (1280, 720)               # (W, H)
IR_SIZE = (640, 576)


# ---------- I/O ----------
def load_factory(path):
    with open(path) as f:
        data = json.load(f)
    blk = data["from_camera_param"]
    K_color = np.array(blk["color_intrinsics"]["K"], dtype=np.float64)
    K_ir = np.array(blk["depth_intrinsics"]["K"], dtype=np.float64)
    # Orbbec stores [k1,k2,p1,p2,k3,k4,k5,k6] which is exactly OpenCV's order.
    D_color = np.array(blk["color_distortion"]["opencv_order"], dtype=np.float64).reshape(1, 8)
    D_ir = np.array(blk["depth_distortion"]["opencv_order"], dtype=np.float64).reshape(1, 8)
    R0 = np.array(blk["depth_to_color"]["R"], dtype=np.float64)
    t0_mm = np.array(blk["depth_to_color"]["t_mm"], dtype=np.float64)
    return K_color, D_color, K_ir, D_ir, R0, t0_mm


# ---------- Image prep ----------
def to_uint8_gray(img):
    """Make any IR image (Y16 / Y8 / BGR) into 8-bit single channel."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        # Robust percentile stretch — IR has long tails.
        lo, hi = np.percentile(img, (1, 99))
        if hi <= lo:
            hi = lo + 1
        out = np.clip((img.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
        return out.astype(np.uint8)
    return img


def detect_corners(gray, pattern_size):
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        # Try an inverted pass — IR is sometimes inverted vs RGB on a checkerboard.
        found, corners = cv2.findChessboardCorners(255 - gray, pattern_size, flags)
        if not found:
            return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def make_object_points(rows, cols, square_size):
    """Object points for one view, planar Z=0, units = square_size (mm)."""
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def collect_pairs(ir_dir, rgb_dir, pattern_size, square_size, debug_dir=None):
    exts = ("*.png", "*.tif", "*.tiff", "*.jpg")
    ir_paths = sorted(p for e in exts for p in Path(ir_dir).glob(e))
    rgb_paths = sorted(p for e in exts for p in Path(rgb_dir).glob(e))
    if len(ir_paths) != len(rgb_paths):
        print(f"[warn] {len(ir_paths)} IR vs {len(rgb_paths)} RGB; using min.", file=sys.stderr)
    n = min(len(ir_paths), len(rgb_paths))

    objp = make_object_points(pattern_size[1], pattern_size[0], square_size)
    obj_pts, ir_pts, rgb_pts, used_pairs = [], [], [], []

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        ir_raw = cv2.imread(str(ir_paths[i]), cv2.IMREAD_UNCHANGED)
        rgb = cv2.imread(str(rgb_paths[i]), cv2.IMREAD_COLOR)
        if ir_raw is None or rgb is None:
            print(f"[skip] could not read pair {i}: {ir_paths[i].name} / {rgb_paths[i].name}")
            continue

        # Sanity-check sizes (warn but don't fail).
        if rgb.shape[1::-1] != COLOR_SIZE:
            print(f"[warn] RGB {rgb_paths[i].name} is {rgb.shape[1::-1]}, expected {COLOR_SIZE}")
        if ir_raw.shape[1::-1] != IR_SIZE:
            print(f"[warn] IR  {ir_paths[i].name} is {ir_raw.shape[1::-1]}, expected {IR_SIZE}")

        ir_gray = to_uint8_gray(ir_raw)
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        ok_ir, ir_corners = detect_corners(ir_gray, pattern_size)
        ok_rgb, rgb_corners = detect_corners(rgb_gray, pattern_size)
        status = f"IR={'OK' if ok_ir else 'MISS'}  RGB={'OK' if ok_rgb else 'MISS'}"
        print(f"[{i:03d}] {ir_paths[i].name:<28s} {rgb_paths[i].name:<28s} {status}")

        if ok_ir and ok_rgb:
            obj_pts.append(objp.copy())
            ir_pts.append(ir_corners)
            rgb_pts.append(rgb_corners)
            used_pairs.append([ir_paths[i].name, rgb_paths[i].name])

            if debug_dir is not None:
                ir_vis = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(ir_vis, pattern_size, ir_corners, True)
                cv2.drawChessboardCorners(rgb, pattern_size, rgb_corners, True)
                cv2.imwrite(str(debug_dir / f"ir_{i:03d}.png"), ir_vis)
                cv2.imwrite(str(debug_dir / f"rgb_{i:03d}.png"), rgb)

    return obj_pts, ir_pts, rgb_pts, used_pairs


# ---------- Calibration ----------
def per_view_reproj_error(obj_pts, img_pts, rvecs, tvecs, K, D):
    errs = []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, D)
        e = np.linalg.norm(img_pts[i] - proj, axis=2).ravel()
        errs.append(float(np.sqrt(np.mean(e ** 2))))
    return errs


def calibrate_single(obj_pts, img_pts, image_size, K_init, D_init, name):
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_RATIONAL_MODEL
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, image_size, K_init.copy(), D_init.copy(),
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
    )
    per_view = per_view_reproj_error(obj_pts, img_pts, rvecs, tvecs, K, D)
    print(f"\n=== {name} single calib ===")
    print(f"  Overall RMS px       : {rms:.4f}")
    print(f"  Per-view  median/max : {np.median(per_view):.4f} / {np.max(per_view):.4f}")
    print(f"  fx, fy               : {K[0,0]:.3f}, {K[1,1]:.3f}")
    print(f"  cx, cy               : {K[0,2]:.3f}, {K[1,2]:.3f}")
    return K, D, rms, per_view


def stereo_calibrate(obj_pts, ir_pts, rgb_pts, K_ir, D_ir, K_rgb, D_rgb):
    flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    # imageSize here is unused when CALIB_FIX_INTRINSIC is set; pass IR size.
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, ir_pts, rgb_pts,
        K_ir, D_ir, K_rgb, D_rgb,
        IR_SIZE,
        flags=flags, criteria=criteria,
    )
    print(f"\n=== Stereo (IR -> RGB) ===")
    print(f"  Stereo RMS px : {rms:.4f}")
    print(f"  T (mm)        : [{T[0,0]:.3f}, {T[1,0]:.3f}, {T[2,0]:.3f}]")
    print(f"  ||T||  (mm)   : {float(np.linalg.norm(T)):.3f}")
    rotvec, _ = cv2.Rodrigues(R)
    print(f"  R as rotvec   : [{rotvec[0,0]:+.5f}, {rotvec[1,0]:+.5f}, {rotvec[2,0]:+.5f}]")
    return R, T, E, F, rms


# ---------- Output ----------
def to_orbbec_distortion_dict(D):
    """OpenCV order [k1,k2,p1,p2,k3,k4,k5,k6] -> Orbbec keys."""
    d = D.ravel().tolist()
    while len(d) < 8:
        d.append(0.0)
    return {
        "k1": d[0], "k2": d[1], "k3": d[4],
        "k4": d[5], "k5": d[6], "k6": d[7],
        "p1": d[2], "p2": d[3],
        "opencv_order": d[:8],
    }


def K_to_intrinsics_dict(K, w, h):
    return {
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "width": int(w), "height": int(h),
        "K": K.tolist(),
    }


def save_results(out_path, K_color, D_color, K_ir, D_ir, R, T_mm, meta):
    out = {
        "device": "Orbbec Femto Bolt",
        "calibration_source": "user stereo calibration (OpenCV)",
        "meta": meta,
        "user_calibrated": {
            "color_intrinsics": K_to_intrinsics_dict(K_color, *COLOR_SIZE),
            "depth_intrinsics": K_to_intrinsics_dict(K_ir, *IR_SIZE),
            "color_distortion": to_orbbec_distortion_dict(D_color),
            "depth_distortion": to_orbbec_distortion_dict(D_ir),
            "depth_to_color": {
                "R": np.asarray(R).tolist(),
                "t_mm": np.asarray(T_mm).ravel().tolist(),
                "t_m": (np.asarray(T_mm).ravel() / 1000.0).tolist(),
                "note": "X_color = R @ X_depth + t  (applied in matching units)",
            },
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}")


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory-json", default=DEFAULT_FACTORY_JSON)
    ap.add_argument("--ir-dir", default=DEFAULT_IR_DIR)
    ap.add_argument("--rgb-dir", default=DEFAULT_RGB_DIR)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--rows", type=int, default=DEFAULT_BOARD_INNER[1],
                    help="inner-corner rows (squares-1 along height)")
    ap.add_argument("--cols", type=int, default=DEFAULT_BOARD_INNER[0],
                    help="inner-corner cols (squares-1 along width)")
    ap.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_SIZE_MM)
    ap.add_argument("--debug-dir", default=None,
                    help="if set, dump per-view corner overlays here")
    args = ap.parse_args()

    pattern_size = (args.cols, args.rows)  # OpenCV order: (cols, rows)
    print(f"Board pattern (inner corners, cols x rows): {pattern_size}")
    print(f"Square size: {args.square_mm} mm")

    K_c0, D_c0, K_i0, D_i0, R0, t0 = load_factory(args.factory_json)

    obj_pts, ir_pts, rgb_pts, pairs = collect_pairs(
        args.ir_dir, args.rgb_dir, pattern_size, args.square_mm,
        debug_dir=args.debug_dir,
    )
    print(f"\nUsable pairs: {len(obj_pts)}")
    if len(obj_pts) < 5:
        print("ERROR: too few views to calibrate (need >=5, ideally 15-30).", file=sys.stderr)
        sys.exit(1)
    if len(obj_pts) < 12:
        print("WARNING: <12 views; calibration may be unreliable.", file=sys.stderr)

    K_ir, D_ir, rms_ir, _ = calibrate_single(
        obj_pts, ir_pts, IR_SIZE, K_i0, D_i0, "IR / depth")
    K_rgb, D_rgb, rms_rgb, _ = calibrate_single(
        obj_pts, rgb_pts, COLOR_SIZE, K_c0, D_c0, "RGB / color")

    R, T, E, F, rms_stereo = stereo_calibrate(
        obj_pts, ir_pts, rgb_pts, K_ir, D_ir, K_rgb, D_rgb,
    )

    meta = {
        "num_views": len(obj_pts),
        "pattern_inner_corners": list(pattern_size),
        "square_size_mm": args.square_mm,
        "rms_ir_px": float(rms_ir),
        "rms_rgb_px": float(rms_rgb),
        "rms_stereo_px": float(rms_stereo),
        "pairs": pairs,
    }
    save_results(args.output, K_rgb, D_rgb, K_ir, D_ir, R, T, meta)

    # ---- Side-by-side comparison vs factory ----
    print("\n--- vs factory ---")
    print(f"  IR  fx,fy   factory: {K_i0[0,0]:.3f}, {K_i0[1,1]:.3f}   "
          f"ours: {K_ir[0,0]:.3f}, {K_ir[1,1]:.3f}")
    print(f"  IR  cx,cy   factory: {K_i0[0,2]:.3f}, {K_i0[1,2]:.3f}   "
          f"ours: {K_ir[0,2]:.3f}, {K_ir[1,2]:.3f}")
    print(f"  RGB fx,fy   factory: {K_c0[0,0]:.3f}, {K_c0[1,1]:.3f}   "
          f"ours: {K_rgb[0,0]:.3f}, {K_rgb[1,1]:.3f}")
    print(f"  RGB cx,cy   factory: {K_c0[0,2]:.3f}, {K_c0[1,2]:.3f}   "
          f"ours: {K_rgb[0,2]:.3f}, {K_rgb[1,2]:.3f}")
    print(f"  T (mm)      factory: [{t0[0]:.3f}, {t0[1]:.3f}, {t0[2]:.3f}]   "
          f"ours: [{T[0,0]:.3f}, {T[1,0]:.3f}, {T[2,0]:.3f}]")
    # Rotation difference: R_diff = R @ R0.T  -> angle should be small (a few deg)
    R_diff = R @ R0.T
    angle_deg = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
    print(f"  Angle(R, R_factory) : {angle_deg:.4f} deg")


if __name__ == "__main__":
    main()