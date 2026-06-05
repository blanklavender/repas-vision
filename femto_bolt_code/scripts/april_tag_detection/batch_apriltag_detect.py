#!/usr/bin/env python3
"""
Batch AprilTag detection.

Reads every image in an input folder, runs tag36h11 detection + pose estimation
using camera parameters loaded from a calibration file, and writes annotated
images for SUCCESSFUL detections only into an output folder.

Usage:
    python batch_apriltag_detect.py --images ./captures --calib ./camera_params.txt
    python batch_apriltag_detect.py -i ./captures -c ./camera_params.txt -o ./out --tag-size 0.0303
    python batch_apriltag_detect.py -i ..\..\..\flora_turntable\captures\yellow_pot_08-25_deg_10_speed_5 --c ..\..\..\femto_bolt_code\scripts\calibration_parameters\CameraParam_Femto_BoltCL8956200ZM_Color1280x720_Depth640x576.ini -o ..\..\..\flora_turntable\captures\april_tag_yellow_pot_08-25_deg_10_speed_5 --tag-size 0.030

"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
from pupil_apriltags import Detector

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}


# ------------------------------------------------------------------ calibration
def parse_calibration(path):
    """Parse an INI-style calibration file into {section: {key: float}}."""
    sections = {}
    current = None
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(("#", ";")):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                sections.setdefault(current, {})
                continue
            if current is None or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip()
            try:
                sections[current][key] = float(val)
            except ValueError:
                sections[current][key] = val
    return sections


def build_camera_model(sections, stream="Color"):
    """Return (K, dist_coeffs, (width, height)) for 'Color' or 'Depth' stream."""
    intr_key = f"{stream}Intrinsic"
    dist_key = f"{stream}Distortion"
    if intr_key not in sections:
        raise KeyError(f"[{intr_key}] not found in calibration file. "
                       f"Sections present: {list(sections)}")
    i = sections[intr_key]
    K = np.array([[i["fx"], 0.0,     i["cx"]],
                  [0.0,     i["fy"], i["cy"]],
                  [0.0,     0.0,     1.0    ]], dtype=np.float64)

    d = sections.get(dist_key, {})
    k1, k2, k3 = d.get("k1", 0.0), d.get("k2", 0.0), d.get("k3", 0.0)
    k4, k5, k6 = d.get("k4", 0.0), d.get("k5", 0.0), d.get("k6", 0.0)
    p1, p2 = d.get("p1", 0.0), d.get("p2", 0.0)

    # OpenCV order. Use the 8-param rational model only if k4..k6 are non-zero.
    if any(abs(v) > 1e-12 for v in (k4, k5, k6)):
        dist = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float64)
    else:
        dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    size = (int(i.get("width", 0)), int(i.get("height", 0)))
    return K, dist, size


def build_extrinsics(sections, key="D2CTransformParam"):
    """Return (R 3x3, t 3x1) from a rot0..rot8 / trans0..trans2 section, or None."""
    if key not in sections:
        return None
    s = sections[key]
    try:
        R = np.array([s[f"rot{n}"] for n in range(9)], dtype=np.float64).reshape(3, 3)
        t = np.array([s[f"trans{n}"] for n in range(3)], dtype=np.float64).reshape(3, 1)
    except KeyError:
        return None
    return R, t


# ------------------------------------------------------------------ detection
def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def make_detector(quad_decimate, quad_sigma, families, nthreads):
    return Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )


def detect_best(gray, detectors, margin_thresh):
    """Try (variant x detector-setting) combos; return first set of good tags."""
    variants = (("orig", gray), ("clahe", enhance_contrast(gray)))
    for vname, g in variants:
        for dname, det in detectors:
            tags = det.detect(g)
            good = [t for t in tags if t.decision_margin >= margin_thresh]
            if good:
                return good, f"{vname}/{dname}"
    return [], None


def annotate(img, tags, K, dist, tag_size, draw_axes=True):
    """Draw detections + pose on img (in place). Returns list of pose dicts."""
    h = tag_size / 2.0
    obj_pts = np.array([[-h, -h, 0],
                        [ h, -h, 0],
                        [ h,  h, 0],
                        [-h,  h, 0]], dtype=np.float32)
    poses = []
    for tag in tags:
        corners = tag.corners.astype(np.float32)
        cv2.polylines(img, [corners.astype(int)], True, (0, 255, 0), 2)
        cx, cy = map(int, tag.center)
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img, f"ID {tag.tag_id}", (cx - 20, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        ok, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, dist,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, dist,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            poses.append({"tag_id": tag.tag_id,
                          "margin": float(tag.decision_margin),
                          "rvec": None, "tvec": None})
            continue

        if draw_axes:
            cv2.drawFrameAxes(img, K, dist, rvec, tvec, tag_size * 0.5, 2)

        t = tvec.ravel()
        txt = f"X:{t[0]:.3f} Y:{t[1]:.3f} Z:{t[2]:.3f} m"
        cv2.putText(img, txt, (cx + 10, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, txt, (cx + 10, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        poses.append({"tag_id": int(tag.tag_id),
                      "margin": float(tag.decision_margin),
                      "rvec": rvec.ravel().tolist(),
                      "tvec": t.tolist()})
    return poses


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Batch AprilTag detection over a folder of images.")
    ap.add_argument("-i", "--images", required=True, help="Folder containing input images")
    ap.add_argument("-c", "--calib", required=True, help="Path to calibration (intrinsics/extrinsics) file")
    ap.add_argument("-o", "--output", default=None,
                    help="Output folder (default: <images>_apriltag_detections)")
    ap.add_argument("--stream", default="Color", choices=["Color", "Depth"],
                    help="Which intrinsics block to use (default: Color)")
    ap.add_argument("--tag-size", type=float, default=0.0303, help="Tag side length in meters")
    ap.add_argument("--margin", type=float, default=10.0, help="Min decision_margin to accept")
    ap.add_argument("--families", default="tag36h11", help="AprilTag family")
    ap.add_argument("--nthreads", type=int, default=4)
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--no-axes", action="store_true", help="Skip drawing pose axes")
    args = ap.parse_args()

    img_dir = os.path.abspath(os.path.expanduser(args.images))
    if not os.path.isdir(img_dir):
        sys.exit(f"ERROR: not a folder: {img_dir}")

    calib_path = os.path.abspath(os.path.expanduser(args.calib))
    if not os.path.isfile(calib_path):
        sys.exit(f"ERROR: calibration file not found: {calib_path}")

    out_dir = args.output or (img_dir.rstrip(os.sep) + "_apriltag_detections")
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    sections = parse_calibration(calib_path)
    K, dist, cal_size = build_camera_model(sections, args.stream)
    ext = build_extrinsics(sections)

    print(f"Calibration : {calib_path}")
    print(f"K =\n{K}")
    print(f"dist = {dist.ravel()}")
    if cal_size != (0, 0):
        print(f"calibrated resolution = {cal_size[0]}x{cal_size[1]}")
    if ext is not None:
        R, t = ext
        print(f"D2C extrinsics R =\n{R}\nD2C extrinsics t = {t.ravel()}")
    print(f"Input  : {img_dir}\nOutput : {out_dir}\n")

    # gather images
    files = []
    if args.recursive:
        for root, _, names in os.walk(img_dir):
            for n in names:
                if os.path.splitext(n)[1].lower() in IMAGE_EXTS:
                    files.append(os.path.join(root, n))
    else:
        for n in os.listdir(img_dir):
            p = os.path.join(img_dir, n)
            if os.path.isfile(p) and os.path.splitext(n)[1].lower() in IMAGE_EXTS:
                files.append(p)
    files.sort()
    if not files:
        sys.exit(f"ERROR: no images found in {img_dir}")

    detectors = [
        ("fast", make_detector(1.0, 0.0, args.families, args.nthreads)),
        ("thorough", make_detector(0.5, 1.0, args.families, args.nthreads)),
    ]

    csv_path = os.path.join(out_dir, "detections.csv")
    n_ok = n_fail = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["image", "tag_id", "decision_margin", "pipeline",
                         "rx", "ry", "rz", "tx_m", "ty_m", "tz_m"])

        for path in files:
            img = cv2.imread(path)
            if img is None:
                print(f"[SKIP] unreadable: {path}")
                n_fail += 1
                continue

            # warn once if image size differs from calibrated size
            if cal_size != (0, 0) and (img.shape[1], img.shape[0]) != cal_size:
                print(f"[WARN] {os.path.basename(path)} is {img.shape[1]}x{img.shape[0]} "
                      f"but calibration is {cal_size[0]}x{cal_size[1]} — pose will be wrong.")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            tags, pipeline = detect_best(gray, detectors, args.margin)
            if not tags:
                n_fail += 1
                print(f"[MISS] {os.path.basename(path)}")
                continue

            poses = annotate(img, tags, K, dist, args.tag_size, draw_axes=not args.no_axes)
            cv2.putText(img, f"{len(tags)} tag(s) | {pipeline}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            rel = os.path.relpath(path, img_dir)
            out_name = rel.replace(os.sep, "_")
            stem, e = os.path.splitext(out_name)
            out_path = os.path.join(out_dir, f"{stem}_detected{e or '.png'}")
            cv2.imwrite(out_path, img)
            n_ok += 1

            for p in poses:
                r = p["rvec"] or ["", "", ""]
                tv = p["tvec"] or ["", "", ""]
                writer.writerow([rel, p["tag_id"], f"{p['margin']:.2f}", pipeline,
                                 *[f"{v:.6f}" if isinstance(v, float) else v for v in r],
                                 *[f"{v:.6f}" if isinstance(v, float) else v for v in tv]])
            ids = ", ".join(str(p["tag_id"]) for p in poses)
            print(f"[ OK ] {rel} -> {os.path.basename(out_path)}  (ids: {ids})")

    print(f"\nDone. {n_ok} succeeded, {n_fail} with no detection, {len(files)} total.")
    print(f"Images : {out_dir}")
    print(f"CSV    : {csv_path}")


if __name__ == "__main__":
    main()