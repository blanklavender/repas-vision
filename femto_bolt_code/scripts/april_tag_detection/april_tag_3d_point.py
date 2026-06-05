#!/usr/bin/env python3
"""
AprilTag 3D position (X, Y, Z) from a color image + depth image pair.

Usage:
    python apriltag_xyz.py color.png depth.png
    python apriltag_xyz.py color.png depth.png --intrinsics intrinsics.json
"""

import argparse, json
import cv2
import numpy as np
from pupil_apriltags import Detector

FAMILY       = "tag36h11"
DEPTH_WINDOW = 5
MIN_Z, MAX_Z = 0.25, 8.0
DEPTH_SCALE  = 0.001  # raw uint16 → meters (1 count = 1mm for Femto Bolt)


def load_intrinsics(path):
    with open(path) as f:
        k = json.load(f)
    return k["fx"], k["fy"], k["cx"], k["cy"]


def median_depth_m(depth_u16, u, v, win=5):
    h, w = depth_u16.shape
    r = win // 2
    patch = depth_u16[max(0,v-r):min(h,v+r+1), max(0,u-r):min(w,u+r+1)]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    z_m = float(np.median(valid)) * DEPTH_SCALE
    return z_m if MIN_Z <= z_m <= MAX_Z else None


def backproject(u, v, z, fx, fy, cx, cy):
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return x, y, z


def main():
    parser = argparse.ArgumentParser(description="AprilTag XYZ from color + depth images")
    parser.add_argument("color", help="Path to color image (PNG/JPG)")
    parser.add_argument("depth", help="Path to depth image (16-bit PNG)")
    parser.add_argument("--intrinsics", default="intrinsics.json", help="Path to intrinsics JSON")
    args = parser.parse_args()

    fx, fy, cx, cy = load_intrinsics(args.intrinsics)

    img = cv2.imread(args.color)
    depth_u16 = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED).astype(np.uint16)

    assert img is not None, f"Failed to load color image: {args.color}"
    assert depth_u16 is not None, f"Failed to load depth image: {args.depth}"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = Detector(families=FAMILY, nthreads=2, quad_decimate=1.0,
                        quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)
    tags = detector.detect(gray, estimate_tag_pose=False)

    for t in tags:
        corners = t.corners.astype(np.int32)
        cu, cv_ = int(t.center[0]), int(t.center[1])

        cv2.polylines(img, [corners], True, (0, 255, 0), 2)
        cv2.circle(img, (cu, cv_), 4, (0, 0, 255), -1)

        z = median_depth_m(depth_u16, cu, cv_, DEPTH_WINDOW)
        if z is not None:
            x, y, z = backproject(cu, cv_, z, fx, fy, cx, cy)
            label = f"X:{x:+.3f}  Y:{y:+.3f}  Z:{z:.3f} m"
            print(f"Tag {t.tag_id}: X={x:+.4f}  Y={y:+.4f}  Z={z:.4f} m  (pixel: {cu}, {cv_})")
            color = (255, 255, 255)
        else:
            label = "depth invalid"
            print(f"Tag {t.tag_id}: depth invalid at pixel ({cu}, {cv_})")
            color = (0, 0, 255)

        cv2.putText(img, label, (cu + 8, cv_ - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if not tags:
        print("No tags detected.")

    cv2.imshow("AprilTag XYZ", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()