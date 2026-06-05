#!/usr/bin/env python3
"""
Click any point on a color image to get its 3D (X, Y, Z) position.

Usage:
    python click_xyz.py color.png depth.png
    python click_xyz.py color.png depth.png --intrinsics intrinsics.json
"""

import argparse, json
import cv2
import numpy as np

DEPTH_WINDOW = 5
MIN_Z, MAX_Z = 0.25, 8.0
DEPTH_SCALE  = 0.001


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
    parser = argparse.ArgumentParser(description="Click to get XYZ from color + depth")
    parser.add_argument("color", help="Path to color image (PNG/JPG)")
    parser.add_argument("depth", help="Path to depth image (16-bit PNG)")
    parser.add_argument("--intrinsics", default="intrinsics.json", help="Path to intrinsics JSON")
    args = parser.parse_args()

    fx, fy, cx, cy = load_intrinsics(args.intrinsics)
    base_img = cv2.imread(args.color)
    depth_u16 = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED).astype(np.uint16)
    assert base_img is not None, f"Failed to load: {args.color}"
    assert depth_u16 is not None, f"Failed to load: {args.depth}"

    img = base_img.copy()
    points = []

    def on_click(event, x, y, flags, param):
        nonlocal img
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        img = base_img.copy()

        # draw all previous points
        for p in points:
            cv2.circle(img, (p["u"], p["v"]), 4, (0, 255, 0), -1)
            cv2.putText(img, p["label"], (p["u"] + 8, p["v"] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # new point
        z = median_depth_m(depth_u16, x, y, DEPTH_WINDOW)
        if z is not None:
            px, py, pz = backproject(x, y, z, fx, fy, cx, cy)
            label = f"X:{px:+.3f} Y:{py:+.3f} Z:{pz:.3f}m"
            print(f"Pixel ({x}, {y}) -> X={px:+.4f}  Y={py:+.4f}  Z={pz:.4f} m")
        else:
            label = "depth invalid"
            print(f"Pixel ({x}, {y}) -> depth invalid")

        points.append({"u": x, "v": y, "label": label})
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(img, label, (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Click for XYZ (q to quit)", img)

    cv2.imshow("Click for XYZ (q to quit)", img)
    cv2.setMouseCallback("Click for XYZ (q to quit)", on_click)

    print("Click anywhere on the image. Press 'q' to quit.")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord('q'), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()