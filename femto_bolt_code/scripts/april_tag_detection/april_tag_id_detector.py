#!/usr/bin/env python3
import argparse
import cv2
from pupil_apriltags import Detector

def main():
    parser = argparse.ArgumentParser(description="Detect AprilTag IDs in an image.")
    parser.add_argument("image", help="Path to input image (RGB or grayscale).")
    parser.add_argument("--family", default="tag36h11",
                        help="AprilTag family (e.g., tag36h11, tag25h9, tag16h5).")
    parser.add_argument("--nthreads", type=int, default=2)
    parser.add_argument("--quad-decimate", type=float, default=1.0,
                        help=">1 for speed, <1 for accuracy.")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    det = Detector(
        families=args.family,
        nthreads=args.nthreads,
        quad_decimate=args.quad_decimate,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25
    )

    tags = det.detect(img, estimate_tag_pose=False)
    if not tags:
        print("No tags detected.")
        return

    print(f"Detected {len(tags)} tag(s) from family '{args.family}':")
    for t in tags:
        tid = getattr(t, "tag_id", getattr(t, "id", None))
        cx, cy = map(float, t.center)  # pixel center
        print(f"  - id={tid:>4}  center=({cx:.1f}, {cy:.1f})")

if __name__ == "__main__":
    main()

# example run command: python april_tag_id_detector.py "./captures_colorworld/capture_20250917_164436/color_20250917_164436.png" --family tag36h11