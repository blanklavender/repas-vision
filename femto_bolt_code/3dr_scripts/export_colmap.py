#!/usr/bin/env python3
"""
export_colmap.py  --  turn a tag_graph_register.py solution into a COLMAP sparse model that
gsplat's simple_trainer.py can train on (poses + intrinsics + seed cloud, no SfM needed).

    python export_colmap.py ../../captures/femto_bolt/flora_unit/yellow_pot_09-30_deg_10_speed_5

Poses come from <capture>/tag_graph/chain/poses.json (the chain solution) by default:
    X_ref = T_shot_to_ref @ X_shot      (camera -> world, world = shot 0's colour camera, metres)
COLMAP stores world -> camera, so each pose is inverted:  T_cw = inv(T_shot_to_ref).

The colour camera has lens distortion, but the PINHOLE model has none. Every image (and its
SAM3 mask) is therefore undistorted with the original K kept, so the PINHOLE intrinsics are
exactly the calibrated fx fy cx cy from camera_params.json.

I/O contract
------------
in   <capture>: camera_params.json, manifest.csv, color_*.png, stops/*/mask.png
     --poses   poses.json (tag_stitch / tag_graph_register format), default tag_graph/chain/poses.json
     --cloud   seed cloud in the same reference frame, default fused.ply next to --poses
out  <capture>/<--out, default colmap_chain>/
        images/<color name>.png    undistorted RGB, background black (unless --no-mask)
        sparse/0/cameras.{txt,bin} one PINHOLE camera
        sparse/0/images.{txt,bin}  one world->camera pose per shot
        sparse/0/points3D.{txt,bin} seed cloud, ERROR 0, empty tracks
Both text and binary files are written; gsplat's loader reads the .bin ones when present.
"""

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
def load_color_camera(capture):
    c = json.loads((capture / "camera_params.json").read_text())
    i = c["color_intrinsic"]
    K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], float)
    return K, np.array(c["color_distortion"]["opencv_order"], float), (i["width"], i["height"])


def load_manifest(capture):
    with open(capture / "manifest.csv", newline="") as f:
        return {int(r["index"]): r for r in csv.DictReader(f)}


PLY_TYPES = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4",
             "uint": "u4", "float": "f4", "double": "f8"}


def read_ply(path):
    """binary little-endian vertex-only PLY (what open3d writes) -> xyz (N,3), rgb uint8 (N,3)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            sys.exit(f"{path}: not a PLY file")
        n, props, fmt = 0, [], None
        while True:
            line = f.readline().decode("ascii").split()
            if not line:
                continue
            if line[0] == "format":
                fmt = line[1]
            elif line[0] == "element":
                if line[1] != "vertex":
                    sys.exit(f"{path}: only vertex-only PLYs are supported")
                n = int(line[2])
            elif line[0] == "property":
                props.append((line[2], "<" + PLY_TYPES[line[1]]))
            elif line[0] == "end_header":
                break
        if fmt != "binary_little_endian":
            sys.exit(f"{path}: expected binary_little_endian, got {fmt}")
        v = np.frombuffer(f.read(n * np.dtype(props).itemsize), dtype=props, count=n)
    xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    if "red" in v.dtype.names:
        rgb = np.column_stack([v["red"], v["green"], v["blue"]])
        rgb = rgb if rgb.dtype == np.uint8 else np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.full((n, 3), 128, np.uint8)
    return xyz, rgb


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def rot_to_qvec(R):
    """rotation matrix -> unit quaternion (w, x, y, z), w >= 0 (COLMAP convention)."""
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q) / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def orthonormalise(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


# ---------------------------------------------------------------------------
# writers (COLMAP text + binary)
# ---------------------------------------------------------------------------
def write_model(sparse, K, wh, images, xyz, rgb):
    """images: list of (image_id, qvec, tvec, name)."""
    sparse.mkdir(parents=True, exist_ok=True)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w, h = wh

    (sparse / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n# Number of cameras: 1\n"
        f"1 PINHOLE {w} {h} {fx:.10g} {fy:.10g} {cx:.10g} {cy:.10g}\n")
    with open(sparse / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, w, h))          # id, model 1 = PINHOLE
        f.write(struct.pack("<4d", fx, fy, cx, cy))

    with open(sparse / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n"
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
                f"# Number of images: {len(images)}, mean observations per image: 0\n")
        for iid, q, t, name in images:
            f.write(f"{iid} {' '.join(f'{v:.12g}' for v in q)} {' '.join(f'{v:.12g}' for v in t)} 1 {name}\n\n")
    with open(sparse / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for iid, q, t, name in images:
            f.write(struct.pack("<i4d3di", iid, *q, *t, 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))                   # no 2D observations

    with open(sparse / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n"
                "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
                f"# Number of points: {len(xyz)}, mean track length: 0\n")
        f.writelines(f"{k + 1} {p[0]:.9g} {p[1]:.9g} {p[2]:.9g} {c[0]} {c[1]} {c[2]} 0\n"
                     for k, (p, c) in enumerate(zip(xyz, rgb)))
    rec = np.zeros(len(xyz), dtype=[("id", "<u8"), ("xyz", "<f8", 3), ("rgb", "u1", 3),
                                     ("err", "<f8"), ("track_len", "<u8")])
    rec["id"] = np.arange(1, len(xyz) + 1)
    rec["xyz"], rec["rgb"] = xyz, rgb
    with open(sparse / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(xyz)))
        f.write(rec.tobytes())


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="tag_graph_register poses -> COLMAP model for gsplat",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--poses", default=None, help="poses.json (default <capture>/tag_graph/chain/poses.json)")
    ap.add_argument("--cloud", default=None, help="seed PLY (default fused.ply next to --poses)")
    ap.add_argument("--out", default="colmap_chain", help="output sub-folder of the capture")
    ap.add_argument("--no-mask", action="store_true", help="keep the background in the images")
    ap.add_argument("--mask-dilate", type=int, default=0, help="px to grow the SAM3 mask before blacking out")
    ap.add_argument("--no-undistort", action="store_true",
                    help="copy images as-is (PINHOLE will then ignore the lens distortion)")
    ap.add_argument("--max-points", type=int, default=0, help="random subsample of the seed cloud; 0 = all")
    args = ap.parse_args()

    capture = Path(args.capture)
    poses_path = Path(args.poses) if args.poses else capture / "tag_graph" / "chain" / "poses.json"
    cloud_path = Path(args.cloud) if args.cloud else poses_path.parent / "fused.ply"
    for p in (capture / "camera_params.json", capture / "manifest.csv", poses_path, cloud_path):
        if not p.is_file():
            sys.exit(f"missing {p}")

    K, dist, wh = load_color_camera(capture)
    manifest = load_manifest(capture)
    poses = json.loads(poses_path.read_text())
    print(f"[POSE] {poses_path}  ({poses.get('method', '?')}, {len(poses['shots'])} shots)")
    if "T_shot_to_ref" not in poses["shots"][0]:
        sys.exit("poses.json: expected 'T_shot_to_ref' per shot (camera -> world)")

    out = capture / args.out
    img_dir, sparse = out / "images", out / "sparse" / "0"
    img_dir.mkdir(parents=True, exist_ok=True)
    if args.no_undistort:
        maps = None
    else:
        maps = cv2.initUndistortRectifyMap(K, dist, None, K, wh, cv2.CV_32FC1)

    images, centres = [], []
    for s in poses["shots"]:
        idx = s["index"]
        if idx not in manifest:
            sys.exit(f"shot {idx} not in manifest.csv")
        row = manifest[idx]
        col = cv2.imread(str(capture / row["color"]), cv2.IMREAD_COLOR)
        if col is None:
            sys.exit(f"could not read {capture / row['color']}")
        if (col.shape[1], col.shape[0]) != tuple(wh):
            sys.exit(f"{row['color']}: size {col.shape[1]}x{col.shape[0]} != intrinsics {wh}")
        if maps is not None:
            col = cv2.remap(col, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        if not args.no_mask:
            m = cv2.imread(str(capture / row["stop_dir"] / "mask.png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                sys.exit(f"missing {capture / row['stop_dir'] / 'mask.png'} (run sam3_batch_masks.py, "
                         f"or pass --no-mask)")
            if maps is not None:
                m = cv2.remap(m, *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            if args.mask_dilate > 0:
                m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.mask_dilate + 1,) * 2))
            col[m == 0] = 0
        name = Path(row["color"]).with_suffix(".png").name
        cv2.imwrite(str(img_dir / name), col)

        T_wc = np.asarray(s["T_shot_to_ref"], float)              # camera -> world
        R_cw = orthonormalise(T_wc[:3, :3]).T                      # world -> camera
        t_cw = -R_cw @ T_wc[:3, 3]
        images.append((idx + 1, rot_to_qvec(R_cw), t_cw, name))
        centres.append(T_wc[:3, 3])

    xyz, rgb = read_ply(cloud_path)
    if args.max_points and len(xyz) > args.max_points:
        keep = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    write_model(sparse, K, wh, images, xyz, rgb)

    # sanity: the cloud should sit in front of every camera
    front = [float(np.mean(((xyz - c) @ np.asarray(T)[:3, 2]) > 0))
             for c, T in zip(centres, (s["T_shot_to_ref"] for s in poses["shots"]))]
    centres = np.array(centres)
    r = np.linalg.norm(centres - centres.mean(0), axis=1)
    print(f"[CAM ] PINHOLE {wh[0]}x{wh[1]}  fx {K[0, 0]:.4f} fy {K[1, 1]:.4f} cx {K[0, 2]:.4f} cy {K[1, 2]:.4f}"
          f"  ({'undistorted' if maps is not None else 'distortion IGNORED'})")
    print(f"[CHK ] camera ring radius {r.mean() * 1000:.1f} mm (+-{r.std() * 1000:.1f}); cloud in front of "
          f"camera: min {min(front) * 100:.0f}% / median {np.median(front) * 100:.0f}% of points")
    if min(front) < 0.9:
        print("[WARN] some cameras look away from the cloud -- check the pose direction")
    print(f"[OUT ] {out}  ({len(images)} images, {len(xyz)} points; sparse/0 has .txt and .bin)")


if __name__ == "__main__":
    main()
