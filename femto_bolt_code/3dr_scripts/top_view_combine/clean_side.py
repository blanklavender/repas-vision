#!/usr/bin/env python3
"""
clean_side.py  --  step 2 of top_view_combine: clean every SIDE shot of the sweep (SAM3
prompts, radial crop around the turntable axis, depth filters) into one cloud per shot.

Per shot (colour image + D2C-aligned depth from sweep_capture.py, both 1280x720):
    1. SAM3 union mask   every instance of every --prompts phrase ("yellow cylinder",
                         "green plant"), colour resolution
    2. radial crop       keep depth whose 3D point lies within --axis-radius of the turntable
                         axis (registration.json, moved into this shot's frame): a cylinder
                         around the pot that cuts the wall, the floor and anything off the
                         turntable, at any camera tilt
    3. depth gradient    drop pixels on depth edges steeper than --grad-max mm/px (flying
                         pixels on silhouettes)
    4. erode             the SAM3 mask by --mask-erode px before it cuts the depth
    5. occlusion         drop pixels more than --occlusion-tol-mm behind the nearest kept
                         pixel within --occlusion-window-px (depth the colour camera cannot
                         see, shifted onto pot pixels by the depth->colour offset)
    6. back-project      through the colour intrinsics, then statistical outlier removal

    python clean_side.py ../../../captures/femto_bolt/flora_unit/plant_10-16_deg_5_speed_5
    python clean_side.py <capture> --axis-radius 0.10 --force

A work folder already cleaned with the same settings and registration is skipped.

I/O
---
in   <capture>/manifest.csv, color_<stem>.png, depth_<stem>.png, camera_params.json
     <capture>/top_view_combine/registration.json                       register_side.py
out  <capture>/top_view_combine/side/
        mask/mask_<stem>.png          SAM3 union mask, 255 = pot/plant
        rgb/color_<stem>.png          colour, black outside the mask
        depth/depth_<stem>.png        aligned depth, 0 outside the kept pixels
        pointcloud/cloud_<stem>.ply   cleaned cloud, that shot's colour-camera frame, metres
        cleaning.json                 settings, per-shot counts, which shots worked
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

import tv_common as tv
from flora_lib import dataset as ds
from flora_lib import geometry as geo
from flora_lib import registration as reg

SETTING_KEYS = ("prompts", "confidence", "checkpoint", "axis_radius", "grad_max", "hole_guard",
                "occlusion_tol_mm", "occlusion_window_px", "mask_erode", "sor_neighbors", "sor_std")


def backproject(color_rgb, z, mask, ray):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ray[ys, xs] * z[ys, xs, None]))
    pcd.colors = o3d.utility.Vector3dVector(color_rgb[ys, xs].astype(np.float64) / 255.0)
    return pcd


def main():
    ap = argparse.ArgumentParser(description="SAM3 prompt masks + radial crop around the turntable axis, side shots",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--work", default=tv.WORK_DIR, help="work sub-folder of the capture")
    ap.add_argument("--prompts", nargs="+", default=["yellow cylinder", "green plant"], help="SAM3 text prompts (union)")
    ap.add_argument("--confidence", type=float, default=tv.CONFIDENCE, help="SAM3 detection threshold")
    ap.add_argument("--checkpoint", default=tv.CHECKPOINT, help="SAM3 weights")
    ap.add_argument("--axis-radius", type=float, default=0.15,
                    help="keep points within this distance of the turntable axis, metres")
    ap.add_argument("--grad-max", type=float, default=20.0, help="max depth gradient, mm/px (0 = off)")
    ap.add_argument("--hole-guard", type=int, default=1, help="px of hole filling before the gradient test")
    ap.add_argument("--occlusion-tol-mm", type=float, default=40.0, help="occlusion test tolerance (0 = off)")
    ap.add_argument("--occlusion-window-px", type=float, default=24.0, help="occlusion test half window")
    ap.add_argument("--mask-erode", type=int, default=2, help="px the SAM3 mask is eroded before cutting depth")
    ap.add_argument("--sor-neighbors", type=int, default=20, help="statistical outlier removal neighbours")
    ap.add_argument("--sor-std", type=float, default=2.0, help="statistical outlier removal std ratio (0 = off)")
    ap.add_argument("--force", action="store_true", help="clean again even if cleaning.json is current")
    args = ap.parse_args()

    capture = Path(args.capture)
    work = tv.work_dir(capture, args.work)
    side = work / tv.SIDE_DIR
    try:
        rows = ds.read_manifest(capture)
        cam = ds.load_color_camera(capture)
        regn = ds.read_registration(work)
    except ds.DatasetError as e:
        sys.exit(f"{e}  (run register_side.py first)")
    settings = {k: getattr(args, k) for k in SETTING_KEYS}
    settings["registration"] = regn["finished"]
    stems = [r["stem"] for r in rows]

    summary_path = side / ds.CLEANING
    if summary_path.is_file() and not args.force:
        old = json.loads(summary_path.read_text())
        if old.get("settings") == settings and [f["stem"] for f in old.get("frames", [])] == stems:
            n_ok = sum(f["ok"] for f in old["frames"])
            print(f"[SKIP] {side} already cleaned with these settings ({n_ok}/{len(stems)} shots usable); "
                  f"--force to redo")
            return

    for sub in (ds.MASK_DIR, ds.RGB_DIR, ds.DEPTH_DIR, ds.CLOUD_DIR):
        (side / sub).mkdir(parents=True, exist_ok=True)
    print(f"[SAM3] {tv.device_name()}")
    print(f"[RUN ] {len(rows)} side shots | prompts {args.prompts} | intrinsics {cam['source']}")
    tag_r = regn["axis"]["tag_radius_mm"]["median"]
    print(f"[CROP] within {args.axis_radius * 1000:.0f} mm of the turntable axis (tags at {tag_r:.0f} mm) "
          f"| grad < {args.grad_max:g} mm/px | occlusion {args.occlusion_tol_mm:g} mm / "
          f"{args.occlusion_window_px:g} px | sor {args.sor_neighbors}/{args.sor_std:g}")
    t0 = time.time()
    processor = tv.load_model(args.checkpoint, args.confidence)
    print(f"[SAM3] model loaded in {time.time() - t0:.0f} s\n")

    frames, rays = [], {}
    for r in rows:
        s, rec = r["stem"], dict(index=r["index"], stem=r["stem"], true_deg=r["true_deg"], ok=False)
        frames.append(rec)
        color_bgr = cv2.imread(str(capture / r["color"]), cv2.IMREAD_COLOR)
        dpath = capture / r["depth"] if r["depth"] else None
        if color_bgr is None or dpath is None or not dpath.is_file():
            rec["reason"] = "missing colour or depth image"
            print(f"[SKIP] {s}: {rec['reason']}")
            continue
        depth = np.load(dpath) if dpath.suffix == ".npy" else cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        h, w = color_bgr.shape[:2]
        if depth is None or depth.shape[:2] != (h, w):
            rec["reason"] = f"depth {None if depth is None else depth.shape[:2]} != colour {(h, w)}"
            print(f"[SKIP] {s}: {rec['reason']}")
            continue
        shot = regn["by_stem"].get(s)
        if shot is None:
            rec["reason"] = "no registered pose (re-run register_side.py)"
            print(f"[SKIP] {s}: {rec['reason']}")
            continue
        if (w, h) not in rays:
            K = cam["K"] if (w, h) == cam["size"] else geo.scale_K(cam["K"], cam["size"], (w, h))
            rays[(w, h)] = geo.ray_grid(K, cam["dist"], w, h)
        ray = rays[(w, h)]
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        seg, counts = tv.union_mask(processor, Image.fromarray(color_rgb), args.prompts, (h, w))
        rec["detections"] = counts
        cv2.imwrite(str(ds.mask_path(side, s)), seg.astype(np.uint8) * 255)
        cv2.imwrite(str(ds.clean_rgb_path(side, s)), np.where(seg[..., None], color_bgr, 0).astype(np.uint8))
        if not seg.any():
            rec["reason"] = f"empty SAM3 mask {counts}"
            print(f"[SKIP] {s}: {rec['reason']}")
            continue

        depth_mm = depth.astype(np.float32) * cam["depth_scale_mm"]
        z = depth_mm / 1000.0
        axis_s, point_s = reg.axis_in_shot(shot["T"], regn["axis"]["axis"], regn["axis"]["point"])
        near = (z > 0) & (reg.radial_distance(ray * z[..., None], axis_s, point_s) < args.axis_radius)
        grad = tv.gradient_mask(depth_mm, args.grad_max, args.hole_guard)
        cut = tv.erode(seg, args.mask_erode) & near & grad
        occl = tv.near_filter(z, cut, args.occlusion_tol_mm, args.occlusion_window_px)
        keep = cut & occl
        cv2.imwrite(str(ds.clean_depth_path(side, s)), np.where(keep, depth, 0).astype(depth.dtype))

        pcd = backproject(color_rgb, z, keep, ray)
        if pcd is None:
            rec["reason"] = "nothing survives the radial crop and depth filters"
            print(f"[SKIP] {s}: {rec['reason']} {counts}")
            continue
        n_bp = len(pcd.points)
        if args.sor_std > 0 and n_bp > args.sor_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(args.sor_neighbors, args.sor_std)
        o3d.io.write_point_cloud(str(ds.clean_cloud_path(side, s)), pcd)
        rec.update(ok=True, n_seg=int(seg.sum()), n_axis=int((seg & near).sum()),
                   n_grad=int((seg & near & grad).sum()), n_occluded=int((cut & ~occl).sum()),
                   n_points=len(pcd.points))
        print(f"[OK  ] {s}  seg {rec['n_seg']:,} -radial-> {rec['n_axis']:,} -grad-> {rec['n_grad']:,} "
              f"-erode/occl-> {n_bp:,} -sor-> {len(pcd.points):,}  {counts}")

    n_ok = sum(f["ok"] for f in frames)
    summary_path.write_text(json.dumps(dict(
        finished=datetime.now().isoformat(timespec="seconds"), settings=settings,
        intrinsics=cam["source"], frames=frames), indent=2))
    print(f"\n[DONE] {n_ok}/{len(frames)} side shots cleaned in {time.time() - t0:.0f} s -> {side}")
    if n_ok < 2:
        sys.exit("fewer than 2 usable shots -- check the prompts / --axis-radius")


if __name__ == "__main__":
    main()
