#!/usr/bin/env python3
"""
clean_top.py  --  step 3 of top_view_combine: the TOP shot (robot_rgbd/, taken with the
camera on the UR5e at pose 2) -> a cloud of the green plant pixels only. No radial crop.

The top shot's depth is the raw depth sensor image (640x576, its own intrinsics and
distortion), not aligned to the 1280x720 colour image, so every depth pixel is carried
into the colour camera:
    1. SAM3 mask         every instance of "green plant" (--prompts) on the colour image
    2. depth gradient    drop depth pixels on edges steeper than --grad-max mm/px (flying
                         pixels on leaf silhouettes), depth resolution
    3. back-project      each valid depth pixel through the DEPTH intrinsics + distortion
                         -> X_depth;  X_color = R_dc @ X_depth + t_dc (camera_params.json
                         depth_to_color)
    4. project           X_color through the colour intrinsics + distortion -> colour pixel
    5. visibility        z-buffer of all valid projected points in the colour image; a point more
                         than --occlusion-tol-mm behind the nearest one within
                         --occlusion-window-px is hidden from the colour camera, so its
                         mask / colour lookup would be wrong -> dropped
    6. mask lookup       keep points landing on the SAM3 mask (eroded by --mask-erode px),
                         colour from the colour image, then statistical outlier removal
If the depth image already has the colour image's size it is taken as D2C-aligned and
back-projected through the colour intrinsics instead (steps 3-5 collapse into the
side-shot style occlusion test).

    python clean_top.py ../../../captures/femto_bolt/flora_unit/plant_10-16_deg_5_speed_5
    python clean_top.py <capture> --confidence 0.4 --force

I/O
---
in   <capture>/robot_rgbd/color.png | color_bgr.npy, depth_mm.npy | depth_raw.npy | depth_raw.png,
     <capture>/robot_rgbd/camera_params.json (+ <capture>/camera_params.json for the colour
     intrinsics, depth distortion and depth->colour extrinsic)
out  <capture>/top_view_combine/top/
        mask_top.png        SAM3 "green plant" mask, colour resolution, 255 = plant
        color_top.png       colour, black outside the mask
        depth_top.png       depth image (its own resolution, mm), 0 outside the kept pixels
        cloud_top.ply       cleaned cloud, POSE-2 colour-optical frame (C2), metres
        cleaning_top.json   settings, camera sources, counts
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

SETTING_KEYS = ("prompts", "confidence", "checkpoint", "grad_max", "hole_guard", "occlusion_tol_mm",
                "occlusion_window_px", "mask_erode", "sor_neighbors", "sor_std", "robot_dir")


def depth_to_color_points(depth_mm, cam, keep_d):
    """Depth pixels `keep_d` (bool, depth grid) -> (X in the colour frame (N, 3) metres,
    flat depth-pixel indices)."""
    dw, dh = depth_mm.shape[1], depth_mm.shape[0]
    K = cam["depth"]["K"] if (dw, dh) == cam["depth"]["size"] else geo.scale_K(cam["depth"]["K"], cam["depth"]["size"], (dw, dh))
    ray = geo.ray_grid(K, cam["depth"]["dist"], dw, dh)
    idx = np.flatnonzero(keep_d)
    Xd = ray.reshape(-1, 3)[idx] * (depth_mm.ravel()[idx] / 1000.0)[:, None]
    return Xd @ cam["R_dc"].T + cam["t_dc"], idx


def visible_in_color(Xc, uv, shape, tol_mm, window_px):
    """Per point: inside the colour image and not hidden behind a nearer projected point."""
    h, w = shape
    ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inside = (Xc[:, 2] > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi = np.where(inside, ui, 0), np.where(inside, vi, 0)
    if tol_mm <= 0:
        return inside, ui, vi
    zbuf = np.full((h, w), np.inf, np.float32)
    np.minimum.at(zbuf, (vi[inside], ui[inside]), Xc[inside, 2].astype(np.float32))
    k = 2 * int(round(window_px)) + 1
    znear = cv2.erode(zbuf, np.ones((k, k), np.uint8))
    return inside & (Xc[:, 2] <= znear[vi, ui] + tol_mm / 1000.0), ui, vi


def main():
    ap = argparse.ArgumentParser(description="SAM3 'green plant' cloud of the robot-arm top shot",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--work", default=tv.WORK_DIR, help="work sub-folder of the capture")
    ap.add_argument("--robot-dir", default=tv.ROBOT_DIR, help="sub-folder of the capture holding the top shot")
    ap.add_argument("--color", default=None, help="colour file in --robot-dir (default color.png, else color_bgr.npy)")
    ap.add_argument("--depth", default=None,
                    help="depth file in --robot-dir (default depth_mm.npy, else depth_raw.npy / depth_raw.png)")
    ap.add_argument("--prompts", nargs="+", default=["green plant"], help="SAM3 text prompt(s)")
    ap.add_argument("--confidence", type=float, default=tv.CONFIDENCE, help="SAM3 detection threshold")
    ap.add_argument("--checkpoint", default=tv.CHECKPOINT, help="SAM3 weights")
    ap.add_argument("--grad-max", type=float, default=20.0, help="max depth gradient, mm/px (0 = off)")
    ap.add_argument("--hole-guard", type=int, default=1, help="px of hole filling before the gradient test")
    ap.add_argument("--occlusion-tol-mm", type=float, default=15.0,
                    help="visibility test tolerance in the colour image (0 = off)")
    ap.add_argument("--occlusion-window-px", type=float, default=5.0,
                    help="visibility test half window, colour pixels")
    ap.add_argument("--mask-erode", type=int, default=2, help="px the SAM3 mask is eroded before the lookup")
    ap.add_argument("--sor-neighbors", type=int, default=20, help="statistical outlier removal neighbours")
    ap.add_argument("--sor-std", type=float, default=2.0, help="statistical outlier removal std ratio (0 = off)")
    ap.add_argument("--force", action="store_true", help="clean again even if cleaning_top.json is current")
    args = ap.parse_args()

    capture = Path(args.capture)
    top = tv.work_dir(capture, args.work) / tv.TOP_DIR
    try:
        cam = tv.load_top_camera(capture, args.robot_dir)
        color_bgr, depth_mm, cpath, dpath = tv.read_top_images(capture, args.robot_dir, args.color, args.depth,
                                                               cam["depth_scale_mm"])
    except ds.DatasetError as e:
        sys.exit(str(e))
    settings = {k: getattr(args, k) for k in SETTING_KEYS}
    settings.update(color=str(cpath), depth=str(dpath),
                    images_mtime=[cpath.stat().st_mtime, dpath.stat().st_mtime])

    summary_path = top / tv.TOP_CLEANING
    if summary_path.is_file() and not args.force:
        old = json.loads(summary_path.read_text())
        if old.get("settings") == settings and old.get("ok"):
            print(f"[SKIP] {top} already cleaned with these settings ({old['n_points']:,} points); --force to redo")
            return

    h, w = color_bgr.shape[:2]
    dh, dw = depth_mm.shape[:2]
    aligned = (dh, dw) == (h, w)
    print(f"[TOP ] colour {cpath.name} {w}x{h} | depth {dpath.name} {dw}x{dh} "
          f"({'D2C-aligned' if aligned else 'raw depth sensor -> carried into the colour camera'})")
    print(f"[TOP ] cameras from {cam['sources']}")
    if not aligned and not cam["has_extrinsic"]:
        print("[WARN] no depth_to_color extrinsic found -- treating the depth and colour cameras as one")

    top.mkdir(parents=True, exist_ok=True)
    print(f"[SAM3] {tv.device_name()}")
    t0 = time.time()
    processor = tv.load_model(args.checkpoint, args.confidence)
    print(f"[SAM3] model loaded in {time.time() - t0:.0f} s")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    seg, counts = tv.union_mask(processor, Image.fromarray(color_rgb), args.prompts, (h, w))
    cv2.imwrite(str(top / "mask_top.png"), seg.astype(np.uint8) * 255)
    cv2.imwrite(str(top / "color_top.png"), np.where(seg[..., None], color_bgr, 0).astype(np.uint8))
    rec = dict(finished=datetime.now().isoformat(timespec="seconds"), settings=settings,
               frame="camera_color_optical_frame at robot pose 2 (C2), metres", aligned=aligned,
               camera_sources=cam["sources"], detections=counts, n_seg=int(seg.sum()), ok=False)
    print(f"[SAM3] {counts}, {rec['n_seg']:,} mask pixels")
    if not seg.any():
        summary_path.write_text(json.dumps(dict(rec, reason=f"empty SAM3 mask {counts}"), indent=2))
        sys.exit(f"[FAIL] SAM3 found no {args.prompts} in {cpath}")

    grad = tv.gradient_mask(depth_mm, args.grad_max, args.hole_guard)
    valid = depth_mm > 0
    seg_e = tv.erode(seg, args.mask_erode)
    if aligned:
        K = cam["color"]["K"] if (w, h) == cam["color"]["size"] else geo.scale_K(cam["color"]["K"], cam["color"]["size"], (w, h))
        ray = geo.ray_grid(K, cam["color"]["dist"], w, h)
        z = depth_mm / 1000.0
        cut = seg_e & valid & grad
        keep_img = cut & tv.near_filter(z, cut, args.occlusion_tol_mm, args.occlusion_window_px)
        idx = np.flatnonzero(keep_img)
        P = ray.reshape(-1, 3)[idx] * z.ravel()[idx, None]
        C = color_rgb.reshape(-1, 3)[idx]
        n_vis = int((valid & grad & seg_e).sum())
    else:
        # every valid pixel occludes (also the flying ones the gradient test drops: without
        # them background seen past a leaf edge lands on the leaf's mask pixels unopposed)
        Xc, idx = depth_to_color_points(depth_mm, cam, valid)
        uv = geo.project(Xc, cam["color"]["K"], cam["color"]["dist"])
        vis, ui, vi = visible_in_color(Xc, uv, (h, w), args.occlusion_tol_mm, args.occlusion_window_px)
        vis &= grad.ravel()[idx]
        on_plant = vis & seg_e[vi, ui]
        n_vis = int(vis.sum())
        P, C, idx = Xc[on_plant], color_rgb[vi[on_plant], ui[on_plant]], idx[on_plant]
    depth_keep = np.zeros(depth_mm.size, np.uint16)
    depth_keep[idx] = np.round(depth_mm.ravel()[idx]).astype(np.uint16)
    cv2.imwrite(str(top / "depth_top.png"), depth_keep.reshape(depth_mm.shape))

    n_valid, n_grad, n_bp = int(valid.sum()), int((valid & grad).sum()), len(P)
    if not n_bp:
        summary_path.write_text(json.dumps(dict(rec, reason="no depth lands on the plant mask"), indent=2))
        sys.exit("[FAIL] no depth point lands on the green-plant mask")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.colors = o3d.utility.Vector3dVector(C.astype(np.float64) / 255.0)
    if args.sor_std > 0 and n_bp > args.sor_neighbors:
        pcd, _ = pcd.remove_statistical_outlier(args.sor_neighbors, args.sor_std)
    o3d.io.write_point_cloud(str(top / "cloud_top.ply"), pcd)
    zr = np.percentile(np.asarray(pcd.points)[:, 2] * 1000, [1, 50, 99])
    rec.update(ok=True, n_valid_depth=n_valid, n_grad=n_grad, n_visible=n_vis, n_on_mask=n_bp,
               n_points=len(pcd.points), range_mm_p1_p50_p99=zr.round(1).tolist())
    summary_path.write_text(json.dumps(rec, indent=2))
    print(f"[OK  ] depth {n_valid:,} -grad-> {n_grad:,} -visible in colour-> {n_vis:,} -green plant-> {n_bp:,} "
          f"-sor-> {len(pcd.points):,}  (range {zr[0]:.0f} / {zr[1]:.0f} / {zr[2]:.0f} mm, p1/p50/p99)")
    print(f"[DONE] {time.time() - t0:.0f} s -> {top}")


if __name__ == "__main__":
    main()
