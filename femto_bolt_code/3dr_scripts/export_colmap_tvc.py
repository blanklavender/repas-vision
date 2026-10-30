#!/usr/bin/env python3
"""
export_colmap_tvc.py  --  COLMAP export for a top_view_combine capture (side sweep + robot top shot),
so gsplat's simple_trainer.py can train on it. Same output format as export_colmap.py, whose
helpers it reuses; export_colmap.py itself is unchanged and still serves the older captures.

    python export_colmap_tvc.py ../../captures/femto_bolt/flora_unit/plant_10-16_deg_5_speed_5
    python export_colmap_tvc.py <capture> --views side          # side sweep only

Poses: <capture>/top_view_combine/stitch/poses.json (top_view_combine/stitch_view.py)
    side   shots[i].T_shot_to_ref     camera -> ref (ref = side shot 0's colour camera, metres)
    top    top.T_top_to_ref           camera -> ref (C1_T_C2 from the robot transform file)
Both are inverted to COLMAP's world -> camera. The top shot is the same Femto Bolt colour camera
(1280x720, the capture's colour K + distortion), so all images share one PINHOLE camera after
undistortion.

Masks:
    side   top_view_combine/side/mask/mask_<stem>.png   SAM3 "yellow cylinder" + "green plant"
    top    top_view_combine/top/mask_top.png            SAM3 "green plant" only
The top mask has no pot, while every side image keeps the pot; 3DGS would see the pot as black from
above and yellow from the side. With --top-mask plant+pot (default) the yellow (pot) points of the
side cloud are projected into the top image and their filled outline is added to the top mask, so
pot rim and soil are kept. (Leaf points are not used: their fringe noise would let background in.)

The top image is written as color_000_top.png, so in gsplat's name-sorted order it is image 1 and
stays in the training split (gsplat holds out every --test_every-th image, starting at 0).

I/O contract
------------
in   <capture>: camera_params.json, manifest.csv, color_<stem>.png, robot_rgbd/color.png
     <capture>/top_view_combine/: stitch/{poses.json, side_fused.ply, combined.ply},
                                  side/mask/mask_<stem>.png, top/mask_top.png
out  <capture>/<--out, default colmap_side | colmap_side_top>/
        images/                    undistorted RGB, background black (unless --no-mask)
        sparse/0/*.{txt,bin}       one PINHOLE camera, one pose per image, seed cloud
        export.json                what went in (sources, mask mode, per-image mask coverage)
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from export_colmap import load_color_camera, load_manifest, orthonormalise, read_ply, rot_to_qvec, write_model

TOP_NAME = "color_000_top.png"


def project(P, T_cam_to_ref, K, wh):
    """ref-frame points -> integer pixels of the (undistorted) camera, in-image and in front only."""
    T = np.linalg.inv(T_cam_to_ref)
    Xc = P @ T[:3, :3].T + T[:3, 3]
    z = Xc[:, 2]
    front = z > 1e-3
    u = K[0, 0] * Xc[front, 0] / z[front] + K[0, 2]
    v = K[1, 1] * Xc[front, 1] / z[front] + K[1, 2]
    ok = (u >= 0) & (u < wh[0]) & (v >= 0) & (v < wh[1])
    return u[ok].astype(int), v[ok].astype(int), int(front.sum())


def is_yellow(rgb):
    """yellow pot points: red and green both bright, blue well below them."""
    r, g, b = (rgb[:, k].astype(int) for k in range(3))
    return (r > 120) & (g > 100) & (b < 0.6 * np.minimum(r, g))


def silhouette(P, T_cam_to_ref, K, wh, close_px):
    """filled silhouette of a cloud seen from one camera (holes between points closed, then filled)."""
    u, v, _ = project(P, T_cam_to_ref, K, wh)
    m = np.zeros((wh[1], wh[0]), np.uint8)
    m[v, u] = 255
    if close_px > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2))
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m, cnt, -1, 255, cv2.FILLED)
    return m


def main():
    ap = argparse.ArgumentParser(description="top_view_combine poses -> COLMAP model for gsplat",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--views", choices=["side", "both"], default="both", help="side sweep only, or side + top")
    ap.add_argument("--work", default="top_view_combine", help="top_view_combine work folder of the capture")
    ap.add_argument("--cloud", default=None,
                    help="seed PLY (default stitch/side_fused.ply for side, stitch/combined.ply for both)")
    ap.add_argument("--out", default=None, help="output sub-folder (default colmap_side / colmap_side_top)")
    ap.add_argument("--no-mask", action="store_true", help="keep the background in the images")
    ap.add_argument("--mask-dilate", type=int, default=0, help="px to grow every mask before blacking out")
    ap.add_argument("--top-mask", choices=["plant", "plant+pot"], default="plant+pot",
                    help="top image: SAM3 plant mask only, or plus the side cloud's silhouette (pot + soil)")
    ap.add_argument("--top-close-px", type=int, default=6, help="closing radius for the top silhouette")
    ap.add_argument("--max-points", type=int, default=0, help="random subsample of the seed cloud; 0 = all")
    args = ap.parse_args()

    capture = Path(args.capture)
    work = capture / args.work
    stitch = work / "stitch"
    both = args.views == "both"
    poses_path = stitch / "poses.json"
    cloud_path = Path(args.cloud) if args.cloud else stitch / ("combined.ply" if both else "side_fused.ply")
    out = capture / (args.out or ("colmap_side_top" if both else "colmap_side"))
    top_color = capture / "robot_rgbd" / "color.png"
    need = [capture / "camera_params.json", capture / "manifest.csv", poses_path, cloud_path]
    if both:
        need += [top_color, work / "top" / "mask_top.png", stitch / "side_fused.ply"]
    for p in need:
        if not p.is_file():
            sys.exit(f"missing {p} (run top_view_combine/main_pipeline.py first)")

    K, dist, wh = load_color_camera(capture)
    manifest = {r.get("color"): r for r in load_manifest(capture).values()}
    poses = json.loads(poses_path.read_text())
    print(f"[POSE] {poses_path}  ({len(poses['shots'])} side shots{', + top' if both else ''})")
    maps = cv2.initUndistortRectifyMap(K, dist, None, K, wh, cv2.CV_32FC1)

    def undistort(img, interp):
        return cv2.remap(img, *maps, interp, borderMode=cv2.BORDER_CONSTANT)

    def finish_mask(m):
        if args.mask_dilate > 0:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.mask_dilate + 1,) * 2))
        return m

    img_dir, sparse = out / "images", out / "sparse" / "0"
    img_dir.mkdir(parents=True, exist_ok=True)
    for old in img_dir.glob("*.png"):                     # a re-run with other --views must not leave strays
        old.unlink()

    # (image name, T_cam_to_ref) for every image, plus per-image notes for export.json
    views, notes = [], []
    for s in poses["shots"]:
        stem = s["stem"]
        name = f"color_{stem}.png"
        if name not in manifest:
            sys.exit(f"{name} not in manifest.csv")
        col = cv2.imread(str(capture / name), cv2.IMREAD_COLOR)
        if col is None or (col.shape[1], col.shape[0]) != tuple(wh):
            sys.exit(f"could not read {capture / name} at {wh[0]}x{wh[1]}")
        col = undistort(col, cv2.INTER_LINEAR)
        cover = None
        if not args.no_mask:
            mp = work / "side" / "mask" / f"mask_{stem}.png"
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m is None:
                sys.exit(f"missing {mp} (top_view_combine clean_side step), or pass --no-mask")
            m = finish_mask(undistort(m, cv2.INTER_NEAREST))
            col[m == 0] = 0
            cover = float((m > 0).mean())
        cv2.imwrite(str(img_dir / name), col)
        views.append((name, np.asarray(s["T_shot_to_ref"], float)))
        notes.append(dict(name=name, kind="side", stem=stem, mask_coverage=cover))

    if both:
        T_top = np.asarray(poses["top"]["T_top_to_ref"], float)
        col = cv2.imread(str(top_color), cv2.IMREAD_COLOR)
        if col is None or (col.shape[1], col.shape[0]) != tuple(wh):
            sys.exit(f"could not read {top_color} at {wh[0]}x{wh[1]}")
        col = undistort(col, cv2.INTER_LINEAR)
        cover = None
        if not args.no_mask:
            m = undistort(cv2.imread(str(work / "top" / "mask_top.png"), cv2.IMREAD_GRAYSCALE), cv2.INTER_NEAREST)
            if args.top_mask == "plant+pot":
                side_xyz, side_rgb = read_ply(stitch / "side_fused.ply")
                pot = side_xyz[is_yellow(side_rgb)]
                print(f"[TOP ] pot silhouette from {len(pot)} yellow side points")
                m = np.maximum(m, silhouette(pot, T_top, K, wh, args.top_close_px))
            m = finish_mask(m)
            col[m == 0] = 0
            cover = float((m > 0).mean())
        cv2.imwrite(str(img_dir / TOP_NAME), col)
        views.append((TOP_NAME, T_top))
        notes.append(dict(name=TOP_NAME, kind="top", mask=args.top_mask, mask_coverage=cover,
                          fit_to_side=poses["top"].get("fit_to_side")))

    # COLMAP: world -> camera; image ids in name order, 1-based
    images = []
    for iid, (name, T_wc) in enumerate(sorted(views, key=lambda v: v[0]), start=1):
        R_cw = orthonormalise(T_wc[:3, :3]).T
        images.append((iid, rot_to_qvec(R_cw), -R_cw @ T_wc[:3, 3], name))

    xyz, rgb = read_ply(cloud_path)
    if args.max_points and len(xyz) > args.max_points:
        keep = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    write_model(sparse, K, wh, images, xyz, rgb)

    # sanity: the seed cloud should land in front of each camera and on its non-black pixels
    print(f"[CAM ] PINHOLE {wh[0]}x{wh[1]}  fx {K[0, 0]:.4f} fy {K[1, 1]:.4f} cx {K[0, 2]:.4f} cy {K[1, 2]:.4f}"
          f"  (all images undistorted)")
    check = {}
    for name, T_wc in views:
        u, v, n_front = project(xyz, T_wc, K, wh)
        img = cv2.imread(str(img_dir / name))
        on = float((img[v, u].sum(1) > 0).mean()) if len(u) else 0.0
        check[name] = dict(in_front=n_front / len(xyz), on_mask=on)
    side = [c for n, c in check.items() if n != TOP_NAME]
    print(f"[CHK ] side: cloud in front {min(c['in_front'] for c in side) * 100:.0f}% (min), "
          f"in-view points on non-black pixels {np.median([c['on_mask'] for c in side]) * 100:.0f}% (median), "
          f"{min(c['on_mask'] for c in side) * 100:.0f}% (min)")
    if both:
        c = check[TOP_NAME]
        print(f"[CHK ] top : cloud in front {c['in_front'] * 100:.0f}%, on non-black pixels {c['on_mask'] * 100:.0f}%")
    if min(c["in_front"] for c in check.values()) < 0.9:
        print("[WARN] some cameras look away from the cloud -- check the pose direction")
    for n in notes:
        n["check"] = check[n["name"]]
    (out / "export.json").write_text(json.dumps(dict(
        capture=str(capture), poses=str(poses_path), cloud=str(cloud_path), views=args.views,
        mask=not args.no_mask, mask_dilate=args.mask_dilate, top_mask=args.top_mask if both else None,
        camera=dict(model="PINHOLE", width=wh[0], height=wh[1], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]),
        points=len(xyz), images=notes), indent=2))
    print(f"[OUT ] {out}  ({len(images)} images, {len(xyz)} points; sparse/0 has .txt and .bin)")


if __name__ == "__main__":
    main()
