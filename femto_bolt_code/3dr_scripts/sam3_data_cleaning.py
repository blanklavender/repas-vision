#!/usr/bin/env python3
"""
sam3_data_cleaning.py  --  clean every shot of a capture: SAM3 text prompts, then a crop
to a cylinder around the turntable axis and depth filters, then a per-shot point cloud.

Per shot (colour image + D2C-aligned depth from sweep_capture.py):
    1. SAM3 union mask     every instance of every --prompts phrase ("yellow cylinder",
                           "green plant"), colour resolution
    2. axis crop           keep depth whose 3D point lies within --axis-radius of the
                           turntable axis (tag_register.py, moved into this shot's frame):
                           cuts the wall, the floor and anything off the turntable at any
                           camera tilt, with no fixed x/y/z limits
    3. depth gradient      drop pixels on depth edges steeper than --grad-max mm/px
                           (flying pixels on silhouettes)
    4. occlusion           drop pixels more than --occlusion-tol-mm behind the nearest
                           masked pixel within --occlusion-window-px: depth the colour
                           camera cannot see (wall seen past the pot edge, shifted onto
                           pot pixels by the depth->colour camera offset)
    5. erode               the SAM3 mask by --mask-erode px before it cuts the depth
    6. back-project        through the colour intrinsics (camera_params.json), then a
                           statistical outlier removal (--sor-neighbors / --sor-std)

    python sam3_data_cleaning.py ../../captures/femto_bolt/flora_unit/yellow_pot_10-06_deg_10_speed_5
    python sam3_data_cleaning.py <capture> --prompts "yellow cylinder" "green plant" --axis-radius 0.07
    python sam3_data_cleaning.py <capture> --force          # redo although cleaning.json is current

A capture already cleaned with the same settings and registration is skipped (no model
load) unless --force.

I/O contract (layout: flora_pipeline/flora_lib/dataset.py)
------------
in   <capture>/manifest.csv, color_<stem>.png, depth_<stem>.png   sweep_capture.py
     <capture>/camera_params.json   colour intrinsics + depth scale (or --intrinsics-ini)
     <capture>/registration.json    shot poses + turntable axis      tag_register.py
     --checkpoint                   SAM3 weights (default <repo>/sam3_weights/sam3.pt)
out  <capture>/mask/mask_<stem>.png        SAM3 union mask, 255 = pot/plant
     <capture>/rgb/color_<stem>.png        colour, black outside the mask
     <capture>/depth/depth_<stem>.png      aligned depth, 0 outside the cleaned mask
     <capture>/pointcloud/cloud_<stem>.ply cleaned cloud, colour-camera frame, metres
     <capture>/cleaning.json               settings, per-shot counts, which shots worked
"""

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "flora_pipeline"))
from flora_lib import dataset as ds  # noqa: E402
from flora_lib import geometry as geo  # noqa: E402
from flora_lib import registration as reg  # noqa: E402

# defaults (sam3_batch_masks.py reads these too)
PROMPT_A = "yellow cylinder"
PROMPT_B = "green plant"
CHECKPOINT = str(Path(__file__).resolve().parents[2] / "sam3_weights" / "sam3.pt")
CONFIDENCE = 0.5


def __getattr__(name):
    """torch is imported on first use (it takes ~15 s), so an up-to-date capture is skipped
    quickly; sam3_batch_masks.py still reaches it as sam3_data_cleaning.torch."""
    if name == "torch":
        import torch
        return torch
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# SAM3
# ---------------------------------------------------------------------------
def load_model(checkpoint=None, confidence=None):
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint or CHECKPOINT,
                                   load_from_HF=False)
    return Sam3Processor(model, confidence_threshold=CONFIDENCE if confidence is None else confidence)


def autocast_ctx():
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.autocast("cuda", dtype=(torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16))
    return nullcontext()


def union_mask(processor, image, prompts, shape):
    """Union of every detected instance of every prompt -> (bool mask, detections per prompt)."""
    h, w = shape
    combined = np.zeros((h, w), dtype=bool)
    counts = {}
    with autocast_ctx():
        state = processor.set_image(image)
        for prompt in prompts:
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=prompt)
            for i in range(len(state["scores"])):
                m = state["masks"][i].squeeze(0).cpu().numpy()
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                combined |= m.astype(bool)
            counts[prompt] = len(state["scores"])
    return combined, counts


# ---------------------------------------------------------------------------
# depth thresholding
# ---------------------------------------------------------------------------
def erode(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def axis_mask(z, ray, axis, point, radius):
    """Valid depth whose 3D point (this shot's colour-camera frame, metres) lies within
    `radius` of the turntable axis line (axis, point), given in the same frame."""
    return (z > 0) & (reg.radial_distance(ray * z[..., None], axis, point) < radius)


def gradient_mask(depth_mm, args):
    """False on depth edges steeper than --grad-max mm per pixel (Sobel / 8)."""
    if args.grad_max <= 0:
        return np.ones(depth_mm.shape, bool)
    d = depth_mm.astype(np.float32)
    valid = d > 0
    if args.hole_guard > 0:            # holes would make every neighbour an "edge"
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.hole_guard + 1,) * 2)
        d = np.where(valid, d, cv2.dilate(d, k))
    g = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    return g < args.grad_max


def occlusion_mask(z, keep, args):
    """False where z lies more than --occlusion-tol-mm behind the nearest kept pixel within
    --occlusion-window-px (min filter over the kept pixels only, so the turntable disk or
    other unmasked foreground never counts as an occluder)."""
    if args.occlusion_tol_mm <= 0:
        return np.ones(z.shape, bool)
    zk = np.where(keep, z, np.inf).astype(np.float32)
    k = 2 * int(round(args.occlusion_window_px)) + 1
    znear = cv2.erode(zk, np.ones((k, k), np.uint8))
    return z <= znear + args.occlusion_tol_mm / 1000.0


def backproject(color_rgb, z, mask, ray):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ray[ys, xs] * z[ys, xs, None]))
    pcd.colors = o3d.utility.Vector3dVector(color_rgb[ys, xs].astype(np.float64) / 255.0)
    return pcd


# ---------------------------------------------------------------------------
SETTING_KEYS = ("prompts", "confidence", "checkpoint", "axis_radius", "grad_max", "hole_guard", "occlusion_tol_mm", "occlusion_window_px", "mask_erode",
                "sor_neighbors", "sor_std", "intrinsics_ini")


def main():
    ap = argparse.ArgumentParser(description="SAM3 prompt masks + turntable-axis crop for every shot",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--prompts", nargs="+", default=[PROMPT_A, PROMPT_B], help="SAM3 text prompts (union)")
    ap.add_argument("--confidence", type=float, default=CONFIDENCE, help="SAM3 detection threshold")
    ap.add_argument("--checkpoint", default=CHECKPOINT, help="SAM3 weights")
    ap.add_argument("--axis-radius", type=float, default=0.06,
                    help="keep points within this distance of the turntable axis, metres")
    ap.add_argument("--grad-max", type=float, default=20.0, help="max depth gradient, mm/px (0 = off)")
    ap.add_argument("--hole-guard", type=int, default=1, help="px of hole filling before the gradient test")
    ap.add_argument("--occlusion-tol-mm", type=float, default=40.0, help="occlusion test tolerance (0 = off)")
    ap.add_argument("--occlusion-window-px", type=float, default=24.0, help="occlusion test half window")
    ap.add_argument("--mask-erode", type=int, default=2, help="px the SAM3 mask is eroded before cutting depth")
    ap.add_argument("--sor-neighbors", type=int, default=20, help="statistical outlier removal neighbours")
    ap.add_argument("--sor-std", type=float, default=2.0, help="statistical outlier removal std ratio (0 = off)")
    ap.add_argument("--intrinsics-ini", default=None,
                    help="Orbbec CameraParam .ini, only for captures without camera_params.json")
    ap.add_argument("--force", action="store_true", help="clean again even if cleaning.json is current")
    args = ap.parse_args()

    capture = Path(args.capture)
    try:
        rows = ds.read_manifest(capture)
        cam = ds.load_color_camera(capture, args.intrinsics_ini)
        regn = ds.read_registration(capture)
    except ds.DatasetError as e:
        sys.exit(str(e))
    settings = {k: getattr(args, k) for k in SETTING_KEYS}
    settings["registration"] = regn["finished"]
    stems = [r["stem"] for r in rows]

    summary_path = capture / ds.CLEANING
    if summary_path.is_file() and not args.force:
        old = json.loads(summary_path.read_text())
        if old.get("settings") == settings and [f["stem"] for f in old.get("frames", [])] == stems:
            n_ok = sum(f["ok"] for f in old["frames"])
            print(f"[SKIP] {capture} already cleaned with these settings ({n_ok}/{len(stems)} shots usable); "
                  f"--force to redo")
            return

    import torch
    for sub in (ds.MASK_DIR, ds.RGB_DIR, ds.DEPTH_DIR, ds.CLOUD_DIR):
        (capture / sub).mkdir(exist_ok=True)
    print(f"[SAM3] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (slow)'}")
    print(f"[RUN ] {len(rows)} shots | prompts {args.prompts} | intrinsics {cam['source']}")
    tag_r = regn["axis"]["tag_radius_mm"]["median"]
    print(f"[CROP] within {args.axis_radius * 1000:.0f} mm of the turntable axis (tags at {tag_r:.0f} mm) "
          f"| grad < {args.grad_max:g} mm/px | occlusion "
          f"{args.occlusion_tol_mm:g} mm / {args.occlusion_window_px:g} px | sor {args.sor_neighbors}/{args.sor_std:g}")
    t0 = time.time()
    processor = load_model(args.checkpoint, args.confidence)
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
            rec["reason"] = "no registered pose (re-run tag_register.py)"
            print(f"[SKIP] {s}: {rec['reason']}")
            continue
        if shot.get("rejected"):
            rec.update(rejected=True, reason=f"rejected by tag_register.py: rotation {shot['turn_error_deg']:.1f} deg "
                                             f"off its logged turn")
            print(f"[SKIP] {s}: {rec['reason']}")
            continue
        if (w, h) not in rays:
            K = cam["K"] if (w, h) == cam["size"] else geo.scale_K(cam["K"], cam["size"], (w, h))
            rays[(w, h)] = geo.ray_grid(K, cam["dist"], w, h)
        ray = rays[(w, h)]
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        seg, counts = union_mask(processor, Image.fromarray(color_rgb), args.prompts, (h, w))
        rec["detections"] = counts
        cv2.imwrite(str(ds.mask_path(capture, s)), seg.astype(np.uint8) * 255)
        cv2.imwrite(str(ds.clean_rgb_path(capture, s)), np.where(seg[..., None], color_bgr, 0).astype(np.uint8))
        if not seg.any():
            rec["reason"] = f"empty SAM3 mask {counts}"
            print(f"[SKIP] {s}: {rec['reason']}")
            continue

        depth_mm = depth.astype(np.float32) * cam["depth_scale_mm"]
        z = depth_mm / 1000.0
        near = axis_mask(z, ray, *reg.axis_in_shot(shot["T"], regn["axis"]["axis"], regn["axis"]["point"]),
                         args.axis_radius)
        grad = gradient_mask(depth_mm, args)
        cut = erode(seg, args.mask_erode) & near & grad
        occl = occlusion_mask(z, cut, args)
        keep = cut & occl
        cv2.imwrite(str(ds.clean_depth_path(capture, s)), np.where(keep, depth, 0).astype(depth.dtype))

        pcd = backproject(color_rgb, z, keep, ray)
        if pcd is None:
            rec["reason"] = "nothing survives the axis crop and depth filters"
            print(f"[SKIP] {s}: {rec['reason']} {counts}")
            continue
        n_bp = len(pcd.points)
        if args.sor_std > 0 and n_bp > args.sor_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(args.sor_neighbors, args.sor_std)
        o3d.io.write_point_cloud(str(ds.clean_cloud_path(capture, s)), pcd)
        rec.update(ok=True, n_seg=int(seg.sum()), n_axis=int((seg & near).sum()), n_grad=int((seg & near & grad).sum()),
                   n_occluded=int((cut & ~occl).sum()), n_points=len(pcd.points))
        print(f"[OK  ] {s}  seg {rec['n_seg']:,} -axis-> {rec['n_axis']:,} -grad-> {rec['n_grad']:,} "
              f"-erode/occl-> {n_bp:,} -sor-> {len(pcd.points):,}  {counts}")

    n_ok = sum(f["ok"] for f in frames)
    n_rej = sum(f.get("rejected", False) for f in frames)
    summary_path.write_text(json.dumps(dict(
        finished=datetime.now().isoformat(timespec="seconds"), settings=settings,
        intrinsics=cam["source"], frames=frames), indent=2))
    print(f"\n[DONE] {n_ok}/{len(frames)} shots cleaned"
          f"{f' ({n_rej} rejected by the registration)' if n_rej else ''} in {time.time() - t0:.0f} s -> "
          f"{capture}/{{{ds.MASK_DIR},{ds.RGB_DIR},{ds.DEPTH_DIR},{ds.CLOUD_DIR}}}, {ds.CLEANING}")
    if n_ok < 2:
        sys.exit("fewer than 2 usable shots -- check the prompts / --axis-radius")


if __name__ == "__main__":
    main()
