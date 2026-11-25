#!/usr/bin/env python3
"""
sam3_leaf_masks.py  --  per-leaf SAM3 instance masks for every turntable shot.

Runs SAM3 with a text prompt ("leaf" by default) on each raw colour image of a capture and
keeps every detection as its own instance (unlike sam3_batch_masks.py, which unions them).
These masks are the per-shot input to leaf instance tracking on the 2DGS surface.

    python sam3_leaf_masks.py ../../captures/femto_bolt/plant_10-16_deg_5_speed_5 \
        --cameras ../../captures/femto_bolt/plant_10-16_clean55/results/2dgs_7k/maps/cameras.json
    python sam3_leaf_masks.py <capture> --prompts leaf "plant leaf" --confidence 0.6
    python sam3_leaf_masks.py <capture> --indices 0-5 --overwrite

Image frame
-----------
The raw colour images have lens distortion; the 2DGS cameras (cameras.json) are PINHOLE with
the calibrated K, because export_colmap.py undistorted every image with K kept. This script
does the same remap BEFORE running SAM3, so a mask pixel (u, v) lines up directly with
K @ X_cam from cameras.json. --no-undistort keeps the raw (distorted) frame instead.

I/O contract
------------
in   <capture>/manifest.csv          columns index, color   (sweep_capture.py)
     <capture>/<color>               colour PNG per shot, 1280x720
     <capture>/camera_params.json    color_intrinsic + color_distortion (opencv_order)
     <capture>/mask/mask_<stem>.png  optional binary plant mask (raw frame); instances mostly
                                     outside it are dropped (--plant-mask-dir, --min-in-plant)
     --cameras cameras.json          optional; only shots whose stem it lists are processed
                                     (the 2DGS run dropped shots with bad poses)
out  <out>/<stem>.npz                masks  (N,H,W) bool, one per kept instance, may overlap
                                     scores (N,) float32, sorted high -> low
                                     prompt (N,) int16, index into the run's prompt list
                                     bbox   (N,4) int32  x0 y0 x1 y1 (inclusive)
                                     labels (H,W) uint16  0 = none, i+1 = masks[i]; on overlap
                                            the higher-scoring instance wins
     <out>/<stem>_overlay.jpg        undistorted colour with instances tinted + numbered
     <out>/leaf_masks.json           settings, K, per-shot instance counts
<stem> is the colour name without "color_" and extension, e.g. 000_ang000 -- the same stems
as the 2DGS maps/<stem>.npz.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import sam3_data_cleaning as sdc                       # load_model / autocast_ctx / CHECKPOINT
from export_colmap import load_color_camera, load_manifest
from sam3_batch_masks import parse_indices

PALETTE = np.array([(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
                    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
                    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
                    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)],
                   np.uint8)[:, ::-1]                  # RGB -> BGR for cv2


def stem_of(color_name):
    s = Path(color_name).stem
    return s[len("color_"):] if s.startswith("color_") else s


def detect(processor, image_rgb, prompts):
    """Every instance of every prompt -> list of (mask bool HxW, score, prompt index)."""
    h, w = image_rgb.shape[:2]
    out = []
    with sdc.autocast_ctx():
        state = processor.set_image(Image.fromarray(image_rgb))
        for p, prompt in enumerate(prompts):
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=prompt)
            for i in range(len(state["scores"])):
                m = state["masks"][i].squeeze(0).float().cpu().numpy() > 0.5
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                s = state["scores"][i]
                out.append((m, float(s.item() if hasattr(s, "item") else s), p))
    return out


def filter_instances(dets, plant, min_area, min_in_plant, dedup_iou):
    """Drop tiny / off-plant instances and near-duplicates (same leaf found by two prompts or
    twice by one), keeping the higher score. Returns the survivors sorted by score, and the
    number dropped per reason."""
    dropped = dict(small=0, off_plant=0, duplicate=0)
    kept = []
    for m, s, p in sorted(dets, key=lambda d: -d[1]):
        area = int(m.sum())
        if area < min_area:
            dropped["small"] += 1
            continue
        if plant is not None and (m & plant).sum() < min_in_plant * area:
            dropped["off_plant"] += 1
            continue
        if any((m & k).sum() > dedup_iou * (m | k).sum() for k, _, _ in kept):
            dropped["duplicate"] += 1
            continue
        kept.append((m, s, p))
    return kept, dropped


def label_map(masks, shape):
    labels = np.zeros(shape, np.uint16)
    for i in range(len(masks) - 1, -1, -1):            # paint low score first, high score wins
        labels[masks[i]] = i + 1
    return labels


def bbox(m):
    ys, xs = np.nonzero(m)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def overlay(color_bgr, masks, scores, alpha=0.5):
    out = color_bgr.copy()
    for i, m in enumerate(masks):
        c = PALETTE[i % len(PALETTE)]
        out[m] = (out[m] * (1 - alpha) + c * alpha).astype(np.uint8)
    for i, m in enumerate(masks):
        c = tuple(int(v) for v in PALETTE[i % len(PALETTE)])
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, c, 2)
        ys, xs = np.nonzero(m)
        y, x = int(np.median(ys)), int(np.median(xs))
        label = f"{i} {scores[i]:.2f}"
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description="SAM3 per-leaf instance masks for every shot of a capture",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture", help="raw capture folder (manifest.csv, color_*.png, camera_params.json)")
    ap.add_argument("--out", default="leaf_masks", help="output folder, relative to the capture")
    ap.add_argument("--prompts", nargs="+", default=["leaf"], help="SAM3 text prompts")
    ap.add_argument("--confidence", type=float, default=0.5,
                    help="SAM3 detection threshold; instances scoring below it are dropped")
    ap.add_argument("--checkpoint", default=sdc.CHECKPOINT, help="SAM3 weights")
    ap.add_argument("--cameras", default=None,
                    help="2DGS maps/cameras.json; only its stems are processed (default: all shots)")
    ap.add_argument("--indices", default=None, help="manifest indices, e.g. 0-8,10")
    ap.add_argument("--no-undistort", action="store_true",
                    help="keep the raw distorted frame (masks will NOT match cameras.json K)")
    ap.add_argument("--plant-mask-dir", default="mask",
                    help="folder (relative to the capture) of binary mask_<stem>.png plant masks; "
                         "'' to disable")
    ap.add_argument("--min-in-plant", type=float, default=0.5,
                    help="keep an instance only if this fraction of it lies inside the plant mask")
    ap.add_argument("--min-area", type=int, default=150, help="drop instances smaller than this, px")
    ap.add_argument("--dedup-iou", type=float, default=0.7,
                    help="drop an instance overlapping a higher-scoring one by more than this IoU")
    ap.add_argument("--overwrite", action="store_true", help="redo shots whose <stem>.npz exists")
    args = ap.parse_args()

    capture = Path(args.capture)
    if not capture.is_dir():
        sys.exit(f"capture folder not found: {capture}")
    if not Path(args.checkpoint).is_file():
        sys.exit(f"SAM3 checkpoint not found: {args.checkpoint}  (pass --checkpoint)")
    K, dist, wh = load_color_camera(capture)
    manifest = load_manifest(capture)

    rows = sorted(manifest.values(), key=lambda r: int(r["index"]))
    if args.indices:
        sel = parse_indices(args.indices)
        rows = [r for r in rows if int(r["index"]) in sel]
    if args.cameras:
        cams = json.loads(Path(args.cameras).read_text())
        stems = {v["stem"] for v in cams["views"]}
        if tuple(cams["wh"]) != tuple(wh) or not np.allclose(cams["K"], K, atol=1e-3):
            sys.exit(f"{args.cameras}: K / size differ from {capture / 'camera_params.json'} -- "
                     f"different capture?")
        if args.no_undistort:
            print("[WARN] --no-undistort: masks will be in the distorted frame, not the cameras.json one")
        rows = [r for r in rows if stem_of(r["color"]) in stems]
    if not rows:
        sys.exit("no shots selected")

    out = capture / args.out
    out.mkdir(parents=True, exist_ok=True)
    todo = rows if args.overwrite else [r for r in rows if not (out / f"{stem_of(r['color'])}.npz").exists()]
    print(f"[SHOT] {len(rows)} selected, {len(rows) - len(todo)} already done, {len(todo)} to run")

    maps = None if args.no_undistort else cv2.initUndistortRectifyMap(K, dist, None, K, wh, cv2.CV_32FC1)
    plant_dir = capture / args.plant_mask_dir if args.plant_mask_dir else None
    if plant_dir is not None and not plant_dir.is_dir():
        print(f"[WARN] no plant mask folder {plant_dir}; off-plant filtering disabled")
        plant_dir = None

    summary_path = out / "leaf_masks.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    shots = {s["stem"]: s for s in summary.get("shots", [])}
    if todo and summary.get("prompts", args.prompts) != args.prompts:
        sys.exit(f"{summary_path} was made with prompts {summary['prompts']}; the prompt indices "
                 f"would not match -- use another --out or --overwrite every shot")

    if todo:
        print(f"[SAM3] {sdc.torch.cuda.get_device_name(0) if sdc.torch.cuda.is_available() else 'CPU'}"
              f"  | prompts {args.prompts} | confidence {args.confidence}")
        processor = sdc.load_model(args.checkpoint, args.confidence)

    for r in todo:
        t0 = time.perf_counter()
        stem = stem_of(r["color"])
        color = cv2.imread(str(capture / r["color"]), cv2.IMREAD_COLOR)
        if color is None:
            sys.exit(f"unreadable colour image {capture / r['color']}")
        if (color.shape[1], color.shape[0]) != tuple(wh):
            sys.exit(f"{r['color']}: size {color.shape[1]}x{color.shape[0]} != intrinsics {wh}")
        plant = None
        if plant_dir is not None:
            p = plant_dir / f"mask_{stem}.png"
            if p.is_file():
                plant = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            else:
                print(f"[WARN] {stem}: no {p.name}; off-plant filter skipped for this shot")
        if maps is not None:
            color = cv2.remap(color, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if plant is not None:
                plant = cv2.remap(plant, *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        plant = None if plant is None else plant > 127

        dets = detect(processor, cv2.cvtColor(color, cv2.COLOR_BGR2RGB), args.prompts)
        kept, dropped = filter_instances(dets, plant, args.min_area, args.min_in_plant, args.dedup_iou)
        h, w = color.shape[:2]
        masks = np.array([k[0] for k in kept], bool).reshape(-1, h, w)
        scores = np.array([k[1] for k in kept], np.float32)
        np.savez_compressed(out / f"{stem}.npz", masks=masks, scores=scores,
                            prompt=np.array([k[2] for k in kept], np.int16),
                            bbox=np.array([bbox(m) for m in masks], np.int32).reshape(-1, 4),
                            labels=label_map(masks, (h, w)))
        cv2.imwrite(str(out / f"{stem}_overlay.jpg"), overlay(color, masks, scores),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        shots[stem] = dict(stem=stem, index=int(r["index"]), color=r["color"], detections=len(dets),
                           kept=len(kept), dropped=dropped,
                           areas_px=[int(m.sum()) for m in masks],
                           scores=[round(float(s), 3) for s in scores])
        print(f"[OK  ] {stem:<12} {len(kept):3d} leaves  ({len(dets)} detected, dropped {dropped})"
              f"  {time.perf_counter() - t0:.1f} s")

    summary.update(prompts=args.prompts, confidence=args.confidence, checkpoint=args.checkpoint,
                   undistorted=maps is not None, K=K.tolist(), wh=list(wh),
                   plant_mask_dir=str(plant_dir) if plant_dir else None,
                   min_in_plant=args.min_in_plant, min_area=args.min_area, dedup_iou=args.dedup_iou,
                   shots=sorted(shots.values(), key=lambda s: s["index"]))
    summary_path.write_text(json.dumps(summary, indent=2))
    empty = [s["stem"] for s in summary["shots"] if s["kept"] == 0]
    print(f"\n[DONE] {len(summary['shots'])} shots in {out}")
    if empty:
        print(f"[WARN] no leaves kept in {empty} -- lower --confidence or try other --prompts")


if __name__ == "__main__":
    main()
