#!/usr/bin/env python3
"""
sam3_batch_masks.py  --  one SAM3 pot + plant mask per turntable stop.

Runs the SAM3 union mask from sam3_data_cleaning.py (same model loader, same
prompt union) on every stop's colour image and saves it next to that stop's
raw/ and avg/ folders, for tag_stitch.py --mask.

    python sam3_batch_masks.py ../../captures/femto_bolt/flora_unit/plant_A
    python sam3_batch_masks.py <capture> --prompts "yellow cylinder" "green plant" --confidence 0.4
    python sam3_batch_masks.py <capture> --indices 0-5 --overwrite

I/O contract
------------
in   <capture>/manifest.csv        columns index, color, stop_dir (sweep_capture.py)
     <capture>/<color>             colour PNG per stop, 1280x720
     --checkpoint                  SAM3 weights (default: sam3_data_cleaning.CHECKPOINT)
out  <capture>/<stop_dir>/mask.png          uint8, COLOUR resolution, 255 = pot/plant.
                                            Not eroded -- tag_stitch.py --mask-erode does that.
     <capture>/<stop_dir>/mask_overlay.png  colour image with the mask tinted green, for checking
     <capture>/<stop_dir>/mask.json         prompts, confidence, detections per prompt, pixel count
     <capture>/sam3_masks.json              run summary

A stop whose union mask is empty gets NO mask.png and is listed at the end;
tag_stitch.py --mask then refuses to run until it has one (re-run that stop
with other prompts / a lower --confidence).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import sam3_data_cleaning as sdc          # load_model / union_mask / defaults


def parse_indices(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return set(out)


def load_stops(capture, indices):
    path = capture / "manifest.csv"
    if not path.is_file():
        sys.exit(f"missing {path}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"index", "color", "stop_dir"} <= set(rows[0]):
        sys.exit(f"{path} needs columns index, color, stop_dir "
                 f"(capture with the burst-capable sweep_capture.py)")
    rows = [r for r in rows if indices is None or int(r["index"]) in indices]
    if not rows:
        sys.exit("no manifest rows selected")
    for r in rows:
        if not (capture / r["color"]).is_file():
            sys.exit(f"missing colour image {capture / r['color']}")
        if not (capture / r["stop_dir"]).is_dir():
            sys.exit(f"missing stop folder {capture / r['stop_dir']}")
    return rows


def overlay(color_bgr, mask, alpha=0.45):
    out = color_bgr.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array([0, 200, 0]) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out


def main():
    ap = argparse.ArgumentParser(description="SAM3 pot + plant mask for every stop of a capture",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture", help="capture folder written by sweep_capture.py")
    ap.add_argument("--prompts", nargs="+", default=[sdc.PROMPT_A, sdc.PROMPT_B],
                    help="text prompts; the mask is the union of all their detections")
    ap.add_argument("--confidence", type=float, default=sdc.CONFIDENCE)
    ap.add_argument("--checkpoint", default=sdc.CHECKPOINT,
                    help="SAM3 weights (relative paths are relative to where you run this)")
    ap.add_argument("--indices", default=None, help="manifest indices, e.g. 0-8,10")
    ap.add_argument("--overwrite", action="store_true", help="replace existing mask.png files")
    args = ap.parse_args()

    capture = Path(args.capture)
    if not capture.is_dir():
        sys.exit(f"capture folder not found: {capture}")
    if not Path(args.checkpoint).is_file():
        sys.exit(f"SAM3 checkpoint not found: {args.checkpoint}  (pass --checkpoint)")
    rows = load_stops(capture, parse_indices(args.indices) if args.indices else None)
    if not args.overwrite:
        done = [r["stop_dir"] for r in rows if (capture / r["stop_dir"] / "mask.png").exists()]
        if done:
            sys.exit(f"{len(done)} stops already have mask.png (e.g. {done[0]}); "
                     f"pass --overwrite to replace them")

    sdc.CHECKPOINT, sdc.CONFIDENCE = args.checkpoint, args.confidence
    print(f"[SAM3] {sdc.torch.cuda.get_device_name(0) if sdc.torch.cuda.is_available() else 'CPU'}"
          f"  | prompts {args.prompts} | confidence {args.confidence}")
    processor = sdc.load_model()

    summary, empty = [], []
    for r in rows:
        t0 = time.perf_counter()
        stop = capture / r["stop_dir"]
        color_bgr = cv2.imread(str(capture / r["color"]), cv2.IMREAD_COLOR)
        if color_bgr is None:
            sys.exit(f"unreadable colour image {capture / r['color']}")
        h, w = color_bgr.shape[:2]
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        mask, counts = sdc.union_mask(processor, Image.fromarray(rgb), args.prompts, (h, w))
        frac = float(mask.mean())
        rec = dict(index=int(r["index"]), stop_dir=r["stop_dir"], prompts=args.prompts,
                   confidence=args.confidence, detections=counts,
                   mask_px=int(mask.sum()), mask_frac=frac)
        if not mask.any():
            empty.append(int(r["index"]))
            print(f"[EMPTY] {r['stop_dir']}  {counts}  -- no mask.png written")
        else:
            for name, img in (("mask.png", mask.astype(np.uint8) * 255),
                              ("mask_overlay.png", overlay(color_bgr, mask))):
                if not cv2.imwrite(str(stop / name), img):
                    sys.exit(f"could not write {stop / name}")
            print(f"[OK  ] {r['stop_dir']:<22} {100 * frac:5.1f}% of image  {counts}  "
                  f"({time.perf_counter() - t0:.1f} s)")
        (stop / "mask.json").write_text(json.dumps(rec, indent=2))
        summary.append(rec)

    (capture / "sam3_masks.json").write_text(json.dumps(
        dict(prompts=args.prompts, confidence=args.confidence, checkpoint=args.checkpoint,
             empty_indices=empty, stops=summary), indent=2))
    print(f"\n[DONE] {len(summary) - len(empty)}/{len(summary)} masks written")
    if empty:
        print(f"[WARN] empty masks at indices {empty} -- re-run them with --indices "
              f"{','.join(map(str, empty))} --overwrite and other prompts / lower --confidence")


if __name__ == "__main__":
    main()
