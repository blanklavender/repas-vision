#!/usr/bin/env python3
"""
export_colmap_clean.py  --  COLMAP export from a flora_lib.tof_clean run, leaving out chosen views.

Same format as export_colmap_tvc.py (its writer is reused). Differences:
    seed     <clean_run>/side_fused_clean.ply
    images   copied byte for byte from an existing export of the same capture, minus --exclude
    split    heldout_views.json records the capture's original every-8th views. Removing views would
             otherwise shift gsplat's index-based split onto different views, so the masked trainer and
             diag_depth.py both read this file.

    cd ~/Projects/repas-vision/femto_bolt_code/3dr_scripts
    python export_colmap_clean.py --capture <raw> --clean_run <run> --images_from <export> \
        --exclude 020_ang100 022_ang110 --out <new folder>
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from export_colmap import orthonormalise, read_ply, rot_to_qvec, write_model

TEST_EVERY = 8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", type=Path, required=True, help="raw capture: poses.json, camera_params.json")
    ap.add_argument("--clean_run", type=Path, required=True)
    ap.add_argument("--images_from", type=Path, required=True, help="export of the same capture: images/")
    ap.add_argument("--exclude", nargs="*", default=[], help="view stems to leave out")
    ap.add_argument("--out", type=Path, required=True, help="a new folder")
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists; write each export to a new folder")

    run = json.loads((args.clean_run / "run.json").read_text())
    if sorted(run["dropped_views"]) != sorted(args.exclude):
        sys.exit(f"--exclude {sorted(args.exclude)} differs from the views this run dropped {sorted(run['dropped_views'])}")
    shots = json.loads((args.capture / "top_view_combine/stitch/poses.json").read_text())["shots"]
    cam = json.loads((args.capture / "camera_params.json").read_text())["color_intrinsic"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], float)
    wh = (cam["width"], cam["height"])

    heldout = [f"color_{s['stem']}.png" for i, s in enumerate(shots) if i % TEST_EVERY == 0]
    if set(args.exclude) & {n[len("color_"):-len(".png")] for n in heldout}:
        sys.exit("an excluded view is a held-out view; the split would not match the earlier models")

    (args.out / "images").mkdir(parents=True)
    kept = [s for s in shots if s["stem"] not in args.exclude]
    images = []
    for iid, s in enumerate(kept, start=1):
        name = f"color_{s['stem']}.png"
        shutil.copyfile(args.images_from / "images" / name, args.out / "images" / name)
        T = np.asarray(s["T_shot_to_ref"], float)
        R_cw = orthonormalise(T[:3, :3]).T
        images.append((iid, rot_to_qvec(R_cw), -R_cw @ T[:3, 3], name))

    xyz, rgb = read_ply(args.clean_run / "side_fused_clean.ply")
    write_model(args.out / "sparse" / "0", K, wh, images, xyz, rgb)
    (args.out / "heldout_views.json").write_text(json.dumps(dict(
        note="the capture's original every-8th views, by name", names=heldout), indent=2))
    (args.out / "export.json").write_text(json.dumps(dict(
        capture=str(args.capture), clean_run=str(args.clean_run), images_from=str(args.images_from),
        cleaning_overrides=run.get("overrides", {}), config_hash=run["config_hash"], excluded=args.exclude,
        n_images=len(images), n_points=len(xyz), heldout=heldout), indent=2))
    print(f"[OUT ] {args.out}: {len(images)} images, {len(xyz):,} points, held out {len(heldout)}")


if __name__ == "__main__":
    main()
