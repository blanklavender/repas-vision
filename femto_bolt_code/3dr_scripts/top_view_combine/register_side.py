#!/usr/bin/env python3
"""
register_side.py  --  step 1 of top_view_combine: AprilTag registration of the side sweep
and the turntable axis (= ../tag_register.py, writing into the work folder).

Runs on the RAW colour images of the side sweep. Per adjacent pair, per shared tag
T = P_(i+1) @ P_i^-1 (colour-only PnP tag poses); the pair transform is the chordal L2 mean
of the per-tag rotations + mean tag-centre translation. The pairs are chained into the
first shot's frame; on a full turn the last->first edge closes the loop and its error is
spread evenly. The turntable axis is the common rotation axis of the registered poses,
pointing up; the tags' distance from it (the pot radius) is printed as a check.

    python register_side.py ../../../captures/femto_bolt/flora_unit/plant_10-16_deg_5_speed_5
    python register_side.py <capture> --max-obliquity 45 --force

I/O
---
in   <capture>/manifest.csv, camera_params.json, color_<stem>.png      sweep_capture.py
out  <capture>/top_view_combine/registration.json   per-shot T_shot_to_ref (X_ref = T @ X_shot,
     metres, colour-camera frames; reference = first shot), edges, loop error, turntable axis
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import tv_common as tv
from flora_lib import dataset as ds
from flora_lib import geometry as geo
from flora_lib import registration as reg
from flora_lib import tags as tg

SETTING_KEYS = ("max_obliquity", "tag_size_mm", "tag_family", "loop")


def main():
    ap = argparse.ArgumentParser(description="AprilTag registration of the side sweep + turntable axis",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--work", default=tv.WORK_DIR, help="work sub-folder of the capture")
    ap.add_argument("--max-obliquity", type=float, default=55.0, help="ignore tags seen more obliquely (deg)")
    ap.add_argument("--tag-size-mm", type=float, default=24.0, help="printed tag size (black square edge)")
    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--loop", choices=["auto", "yes", "no"], default="auto",
                    help="close the loop last->first (auto: when the sweep covers a full turn)")
    ap.add_argument("--force", action="store_true", help="register again even if registration.json is current")
    args = ap.parse_args()

    capture = Path(args.capture)
    work = tv.work_dir(capture, args.work)
    try:
        rows = ds.read_manifest(capture)
        cam = ds.load_color_camera(capture)
    except ds.DatasetError as e:
        sys.exit(str(e))
    settings = {k: getattr(args, k) for k in SETTING_KEYS}
    stems = [r["stem"] for r in rows]

    out_path = work / ds.REGISTRATION
    if out_path.is_file() and not args.force:
        old = json.loads(out_path.read_text())
        if old.get("settings") == settings and [s["stem"] for s in old.get("shots", [])] == stems:
            print(f"[SKIP] {out_path} already made with these settings; --force to redo")
            return

    det = tg.make_detector(args.tag_family)
    tags = []
    for r in rows:
        gray = cv2.imread(str(capture / r["color"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            sys.exit(f"[FAIL] cannot read {capture / r['color']}")
        t = tg.detect_tags(gray, cam["K"], cam["dist"], det, args.tag_size_mm, args.max_obliquity)
        tags.append(t)
        print(f"[SHOT] {r['stem']}  usable tags {sorted(i for i, x in t.items() if x['ok'])}")
    print()

    degs = [r["true_deg"] for r in rows]
    try:
        poses, info = reg.chain_sweep(tags, degs, names=stems, loop=args.loop)
        centres = np.array([geo.apply(T, np.array([x["center"]]))[0]
                            for T, t in zip(poses, tags) for x in t.values() if x["ok"]])
        A = reg.turntable_axis(poses, degs, centres.mean(0))
    except reg.RegistrationError as e:
        sys.exit(f"[FAIL] {e}")
    tag_r = reg.radial_distance(centres, A["axis"], A["point"]) * 1000
    tilt = np.degrees(np.arccos(np.clip(A["axis"] @ np.array([0.0, -1.0, 0.0]), -1, 1)))
    dr = np.abs(A["drift_deg"])

    work.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(
        finished=datetime.now().isoformat(timespec="seconds"), settings=settings,
        convention="X_ref = T_shot_to_ref @ X_shot (metres, colour-camera frames); reference = first shot",
        registration=info,
        axis=dict(axis=A["axis"].tolist(), point=A["point"].tolist(), turn_sign=A["turn_sign"],
                  ref_deg=A["ref_deg"], drift_deg=A["drift_deg"].tolist(),
                  tag_radius_mm=dict(median=float(np.median(tag_r)), min=float(tag_r.min()),
                                     max=float(tag_r.max()), n=len(tag_r))),
        shots=[dict(index=r["index"], stem=r["stem"], true_deg=r["true_deg"],
                    tags=sorted(i for i, x in t.items() if x["ok"]), T_shot_to_ref=T.tolist())
               for r, t, T in zip(rows, tags, poses)]), indent=2))

    print()
    if info.get("loop"):
        print(f"[LOOP ] error {info['loop']['error_deg']:.2f} deg / {info['loop']['error_mm']:.1f} mm "
              f"(spread over the loop)")
    print(f"[POSE ] registered turn vs logged turntable angle: median {np.median(dr):.2f}, max {dr.max():.2f} deg")
    print(f"[AXIS ] {np.round(A['axis'], 4).tolist()} through {np.round(A['point'] * 1000, 1).tolist()} mm "
          f"(reference camera frame), {tilt:.1f} deg from the camera's -y")
    print(f"[AXIS ] tag centres lie {np.median(tag_r):.1f} mm from the axis (median; {tag_r.min():.1f}-"
          f"{tag_r.max():.1f} mm over {len(tag_r)} sightings) -- that is the pot radius")
    print(f"[OUT  ] {out_path}")


if __name__ == "__main__":
    main()
