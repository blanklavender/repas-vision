#!/usr/bin/env python3
"""
tag_register.py  --  register every shot of a turntable sweep from the AprilTags on the pot
and fit the turntable axis. Runs on the RAW colour images, before any cleaning, so the
cleaning step can crop each shot to a cylinder around the axis.

Registration (flora_lib.tags / flora_lib.registration, = tag_pose_register.py "pose"):
    per adjacent pair, per shared tag   T = P_(i+1) @ P_i^-1      (colour-only PnP tag poses)
    pair transform                      chordal L2 mean of the per-tag rotations + mean
                                        tag-centre translation
    the pairs are chained into the first shot's frame; on a full turn the last->first edge
    closes the loop and its error is spread evenly over the shots.

Turntable axis (flora_lib.registration.turntable_axis): the common rotation axis of the
registered poses, pointing up, through the point fitted to all shots' translations (at the
mean tag height). The tags' distance from it is printed as a check: it is the pot radius.

Faulty shots (flora_lib.registration.chain_without_outliers): while the worst shot's
registered rotation is more than --max-turn-error degrees off its logged turn, it is dropped
and the sweep is chained again past it (its neighbours registered to each other directly),
because a bad tag pose also bends the chain for every later shot. Dropped shots are marked
rejected: the cleaning step skips them, so they are never stitched.

    python tag_register.py ../../captures/femto_bolt/flora_unit/plant_10-14_deg_5_speed_5
    python tag_register.py <capture> --max-obliquity 45 --force
    python tag_register.py <capture> --max-turn-error 3 --force

A capture already registered with the same settings is skipped unless --force.

I/O contract (layout: flora_pipeline/flora_lib/dataset.py)
------------
in   <capture>/manifest.csv, camera_params.json, color_<stem>.png      sweep_capture.py
out  <capture>/registration.json   settings, per-shot T_shot_to_ref (X_ref = T @ X_shot,
                                   metres, colour-camera frames; reference = first shot),
                                   turn_error_deg + rejected per shot, edges, loop error,
                                   turntable axis
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "flora_pipeline"))
from flora_lib import dataset as ds  # noqa: E402
from flora_lib import registration as reg  # noqa: E402
from flora_lib import tags as tg  # noqa: E402

SETTING_KEYS = ("max_obliquity", "tag_size_mm", "tag_family", "loop", "max_turn_error")


def main():
    ap = argparse.ArgumentParser(description="AprilTag registration of a turntable sweep + turntable axis",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--max-obliquity", type=float, default=55.0, help="ignore tags seen more obliquely (deg)")
    ap.add_argument("--tag-size-mm", type=float, default=24.0, help="printed tag size (black square edge)")
    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--loop", choices=["auto", "yes", "no"], default="auto",
                    help="close the loop last->first (auto: when the sweep covers a full turn)")
    ap.add_argument("--max-turn-error", type=float, default=2.0,
                    help="reject shots whose registered rotation is more than this many degrees off the logged "
                         "turntable turn (bad tag poses); they are not cleaned or stitched. 0 = keep every shot")
    ap.add_argument("--force", action="store_true", help="register again even if registration.json is current")
    args = ap.parse_args()

    capture = Path(args.capture)
    try:
        rows = ds.read_manifest(capture)
        cam = ds.load_color_camera(capture)
    except ds.DatasetError as e:
        sys.exit(str(e))
    settings = {k: getattr(args, k) for k in SETTING_KEYS}
    stems = [r["stem"] for r in rows]

    out_path = capture / ds.REGISTRATION
    if out_path.is_file() and not args.force:
        old = json.loads(out_path.read_text())
        if old.get("settings") == settings and [s["stem"] for s in old.get("shots", [])] == stems:
            print(f"[SKIP] {capture} already registered with these settings; --force to redo")
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
    n = len(rows)
    try:
        poses, info, A, err, keep = reg.chain_without_outliers(tags, degs, names=stems, loop=args.loop,
                                                               max_turn_error=args.max_turn_error)
    except reg.RegistrationError as e:
        sys.exit(f"[FAIL] {e}")
    rejected = [k for k in range(n) if k not in keep]
    # tag centres of the kept shots, reference frame: anchored the axis point, measure the pot
    centres = reg.tag_centres(tags, poses, keep)
    tag_r = reg.radial_distance(centres, A["axis"], A["point"]) * 1000
    tilt = np.degrees(np.arccos(np.clip(A["axis"] @ np.array([0.0, -1.0, 0.0]), -1, 1)))
    dr = np.abs(A["drift_deg"][keep])

    out_path.write_text(json.dumps(dict(
        finished=datetime.now().isoformat(timespec="seconds"), settings=settings,
        convention="X_ref = T_shot_to_ref @ X_shot (metres, colour-camera frames); reference = first shot",
        registration=info,
        axis=dict(axis=A["axis"].tolist(), point=A["point"].tolist(), turn_sign=A["turn_sign"],
                  ref_deg=A["ref_deg"], drift_deg=A["drift_deg"].tolist(),
                  tag_radius_mm=dict(median=float(np.median(tag_r)), min=float(tag_r.min()),
                                     max=float(tag_r.max()), n=len(tag_r))),
        shots=[dict(index=r["index"], stem=r["stem"], true_deg=r["true_deg"],
                    tags=sorted(i for i, x in t.items() if x["ok"]), turn_error_deg=float(e),
                    rejected=k in rejected, T_shot_to_ref=T.tolist())
               for k, (r, t, T, e) in enumerate(zip(rows, tags, poses, err))]), indent=2))

    print()
    if info.get("loop"):
        print(f"[LOOP ] error {info['loop']['error_deg']:.2f} deg / {info['loop']['error_mm']:.1f} mm "
              f"(spread over the loop)")
    for k in rejected:
        print(f"[DROP ] {stems[k]}  rotation {err[k]:.2f} deg off its logged turn ({A['drift_deg'][k]:+.2f} about "
              f"the axis) > {args.max_turn_error:g} -- left out of cleaning and stitching")
    if args.max_turn_error > 0:
        print(f"[POSE ] {len(keep)}/{n} shots within {args.max_turn_error:g} deg of their logged turn "
              f"(worst kept {err[keep].max():.2f} deg)")
    print(f"[POSE ] registered turn vs logged turntable angle{' (kept shots)' if rejected else ''}: "
          f"median {np.median(dr):.2f}, max {dr.max():.2f} deg")
    print(f"[AXIS ] {np.round(A['axis'], 4).tolist()} through {np.round(A['point'] * 1000, 1).tolist()} mm "
          f"(reference camera frame), {tilt:.1f} deg from the camera's -y")
    print(f"[AXIS ] tag centres lie {np.median(tag_r):.1f} mm from the axis (median; {tag_r.min():.1f}-"
          f"{tag_r.max():.1f} mm over {len(tag_r)} sightings) -- that is the pot radius")
    print(f"[OUT  ] {out_path}")


if __name__ == "__main__":
    main()
