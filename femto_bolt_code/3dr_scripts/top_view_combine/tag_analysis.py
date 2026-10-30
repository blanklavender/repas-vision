#!/usr/bin/env python3
"""
tag_analysis.py  --  how consistent are the AprilTags once every shot is moved into the
reference frame with its registered pose?

The tags are glued to the pot, so each unique tag is ONE fixed object: after registration
every sighting of it should land on the same spot. The spread of its sightings is the
registration + tag-PnP error. Tags are re-detected on the raw colour images exactly as
register_side.py does (same size / family / max obliquity, from registration.json).

Per sighting (shot k, tag i): PnP pose in the shot's colour camera, moved by T_shot_to_ref.
Consensus per tag: median centre and median corners over its usable sightings.

Tables (printed, and CSV in <work>/tag_analysis/):
    per_tag.csv        sightings, angle span, radius / height on the pot, centre spread
                       (rms / max, split into radial / tangential / vertical about the
                       turntable axis), normal spread
    per_shot.csv       tags used, mean / max centre error of its tags, and the rigid
                       correction (rotation, shift at the plant) that would move its tags onto
                       the consensus = that shot's pose error as the tags see it
    pair_distances.csv every pair of tags seen together: centre distance measured INSIDE each
                       shot (PnP only, no registration) vs the consensus distance
    observations.csv   every sighting with its errors
    by-angle and by-obliquity summaries (printed)

Directions: radial = away from the turntable axis, tangential = along the turn (about-axis
rotation error), vertical = along the axis (tilt / height error).

    python tag_analysis.py ../../../captures/femto_bolt/flora_unit/plant_10-16_deg_5_speed_5
    python tag_analysis.py <capture> --include-oblique      # also judge the sightings register_side ignores
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

import tv_common as tv
from flora_lib import dataset as ds
from flora_lib import geometry as geo
from flora_lib import tags as tg

OUT_DIR = "tag_analysis"


def kabsch(src, dst):
    """Rigid T with dst ~ T @ src (least squares)."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    R = Vt.T @ np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)]) @ U.T
    return geo.make_T(R, cd - R @ cs)


def depth_vs_pnp(depth, t, normal, cam, rays, shrink=0.7):
    """ToF depth inside the tag vs the PnP tag plane, along each pixel's ray: median of
    (measured z - PnP-plane z) in mm and median ratio measured / PnP. (nan, nan) without depth."""
    if depth is None:
        return float("nan"), float("nan")
    h, w = depth.shape[:2]
    if (w, h) not in rays:
        K = cam["K"] if (w, h) == cam["size"] else geo.scale_K(cam["K"], cam["size"], (w, h))
        rays[(w, h)] = geo.ray_grid(K, cam["dist"], w, h)
    ray = rays[(w, h)]
    c = t["center"]
    quad = c + shrink * (t["corners"] - c)                      # inner 70 %: no tag-edge pixels
    uv = geo.project(quad, cam["K"], cam["dist"])
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, np.round(uv).astype(np.int32), 1)
    ys, xs = np.nonzero(m & (depth > 0))
    if len(xs) < 10:
        return float("nan"), float("nan")
    rr = ray[ys, xs]
    z_pnp = (normal @ c) / (rr @ normal)                         # ray (z = 1) meets the PnP tag plane
    z_tof = depth[ys, xs].astype(float) * cam["depth_scale_mm"] / 1000.0
    return float(np.median(z_tof - z_pnp) * 1000), float(np.median(z_tof / z_pnp))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Spread of every AprilTag across the registered shots",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--work", default=tv.WORK_DIR, help="work sub-folder of the capture")
    ap.add_argument("--include-oblique", action="store_true",
                    help="also use sightings beyond the registration's max obliquity")
    ap.add_argument("--plant-height-mm", type=float, default=150.0,
                    help="height above the axis point where a shot's pose error is reported")
    args = ap.parse_args()

    capture = Path(args.capture)
    work = tv.work_dir(capture, args.work)
    try:
        rows = ds.read_manifest(capture)
        cam = ds.load_color_camera(capture)
        regn = ds.read_registration(work)
    except ds.DatasetError as e:
        sys.exit(str(e))
    st = regn["settings"]
    A = regn["axis"]
    a, p0 = A["axis"], A["point"]
    X_plant = p0 + a * args.plant_height_mm / 1000.0
    det = tg.make_detector(st["tag_family"])
    by_stem = {r["stem"]: r for r in rows}

    # ---- sightings in the reference frame
    obs, rays = [], {}
    for s in regn["shots"]:
        row = by_stem[s["stem"]]
        gray = cv2.imread(str(capture / row["color"]), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(capture / row["depth"]), cv2.IMREAD_UNCHANGED) if row["depth"] else None
        T = s["T"]
        for tid, t in tg.detect_tags(gray, cam["K"], cam["dist"], det, st["tag_size_mm"], st["max_obliquity"]).items():
            if not (t["ok"] or args.include_oblique):
                continue
            P = tg.tag_pose(t["corners"])
            d_mm, ratio = depth_vs_pnp(depth, t, P[:3, 2], cam, rays)
            obs.append(dict(shot=s["index"], stem=s["stem"], deg=s["true_deg"], tag=tid, ok=t["ok"],
                            obliq=t["obliquity_deg"], reproj=t["reproj_px"], range_mm=np.linalg.norm(t["center"]) * 1000,
                            c_cam=t["center"], c=geo.apply(T, t["center"][None])[0],
                            corners=geo.apply(T, t["corners"]), n=(T[:3, :3] @ P[:3, 2]), dz=d_mm, ratio=ratio))
    if not obs:
        sys.exit("no tag sightings")
    tags = sorted({o["tag"] for o in obs})
    print(f"[TAGS] {len(obs)} sightings of {len(tags)} tags in {len(regn['shots'])} shots "
          f"(tag {st['tag_size_mm']:g} mm, obliquity <= {st['max_obliquity']:g} deg"
          f"{', oblique ones included' if args.include_oblique else ''})")

    # ---- consensus per tag + per-sighting errors
    cons = {}
    for i in tags:
        O = [o for o in obs if o["tag"] == i]
        c = np.median([o["c"] for o in O], axis=0)
        n = geo.unit(np.mean([o["n"] for o in O], axis=0))
        v = (c - p0) - ((c - p0) @ a) * a
        rad = geo.unit(v)
        cons[i] = dict(c=c, n=n, corners=np.median([o["corners"] for o in O], axis=0), rad=rad,
                       tan=np.cross(a, rad), radius=np.linalg.norm(v) * 1000, height=(c - p0) @ a * 1000)
    for o in obs:
        C = cons[o["tag"]]
        d = (o["c"] - C["c"]) * 1000
        o.update(err=np.linalg.norm(d), e_rad=d @ C["rad"], e_tan=d @ C["tan"], e_ver=d @ a,
                 e_n=np.degrees(np.arccos(np.clip(o["n"] @ C["n"], -1, 1))))

    out = work / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    # ---- per tag
    per_tag = []
    print("\nPER TAG  (centre error = distance of a sighting from the tag's consensus centre, mm)")
    print(f"{'tag':>4} {'n':>3} {'seen deg':>13} {'r mm':>6} {'h mm':>6} | {'rms':>5} {'p90':>5} {'max':>5} {'@shot':>6}"
          f" | {'radial':>6} {'tang':>6} {'vert':>6} (rms) | {'normal':>6} deg")
    for i in tags:
        O = [o for o in obs if o["tag"] == i]
        e = np.array([o["err"] for o in O])
        k = int(np.argmax(e))
        degs = np.array([o["deg"] for o in O])
        r = dict(tag=i, n=len(O), first_deg=round(degs.min(), 1), last_deg=round(degs.max(), 1),
                 radius_mm=round(cons[i]["radius"], 1), height_mm=round(cons[i]["height"], 1),
                 err_rms=round(float(np.sqrt((e ** 2).mean())), 2), err_p90=round(float(np.percentile(e, 90)), 2),
                 err_max=round(float(e.max()), 2), worst_shot=O[k]["shot"],
                 radial_rms=round(float(np.sqrt(np.mean([o["e_rad"] ** 2 for o in O]))), 2),
                 tangential_rms=round(float(np.sqrt(np.mean([o["e_tan"] ** 2 for o in O]))), 2),
                 vertical_rms=round(float(np.sqrt(np.mean([o["e_ver"] ** 2 for o in O]))), 2),
                 normal_rms_deg=round(float(np.sqrt(np.mean([o["e_n"] ** 2 for o in O]))), 2),
                 obliquity_mean=round(float(np.mean([o["obliq"] for o in O])), 1))
        per_tag.append(r)
        span = f"{r['first_deg']:5.1f}-{r['last_deg']:5.1f}" if r["last_deg"] - r["first_deg"] < 300 else "wraps 0"
        print(f"{i:>4} {r['n']:>3} {span:>13} {r['radius_mm']:>6.1f} {r['height_mm']:>6.1f} | {r['err_rms']:>5.1f} "
              f"{r['err_p90']:>5.1f} {r['err_max']:>5.1f} {r['worst_shot']:>6} | {r['radial_rms']:>6.1f} "
              f"{r['tangential_rms']:>6.1f} {r['vertical_rms']:>6.1f}       | {r['normal_rms_deg']:>6.2f}")
    write_csv(out / "per_tag.csv", per_tag)

    # ---- per shot
    per_shot = []
    edges = {e["tgt"]: e for e in regn["registration"]["edges"]}
    for s in regn["shots"]:
        O = [o for o in obs if o["shot"] == s["index"]]
        if not O:
            per_shot.append(dict(shot=s["index"], stem=s["stem"], deg=s["true_deg"], tags="", n_tags=0))
            continue
        src = np.vstack([o["corners"] for o in O])
        dst = np.vstack([cons[o["tag"]]["corners"] for o in O])
        Tc = kabsch(src, dst)
        e = np.array([o["err"] for o in O])
        edge = edges.get(s["stem"])
        per_shot.append(dict(shot=s["index"], stem=s["stem"], deg=s["true_deg"], tags=" ".join(str(o["tag"]) for o in O),
                             n_tags=len(O), err_mean=round(float(e.mean()), 2), err_max=round(float(e.max()), 2),
                             corr_rot_deg=round(geo.rot_deg(Tc[:3, :3]), 2),
                             corr_shift_mm=round(float(np.linalg.norm(geo.apply(Tc, X_plant[None])[0] - X_plant) * 1000), 2),
                             edge_in_deg=round(edge["rotation_deg"], 2) if edge else "",
                             edge_logged_deg=round(edge["logged_deg"], 2) if edge else "",
                             drift_deg=round(float(A["drift_deg"][s["index"]]), 2),
                             obliq_mean=round(float(np.mean([o["obliq"] for o in O])), 1),
                             reproj_px=round(float(np.mean([o["reproj"] for o in O])), 3)))
    write_csv(out / "per_shot.csv", per_shot)
    ok_shots = [r for r in per_shot if r["n_tags"]]
    print(f"\nPER SHOT  (correction = the rigid move that puts this shot's tags onto the consensus; "
          f"shift measured {args.plant_height_mm:g} mm up the axis)")
    print(f"{'shot':>4} {'deg':>6} {'tags':>12} | {'mean':>5} {'max':>5} mm | {'corr':>5} deg {'shift':>5} mm | "
          f"{'edge in':>7} {'drift':>6}")
    for r in ok_shots:
        print(f"{r['shot']:>4} {r['deg']:>6.1f} {r['tags']:>12} | {r['err_mean']:>5.1f} {r['err_max']:>5.1f}    | "
              f"{r['corr_rot_deg']:>5.2f}     {r['corr_shift_mm']:>5.1f}    | {str(r['edge_in_deg']):>7} {r['drift_deg']:>6.2f}")

    # ---- by angle / obliquity
    print("\nBY TURNTABLE ANGLE (45 deg bins)")
    for lo in range(0, 360, 45):
        R = [r for r in ok_shots if lo <= r["deg"] < lo + 45]
        if R:
            print(f"  {lo:>3}-{lo + 45:<3} deg: {len(R):>2} shots, tag error mean {np.mean([r['err_mean'] for r in R]):4.1f} mm, "
                  f"pose correction median {np.median([r['corr_rot_deg'] for r in R]):4.2f} deg / "
                  f"{np.median([r['corr_shift_mm'] for r in R]):4.1f} mm")
    print("BY OBLIQUITY (how head-on the tag was seen)")
    for lo, hi in ((0, 20), (20, 35), (35, 45), (45, 55), (55, 90)):
        e = [o["err"] for o in obs if lo <= o["obliq"] < hi]
        if e:
            print(f"  {lo:>2}-{hi:<2} deg: {len(e):>4} sightings, centre error median {np.median(e):4.1f} mm, "
                  f"p90 {np.percentile(e, 90):4.1f} mm")

    # ---- pair distances: inside each shot (no registration) vs consensus
    pairs = {}
    for s in regn["shots"]:
        O = {o["tag"]: o for o in obs if o["shot"] == s["index"]}
        for i in O:
            for j in O:
                if i < j:
                    pairs.setdefault((i, j), []).append(np.linalg.norm(O[i]["c_cam"] - O[j]["c_cam"]) * 1000)
    pd_rows = []
    for (i, j), d in sorted(pairs.items()):
        d = np.array(d)
        dc = np.linalg.norm(cons[i]["c"] - cons[j]["c"]) * 1000
        pd_rows.append(dict(tag_a=i, tag_b=j, n_shots=len(d), consensus_mm=round(dc, 2), in_shot_mean_mm=round(d.mean(), 2),
                            in_shot_std_mm=round(d.std(), 2), in_shot_min_mm=round(d.min(), 2), in_shot_max_mm=round(d.max(), 2),
                            mean_minus_consensus_mm=round(d.mean() - dc, 2)))
    if pd_rows:
        write_csv(out / "pair_distances.csv", pd_rows)
        print("\nTAG-PAIR CENTRE DISTANCES  (in-shot = measured inside one shot by PnP only; consensus = after registration)")
        print(f"{'pair':>8} {'shots':>5} | {'consensus':>9} | {'in-shot mean':>12} {'std':>5} {'min':>6} {'max':>6} | {'diff':>5}")
        for r in sorted(pd_rows, key=lambda r: -r["n_shots"]):
            print(f"{r['tag_a']:>3}-{r['tag_b']:<4} {r['n_shots']:>5} | {r['consensus_mm']:>9.1f} | {r['in_shot_mean_mm']:>12.1f} "
                  f"{r['in_shot_std_mm']:>5.1f} {r['in_shot_min_mm']:>6.1f} {r['in_shot_max_mm']:>6.1f} | "
                  f"{r['mean_minus_consensus_mm']:>+5.1f}")

    # ---- ToF depth vs PnP: do the two agree on where the tags are?
    dz = np.array([o["dz"] for o in obs])
    good = np.isfinite(dz)
    if good.any():
        rng = np.array([o["range_mm"] for o in obs])[good]
        ob = np.array([o["obliq"] for o in obs])[good]
        rt = np.array([o["ratio"] for o in obs])[good]
        slope, icpt = np.polyfit(rng, dz[good], 1)
        print("\nTOF DEPTH vs PnP AT THE TAGS  (+ = ToF puts the tag farther from the camera than PnP)")
        print(f"  {good.sum()} sightings: median {np.median(dz[good]):+.1f} mm (p10 {np.percentile(dz[good], 10):+.1f}, "
              f"p90 {np.percentile(dz[good], 90):+.1f}), ratio ToF/PnP {np.median(rt):.4f}")
        if rng.max() - rng.min() > 100:                         # a scale error grows with range
            print(f"  vs range: diff = {icpt:+.1f} mm + {slope * 1000:+.1f} mm per metre of range "
                  f"(sightings span {rng.min():.0f}-{rng.max():.0f} mm)")
        for lo, hi in ((0, 20), (20, 35), (35, 45), (45, 90)):
            m = (ob >= lo) & (ob < hi)
            if m.any():
                print(f"  obliquity {lo:>2}-{hi:<2} deg: {m.sum():>3} sightings, median {np.median(dz[good][m]):+.1f} mm")
        for i in tags:
            v = [o["dz"] for o in obs if o["tag"] == i and np.isfinite(o["dz"])]
            per_tag[tags.index(i)]["tof_minus_pnp_mm"] = round(float(np.median(v)), 2) if v else ""
        write_csv(out / "per_tag.csv", per_tag)

    write_csv(out / "observations.csv", [dict(shot=o["shot"], deg=o["deg"], tag=o["tag"], ok=o["ok"],
                                              tof_minus_pnp_mm=round(o["dz"], 2) if np.isfinite(o["dz"]) else "",
                                              obliquity_deg=round(o["obliq"], 1), reproj_px=round(o["reproj"], 3),
                                              range_mm=round(o["range_mm"], 1), err_mm=round(o["err"], 2),
                                              radial_mm=round(o["e_rad"], 2), tangential_mm=round(o["e_tan"], 2),
                                              vertical_mm=round(o["e_ver"], 2), normal_err_deg=round(o["e_n"], 2))
                                         for o in obs])
    e_all = np.array([o["err"] for o in obs])
    print(f"\n[ALL ] centre error over {len(obs)} sightings: median {np.median(e_all):.1f}, p90 {np.percentile(e_all, 90):.1f}, "
          f"max {e_all.max():.1f} mm")
    print(f"[OUT ] {out}  (per_tag.csv, per_shot.csv, pair_distances.csv, observations.csv)")


if __name__ == "__main__":
    main()
