#!/usr/bin/env python3
"""
spread_analysis.py  --  spread-of-centre validation for tag_stitch.py poses.

Pick one AprilTag seen in many shots. In each shot k its 3D centre c_k^cam
(shot k's colour-camera frame) is pushed through the stored T_k->0 into the
reference frame, c_k = T_k->0 c_k^cam. If the registration and the depth were
perfect, every c_k would be the same point; their scatter is the error.

Reported
--------
* c_bar (mean of c_k) and residuals r_k = c_k - c_bar
* per-axis std (sx, sy, sz) in CAMERA-LOCAL axes: r_k is rotated back into shot
  k's own camera axes (R_k->0^T r_k), so z is that shot's viewing direction.
  This is the frame where ToF depth noise lives; in the reference frame each
  shot's depth axis points a different way around the turntable and the
  asymmetry smears out. Expectation: sz >> sx, sy -- the ratio sz / mean(sx, sy)
  is reported and tested against --aniso-ratio.
* per-axis std in the reference frame too, plus the principal stds
  (covariance eigenvalues)
* RMS radius of dispersion  sqrt(mean |r_k|^2)  and the 95th-percentile radius
* with --truth: bias = c_bar - truth, reported SEPARATELY from dispersion, plus
  its standard error and the total RMS error = sqrt(|bias|^2 + RMS radius^2)

Centre estimators (--center)
----------------------------
    plane    colour ray through the detected tag centre intersected with the
             plane fitted to all confident depth on the tag (default, preferred)
    corners  mean of the 4 corners lifted with LOCAL median depth at each corner
    pnp      colour-only PnP tag centre (known tag size, no depth at all)

Rigour
------
--holdout-tag ID   the tag must NOT have been used by any registration edge
                   (run tag_stitch.py --exclude-tags ID first). This measures true
                   generalisation. The script refuses if any edge used it.
--tag-id ID        any tag; the report lists the edges that used it, since then
                   the spread is partly in-sample (the fit was pulling on it).

    python spread_analysis.py captures/plant_01/stitch --holdout-tag 15
    python spread_analysis.py captures/plant_01/stitch --tag-id 16 --center corners
    python spread_analysis.py captures/plant_01/stitch_holdout15 --holdout-tag 15 \\
        --truth 12.4 -30.1 512.0 --show

I/O contract
------------
in   <stitch>/poses.json           tag_stitch.py: T_shot_to_ref per shot index
     <stitch>/tags_per_shot.json   tag_stitch.py: per-shot tag centres (m, colour frame)
     <stitch>/pairs.json           tag_stitch.py: which tags each edge used
     --truth X Y Z                 optional, mm, in the reference shot's colour frame
out  <stitch>/spread_tag<ID>_<center>.json   all numbers below
     <stitch>/spread_tag<ID>_<center>.csv    per-shot centres + residuals (mm)
     <stitch>/spread_tag<ID>_<center>.png    3D scatter + covariance ellipsoid +
                                             per-axis histograms
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SERIES, ACCENT = "#2a78d6", "#eb6834"          # shot centres, truth / bias


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_json(path):
    if not path.is_file():
        sys.exit(f"missing {path} -- run tag_stitch.py first")
    return json.loads(path.read_text())


def edges_using(pairs, tag):
    return [f"{e['src_index']:03d}->{e['tgt_index']:03d}" for e in pairs
            if e.get("ok") and tag in e["tag"].get("ids_used", [])]


def gather(stitch, tag, center):
    poses = load_json(stitch / "poses.json")
    tags = load_json(stitch / "tags_per_shot.json")
    T = {s["index"]: np.asarray(s["T_shot_to_ref"]) for s in poses["shots"]}
    key = {"plane": "center_plane", "corners": "center_corners_local",
           "pnp": "center_pnp"}[center]
    rows, skipped = [], []
    for s in tags["shots"]:
        rec = s["tags"].get(str(tag))
        if rec is None:
            continue
        if s["index"] not in T:
            skipped.append((s["index"], "shot has no pose (unreachable)"))
            continue
        if not rec.get("ok_pnp" if center == "pnp" else "ok") or rec.get(key) is None:
            skipped.append((s["index"], rec.get("error", f"no {key}")))
            continue
        Tk = T[s["index"]]
        c_cam = np.asarray(rec[key])
        rows.append(dict(index=s["index"], true_deg=s["true_deg"], R=Tk[:3, :3],
                         c=Tk[:3, :3] @ c_cam + Tk[:3, 3],
                         depth_m=float(c_cam[2]), obliquity_deg=rec.get("obliquity_deg"),
                         plane_rms_mm=rec.get("plane_rms_mm")))
    return poses, rows, skipped


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def chi2_3_ppf(p):
    """Quantile of chi-square with 3 dof (bisection on the closed-form CDF)."""
    cdf = lambda x: math.erf(math.sqrt(x / 2)) - math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def spread(rows, truth_mm, aniso_ratio):
    C = np.array([r["c"] for r in rows]) * 1000.0                     # mm, ref frame
    n = len(C)
    c_bar = C.mean(0)
    R = C - c_bar
    R_cam = np.array([r["R"].T @ res for r, res in zip(rows, R)])     # shot-k camera axes
    rad = np.linalg.norm(R, axis=1)
    cov = np.cov(R.T, ddof=1)
    ev, evec = np.linalg.eigh(cov)
    s_cam = R_cam.std(0, ddof=1)
    lateral = float(np.mean(s_cam[:2]))
    ratio = float(s_cam[2] / lateral) if lateral > 0 else float("inf")
    out = dict(
        n_shots=n, c_bar_mm=c_bar.tolist(),
        camera_local_std_mm=dict(sx=float(s_cam[0]), sy=float(s_cam[1]), sz=float(s_cam[2])),
        anisotropy_sz_over_lateral=ratio,
        anisotropy_expected=f"sz >> sx, sy (ratio > {aniso_ratio:g})",
        anisotropy_holds=bool(ratio > aniso_ratio),
        ref_frame_std_mm=dict(zip(("sX", "sY", "sZ"), R.std(0, ddof=1).tolist())),
        principal_std_mm=np.sqrt(np.maximum(ev, 0))[::-1].tolist(),
        principal_axes_ref=evec[:, ::-1].T.tolist(),
        rms_radius_mm=float(np.sqrt(np.mean(rad ** 2))),
        p95_radius_mm=float(np.percentile(rad, 95)),
        max_radius_mm=float(rad.max()),
        covariance_ref_mm2=cov.tolist())
    if truth_mm is not None:
        b = c_bar - truth_mm
        se = R.std(0, ddof=1) / math.sqrt(n)
        out["bias"] = dict(
            truth_mm=truth_mm.tolist(), bias_mm=b.tolist(), bias_norm_mm=float(np.linalg.norm(b)),
            std_error_of_c_bar_mm=se.tolist(), bias_z_score=(b / np.maximum(se, 1e-12)).tolist(),
            total_rms_error_mm=float(math.sqrt(np.linalg.norm(b) ** 2 + out["rms_radius_mm"] ** 2)),
            note="bias = systematic offset of c_bar from truth; dispersion = scatter about "
                 "c_bar. total^2 = |bias|^2 + rms_radius^2")
    return out, C, R, R_cam, cov


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def figure(path, title, C, c_bar, R_cam, cov, conf, truth_mm, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 6.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 5, width_ratios=[1.2, 1.2, 1, 1, 1], hspace=0.45, wspace=0.35)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)

    # ---- 3D scatter about c_bar + covariance ellipsoid ----------------------
    ax = fig.add_subplot(gs[:, :2], projection="3d")
    ax.set_facecolor(SURFACE)
    D = C - c_bar
    ax.scatter(*D.T, s=28, color=SERIES, edgecolor=SURFACE, linewidth=1.0, depthshade=False,
               label=f"shot centres (n={len(D)})")
    k = math.sqrt(chi2_3_ppf(conf))
    ev, evec = np.linalg.eigh(cov)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 36), np.linspace(0, np.pi, 18))
    sph = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], -1)
    E = sph @ (evec * (k * np.sqrt(np.maximum(ev, 0)))).T
    ax.plot_wireframe(E[..., 0], E[..., 1], E[..., 2], color=INK2, linewidth=0.4, alpha=0.5,
                      label=f"{100 * conf:g}% covariance ellipsoid")
    ax.scatter([0], [0], [0], marker="+", s=80, color=INK, label="c_bar")
    if truth_mm is not None:
        t = truth_mm - c_bar
        ax.scatter(*t, marker="x", s=70, color=ACCENT, linewidth=2, label="truth")
    lim = max(np.abs(E).max(), np.abs(D).max(),
              0 if truth_mm is None else np.abs(truth_mm - c_bar).max()) * 1.05
    ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X - c_bar (mm)"), ax.set_ylabel("Y - c_bar (mm)"), ax.set_zlabel("Z - c_bar (mm)")
    ax.set_title("reference frame (shot 0 colour camera)", fontsize=9, color=INK2)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # ---- histograms: camera-local (top) and reference frame (bottom) -------
    R = C - c_bar
    span = max(np.abs(R_cam).max(), np.abs(R).max()) * 1.05 or 1.0
    bins = np.linspace(-span, span, max(13, min(41, len(R) + 1)))
    for row, (data, names, what) in enumerate((
            (R_cam, ("x lateral", "y lateral", "z depth"), "camera-local"),
            (R, ("X", "Y", "Z"), "reference frame"))):
        for j in range(3):
            a = fig.add_subplot(gs[row, 2 + j])
            a.set_facecolor(SURFACE)
            a.hist(data[:, j], bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=1.0)
            a.axvline(0, color=INK2, linewidth=0.8)
            s = data[:, j].std(ddof=1)
            a.set_title(f"{what}  {names[j]}\nstd {s:.2f} mm", fontsize=8.5)
            a.set_xlim(-span, span)
            a.grid(axis="y", color=INK2, alpha=0.15, linewidth=0.6)
            if row == 1:
                a.set_xlabel("residual (mm)")
            if j == 0:
                a.set_ylabel("shots")

    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Spread of a tag's centre across registered shots",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("stitch", help="tag_stitch.py output folder (e.g. <capture>/stitch)")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--tag-id", type=int, help="any tag (spread may be partly in-sample)")
    which.add_argument("--holdout-tag", type=int,
                       help="a tag no registration edge used (true generalisation)")
    ap.add_argument("--center", choices=["plane", "corners", "pnp"], default="plane",
                    help="plane-fit centre, mean of the 4 local-depth corners, or colour-only PnP")
    ap.add_argument("--truth", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None,
                    help="calibrated tag centre, mm, in the reference shot's colour frame")
    ap.add_argument("--min-shots", type=int, default=3)
    ap.add_argument("--aniso-ratio", type=float, default=2.0,
                    help="sz / mean(sx, sy) above which 'sz >> sx, sy' is declared")
    ap.add_argument("--ellipsoid-conf", type=float, default=0.95,
                    help="probability mass of the drawn covariance ellipsoid")
    ap.add_argument("--show", action="store_true", help="open the figure window as well")
    args = ap.parse_args()

    stitch = Path(args.stitch)
    if not stitch.is_dir():
        sys.exit(f"stitch folder not found: {stitch}")
    if not 0 < args.ellipsoid_conf < 1:
        ap.error("--ellipsoid-conf must be in (0, 1)")
    tag = args.holdout_tag if args.holdout_tag is not None else args.tag_id
    pairs = load_json(stitch / "pairs.json")
    used_by = edges_using(pairs, tag)
    if args.holdout_tag is not None and used_by:
        sys.exit(f"tag {tag} is not held out: edges {used_by} registered with it. "
                 f"Re-run tag_stitch.py --exclude-tags {tag} (optionally --out stitch_holdout{tag}).")

    poses, rows, skipped = gather(stitch, tag, args.center)
    for idx, why in skipped:
        print(f"[SKIP] shot {idx:03d}: {why}")
    if len(rows) < args.min_shots:
        sys.exit(f"tag {tag} has a usable {args.center} centre in only {len(rows)} shots "
                 f"(need >= {args.min_shots})")
    truth = np.asarray(args.truth, float) if args.truth else None
    stats, C, R, R_cam, cov = spread(rows, truth, args.aniso_ratio)

    stats.update(tag_id=tag, center_estimator=args.center,
                 holdout=args.holdout_tag is not None, edges_that_used_tag=used_by,
                 registration_method=poses.get("method"),
                 reference_index=poses.get("reference_index"),
                 shots=[r["index"] for r in rows])
    stem = stitch / f"spread_tag{tag}_{args.center}"
    Path(f"{stem}.json").write_text(json.dumps(stats, indent=2))
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "true_deg", "cx_mm", "cy_mm", "cz_mm", "rX_mm", "rY_mm", "rZ_mm",
                    "rx_cam_mm", "ry_cam_mm", "rz_cam_mm", "radius_mm", "tag_depth_m",
                    "obliquity_deg", "plane_rms_mm"])
        for r, c, res, rc in zip(rows, C, R, R_cam):
            w.writerow([r["index"], f"{r['true_deg']:.4f}", *np.round(c, 4), *np.round(res, 4),
                        *np.round(rc, 4), round(float(np.linalg.norm(res)), 4),
                        round(r["depth_m"], 4), r["obliquity_deg"], r["plane_rms_mm"]])

    sc = stats["camera_local_std_mm"]
    kind = "HOLD-OUT" if stats["holdout"] else "in-sample" if used_by else "not used by any edge"
    print(f"\ntag {tag}  ({kind}, centre = {args.center}, {len(rows)} shots, "
          f"poses by {stats['registration_method']})")
    if used_by and not stats["holdout"]:
        print(f"  used by {len(used_by)} edges -> spread is optimistic; use --holdout-tag for "
              f"generalisation")
    print(f"  c_bar                     {np.round(stats['c_bar_mm'], 2)} mm (reference frame)")
    print(f"  camera-local std          sx {sc['sx']:.3f}  sy {sc['sy']:.3f}  sz {sc['sz']:.3f} mm")
    print(f"  anisotropy sz/lateral     {stats['anisotropy_sz_over_lateral']:.2f}  -> "
          f"{'sz >> sx, sy holds' if stats['anisotropy_holds'] else 'NOT depth-dominated'} "
          f"(threshold {args.aniso_ratio:g})")
    rs = stats["ref_frame_std_mm"]
    print(f"  reference-frame std       sX {rs['sX']:.3f}  sY {rs['sY']:.3f}  sZ {rs['sZ']:.3f} mm")
    print(f"  principal std             {np.round(stats['principal_std_mm'], 3)} mm")
    print(f"  RMS radius                {stats['rms_radius_mm']:.3f} mm")
    print(f"  95th-pct radius           {stats['p95_radius_mm']:.3f} mm")
    if truth is not None:
        b = stats["bias"]
        print(f"  bias (c_bar - truth)      {np.round(b['bias_mm'], 3)} mm, "
              f"|bias| {b['bias_norm_mm']:.3f} mm, z-scores {np.round(b['bias_z_score'], 1)}")
        print(f"  total RMS error           {b['total_rms_error_mm']:.3f} mm "
              f"(= sqrt(|bias|^2 + RMS radius^2))")
    if len(rows) < 8:
        print(f"  [NOTE] n = {len(rows)}: the std estimates carry roughly "
              f"+-{100 / math.sqrt(2 * (len(rows) - 1)):.0f}% relative uncertainty")

    figure(f"{stem}.png",
           f"Tag {tag} centre spread ({kind}, {args.center} centre, n={len(rows)})  -  "
           f"RMS radius {stats['rms_radius_mm']:.2f} mm, 95% radius {stats['p95_radius_mm']:.2f} mm",
           C, np.asarray(stats["c_bar_mm"]), R_cam, cov, args.ellipsoid_conf, truth, args.show)
    print(f"\n[OUT ] {stem}.json / .csv / .png")


if __name__ == "__main__":
    main()
