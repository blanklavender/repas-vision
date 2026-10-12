#!/usr/bin/env python3
"""
Flora Inspection Station -- per-stop burst averaging + confidence for the Femto Bolt.

Reads the raw depth + IR bursts that sweep_capture.py saves at every turntable
stop and, per stop, turns N frames of a static pose into one averaged depth map
with an uncertainty map, a weight map and a validity mask. The raw frames are
only read, never modified.

    python average_confidence.py captures/plant_01
    python average_confidence.py captures/plant_01 --s-max 4 --a-min 0.05 --overwrite

I/O contract
------------
in   <capture>/camera_params.json          (sweep_capture.py) depth fx, depth scale
     <capture>/manifest.csv                column stop_dir  (else: glob stops/*/raw)
     <capture>/stops/<stem>/raw/depth_KK.png   uint16, native depth grid, raw units
     <capture>/stops/<stem>/raw/ir_KK.png      uint16 (or uint8) IR / active brightness

out  <capture>/stops/<stem>/avg/           everything in the NATIVE depth pixel grid
     depth_mean_mm.npy   float32  mean depth over the temporally-valid frames, mm;
                                  0 where the pixel never had enough depth
     depth_std_mm.npy    float32  per-pixel temporal std (ddof=1), mm; NaN if < 2 frames
     depth_sem_mm.npy    float32  standard error of the mean = std / sqrt(n)
     n_valid.png         uint8    frames that contributed to the mean
     ir_mean.npy         float32  mean IR over all N frames (raw counts)
     ir_mean.png         uint16   same, rounded
     amp_norm.npy        float32  amplitude confidence = ir_mean / a_ref, clipped to [0, 1]
     weight.npy          float32  w in [0, 1]; 0 wherever invalid
     weight.png          uint8    w * 255, for eyeballing
     valid_mask.png      uint8    255 = valid, 0 = invalid
     invalid_reason.png  uint8    bitmask: 1 too few depth frames, 2 low amplitude,
                                  4 temporal std > s_max, 8 depth discontinuity
     depth_mean_vis.png  uint8    colour-mapped mean depth, invalid = black
     stats.json          parameters + per-reason counts for this stop
     <capture>/average_confidence.json    run parameters + per-stop summary

Definitions (shared with tag_stitch.py)
---------------------------------------
    per frame      a depth sample is usable if z_min <= z <= z_max and it is not a
                   temporal outlier: |z - median_t(z)| <= k * max(1.4826 * MAD_t, floor)
    mean depth     mean of the usable samples (random noise shrinks ~ 1/sqrt(n))
    uncertainty    temporal std of the usable samples
    amplitude      a = mean IR / a_ref.  a_ref (raw counts) is fixed for the whole
                   capture -- by default the median over stops of each stop's 99th
                   percentile IR -- so a weight means the same thing at every stop
    INVALID if     n < min_valid_frac * N   OR   a < a_min   OR   std > s_max
                   OR the pixel sits on a depth discontinuity
    discontinuity  a neighbour (8-connected) differs by more than
                       max(edge_floor_mm, z * tan(max_incidence) * step_px / fx)
                   i.e. more than a surface seen at <= max_incidence could explain,
                   or (default) the neighbour has no depth at all. Flagged pixels
                   are dilated by edge_dilate px. Removes flying pixels on both sides.
    weight         w = w_std * w_amp on valid pixels, 0 elsewhere
                   w_std = 1 / (1 + (std / s_half)^2)          -> 0.5 at std = s_half
                   w_amp = smoothstep((a - a_min) / (a_full - a_min))  -> 1 at a >= a_full

Assumptions: the IR frame shares the depth sensor's pixel grid (checked on
load: shapes must match); depth raw units x depth_scale_mm_per_unit = mm.
"""

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

DEPTH_GLOB = "depth_*.png"
IR_GLOB = "ir_*.png"
REASON_FEW, REASON_AMP, REASON_STD, REASON_EDGE = 1, 2, 4, 8


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_camera_params(capture):
    path = capture / "camera_params.json"
    if not path.is_file():
        sys.exit(f"missing {path} -- was this capture made with the burst-capable "
                 f"sweep_capture.py?")
    cam = json.loads(path.read_text())
    try:
        fx = float(cam["depth_intrinsic"]["fx"])
        scale = float(cam["depth_scale_mm_per_unit"])
    except (KeyError, TypeError, ValueError) as e:
        sys.exit(f"{path}: missing depth_intrinsic.fx / depth_scale_mm_per_unit ({e})")
    return fx, scale


def find_stops(capture):
    """Stop dirs in manifest order; fall back to a glob if the manifest has none."""
    manifest = capture / "manifest.csv"
    stops = []
    if manifest.is_file():
        with open(manifest, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("stop_dir"):
                    stops.append(capture / row["stop_dir"])
    if not stops:
        stops = sorted(p.parent for p in capture.glob("stops/*/raw"))
    if not stops:
        sys.exit(f"no stops found in {capture} (no manifest stop_dir column and "
                 f"no stops/*/raw folders)")
    for s in stops:
        if not (s / "raw").is_dir():
            sys.exit(f"missing raw burst folder {s / 'raw'}")
    return stops


def read_stack(files, what):
    imgs = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            sys.exit(f"could not read {what} frame {f}")
        if img.ndim != 2:
            sys.exit(f"{f}: expected a single-channel {what} image, got shape {img.shape}")
        imgs.append(img)
    if len({im.shape for im in imgs}) != 1:
        sys.exit(f"{what} frames in {files[0].parent} do not share one size")
    return np.stack(imgs).astype(np.float32)


def load_burst(stop, depth_scale):
    raw = stop / "raw"
    dfiles, ifiles = sorted(raw.glob(DEPTH_GLOB)), sorted(raw.glob(IR_GLOB))
    if not dfiles:
        sys.exit(f"no {DEPTH_GLOB} in {raw}")
    if len(dfiles) != len(ifiles):
        sys.exit(f"{raw}: {len(dfiles)} depth frames but {len(ifiles)} IR frames")
    D = read_stack(dfiles, "depth") * depth_scale          # mm
    I = read_stack(ifiles, "IR")
    if D.shape != I.shape:
        sys.exit(f"{raw}: depth {D.shape[1:]} and IR {I.shape[1:]} grids differ")
    return D, I


# --------------------------------------------------------------------------
# the shared confidence + averaging definition
# --------------------------------------------------------------------------

def temporal_stats(D, args):
    """D (N,H,W) mm -> mean, std, sem, n  over the temporally-consistent samples."""
    N = D.shape[0]
    V = (D > 0) & (D >= args.z_min) & (D <= args.z_max)
    Dn = np.where(V, D, np.nan)
    with warnings.catch_warnings():            # all-NaN pixels are handled below
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(Dn, axis=0)
        mad = np.nanmedian(np.abs(Dn - med), axis=0)
    sig = np.maximum(1.4826 * np.nan_to_num(mad), args.outlier_floor_mm)
    V &= np.abs(np.nan_to_num(Dn - med, nan=np.inf)) <= args.outlier_k * sig

    n = V.sum(0).astype(np.uint16)
    nf = n.astype(np.float64)
    s1 = np.where(V, D, 0).sum(0, dtype=np.float64)
    mean = np.where(n > 0, s1 / np.maximum(nf, 1), 0.0)
    ss = np.where(V, (D - mean) ** 2, 0).sum(0, dtype=np.float64)
    std = np.where(n >= 2, np.sqrt(ss / np.maximum(nf - 1, 1)), np.nan)
    sem = std / np.sqrt(np.maximum(nf, 1))
    enough = n >= max(1, int(np.ceil(args.min_valid_frac * N)))
    mean = np.where(enough, mean, 0.0)
    return mean.astype(np.float32), std.astype(np.float32), sem.astype(np.float32), n, enough


def discontinuities(z, has, fx, args):
    """8-neighbour jump test on the mean depth. z mm, has = pixel has depth."""
    H, W = z.shape
    tan_inc = np.tan(np.radians(args.edge_max_incidence))
    edge = np.zeros((H, W), bool)
    zp = np.pad(z, 1, mode="edge")
    hp = np.pad(has, 1, mode="constant", constant_values=False)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            zn = zp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            hn = hp[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            step = np.hypot(dx, dy)
            thr = np.maximum(args.edge_floor_mm, z * tan_inc * step / fx)
            edge |= has & hn & (np.abs(z - zn) > thr)
            if not args.keep_hole_borders:
                edge |= has & ~hn
    if args.edge_dilate > 0:
        k = np.ones((2 * args.edge_dilate + 1,) * 2, np.uint8)
        edge = cv2.dilate(edge.astype(np.uint8), k).astype(bool)
    return edge & has


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def confidence(mean, std, enough, ir_mean, a_ref, fx, args):
    amp = np.clip(ir_mean / a_ref, 0.0, 1.0).astype(np.float32)
    edge = discontinuities(mean, enough, fx, args)

    reason = np.zeros(mean.shape, np.uint8)
    reason[~enough] |= REASON_FEW
    reason[amp < args.a_min] |= REASON_AMP
    reason[enough & ~(std <= args.s_max)] |= REASON_STD     # NaN std counts as too high
    reason[edge] |= REASON_EDGE
    valid = reason == 0

    w_std = 1.0 / (1.0 + (np.nan_to_num(std, nan=np.inf) / args.s_half) ** 2)
    w_amp = smoothstep((amp - args.a_min) / (args.a_full - args.a_min))
    weight = np.where(valid, w_std * w_amp, 0.0).astype(np.float32)
    return amp, weight, valid, reason


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def depth_vis(mean, valid):
    out = np.zeros(mean.shape + (3,), np.uint8)
    if valid.any():
        lo, hi = np.percentile(mean[valid], [1, 99])
        g = np.clip((mean - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        out = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
        out[~valid] = 0
    return out


def write_stop(avg, mean, std, sem, n, ir_mean, amp, weight, valid, reason):
    avg.mkdir(parents=True, exist_ok=True)
    np.save(avg / "depth_mean_mm.npy", mean)
    np.save(avg / "depth_std_mm.npy", std)
    np.save(avg / "depth_sem_mm.npy", sem)
    np.save(avg / "ir_mean.npy", ir_mean.astype(np.float32))
    np.save(avg / "amp_norm.npy", amp)
    np.save(avg / "weight.npy", weight)
    for name, img in (("n_valid.png", np.clip(n, 0, 255).astype(np.uint8)),
                      ("ir_mean.png", np.clip(np.round(ir_mean), 0, 65535).astype(np.uint16)),
                      ("weight.png", np.round(weight * 255).astype(np.uint8)),
                      ("valid_mask.png", valid.astype(np.uint8) * 255),
                      ("invalid_reason.png", reason),
                      ("depth_mean_vis.png", depth_vis(mean, valid))):
        if not cv2.imwrite(str(avg / name), img):
            sys.exit(f"could not write {avg / name}")


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Average each stop's raw depth+IR burst into depth / weight / validity maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("capture", help="capture folder written by sweep_capture.py")
    p.add_argument("--z-min", type=float, default=150.0, help="mm")
    p.add_argument("--z-max", type=float, default=1500.0, help="mm")
    p.add_argument("--outlier-k", type=float, default=3.5,
                   help="temporal outlier gate, in robust sigmas (MAD) from the median")
    p.add_argument("--outlier-floor-mm", type=float, default=1.0,
                   help="lower bound on the robust sigma (depth is quantised to 1 mm)")
    p.add_argument("--min-valid-frac", type=float, default=0.6,
                   help="fraction of the N frames a pixel must have depth in")
    p.add_argument("--a-ref", type=float, default=None,
                   help="IR counts that map to amplitude 1.0 (default: auto, per capture)")
    p.add_argument("--a-min", type=float, default=0.03,
                   help="normalised amplitude below which a pixel is invalid")
    p.add_argument("--a-full", type=float, default=0.25,
                   help="normalised amplitude at and above which w_amp = 1")
    p.add_argument("--s-max", type=float, default=5.0,
                   help="mm; temporal std above which a pixel is invalid")
    p.add_argument("--s-half", type=float, default=1.5,
                   help="mm; temporal std at which w_std = 0.5")
    p.add_argument("--edge-max-incidence", type=float, default=80.0,
                   help="deg; steepest surface a neighbour jump may come from")
    p.add_argument("--edge-floor-mm", type=float, default=3.0,
                   help="mm; jumps below this never count as discontinuities")
    p.add_argument("--edge-dilate", type=int, default=1,
                   help="px; grow the discontinuity mask by this much")
    p.add_argument("--keep-hole-borders", action="store_true",
                   help="do not treat pixels next to missing depth as discontinuities")
    p.add_argument("--overwrite", action="store_true", help="replace existing avg/ folders")
    args = p.parse_args()

    if not 0 <= args.a_min < args.a_full <= 1:
        p.error("need 0 <= --a-min < --a-full <= 1")
    if args.s_half <= 0 or args.s_max <= 0:
        p.error("--s-half and --s-max must be > 0")
    if not 0 < args.min_valid_frac <= 1:
        p.error("--min-valid-frac must be in (0, 1]")

    capture = Path(args.capture)
    if not capture.is_dir():
        sys.exit(f"capture folder not found: {capture}")
    fx, depth_scale = load_camera_params(capture)
    stops = find_stops(capture)
    if not args.overwrite:
        done = [s for s in stops if (s / "avg").exists()]
        if done:
            sys.exit(f"{len(done)} stops already have avg/ (e.g. {done[0]}); "
                     f"pass --overwrite to replace them")

    # ---- pass 1: temporal stats + IR, and the capture-wide amplitude reference
    print(f"{len(stops)} stops in {capture}  (depth fx {fx:.1f} px, "
          f"{depth_scale:g} mm/unit)")
    cache = []
    for s in stops:
        D, I = load_burst(s, depth_scale)
        mean, std, sem, n, enough = temporal_stats(D, args)
        ir_mean = I.mean(0)
        p99 = float(np.percentile(ir_mean[enough], 99)) if enough.any() else np.nan
        cache.append((s, D.shape[0], mean, std, sem, n, enough, ir_mean, p99))

    if args.a_ref is not None:
        a_ref, a_src = float(args.a_ref), "--a-ref"
    else:
        p99s = np.array([c[-1] for c in cache])
        if not np.isfinite(p99s).any():
            sys.exit("no stop has any valid depth -- check --z-min/--z-max and the raw frames")
        a_ref, a_src = float(np.nanmedian(p99s)), "median over stops of the per-stop 99th pct IR"
    if a_ref <= 0:
        sys.exit(f"amplitude reference {a_ref} <= 0; pass --a-ref")
    print(f"amplitude reference a_ref = {a_ref:.1f} IR counts ({a_src})\n")

    # ---- pass 2: confidence, weights, write
    summary = []
    print("  stop                     N   valid%   few%   amp%   std%  edge%   "
          "med_std_mm  mean_w")
    for s, N, mean, std, sem, n, enough, ir_mean, _ in cache:
        amp, weight, valid, reason = confidence(mean, std, enough, ir_mean, a_ref, fx, args)
        write_stop(s / "avg", mean, std, sem, n, ir_mean, amp, weight, valid, reason)

        tot = reason.size
        frac = {k: float(np.count_nonzero(reason & b)) / tot
                for k, b in (("few", REASON_FEW), ("amp", REASON_AMP),
                             ("std", REASON_STD), ("edge", REASON_EDGE))}
        med_std = float(np.median(std[valid])) if valid.any() else float("nan")
        stats = {"stop": s.name, "frames": int(N), "a_ref": a_ref,
                 "valid_frac": float(valid.mean()), "invalid_frac_by_reason": frac,
                 "median_std_valid_mm": med_std,
                 "mean_weight_valid": float(weight[valid].mean()) if valid.any() else 0.0}
        (s / "avg" / "stats.json").write_text(json.dumps(stats, indent=2))
        summary.append(stats)
        print(f"  {s.name:<22} {N:3d}  {100 * valid.mean():6.1f}  {100 * frac['few']:5.1f}  "
              f"{100 * frac['amp']:5.1f}  {100 * frac['std']:5.1f}  {100 * frac['edge']:5.1f}   "
              f"{med_std:9.3f}  {stats['mean_weight_valid']:.3f}")
        if N < 3:
            print(f"    [WARN] only {N} frames -- the std map is not meaningful")

    (capture / "average_confidence.json").write_text(json.dumps({
        "params": vars(args), "a_ref": a_ref, "a_ref_source": a_src,
        "depth_fx_px": fx, "depth_scale_mm_per_unit": depth_scale,
        "reason_bits": {"too_few_frames": REASON_FEW, "low_amplitude": REASON_AMP,
                        "high_std": REASON_STD, "discontinuity": REASON_EDGE},
        "stops": summary}, indent=2))
    print(f"\nwrote avg/ for {len(summary)} stops; summary -> "
          f"{capture / 'average_confidence.json'}")


if __name__ == "__main__":
    main()
