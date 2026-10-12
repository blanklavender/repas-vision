#!/usr/bin/env python3
"""
tag_graph_register.py  --  register all shots of a turntable sweep using EVERY shot pair
that shares tags, not only neighbours.

Three solutions are computed from the same tag measurements (colour-only PnP tag poses)
and compared against the logged turntable angles:

  chain      adjacent pairs only (pose-averaged), chained, loop error spread evenly.
             = tag_pose_register.py; the baseline and the starting point of the others.
  pairgraph  every pair (i, j) up to --max-gap shots apart that shares tags gets a
             pose-averaged transform T_ij plus a score; pairs passing the acceptance
             test become links of a pose graph, and all shot poses are solved at once
             (robust weighted least squares, shot 0 fixed).
  tagmap     each tag has ONE unknown pose on the pot; all shot poses and tag poses are
             solved together from every tag sighting (robust least squares). Uses the
             same information as all pairs, but counts each sighting exactly once.

Pair score / acceptance (pairgraph):
    n_tags    shared usable tags (>= --min-tags, default 2)
    spread    RMS angle between the per-tag rotations and their average (tags agreeing)
    dev       |pair rotation - logged turntable rotation for that gap|
    accepted  if spread <= --pair-max-spread and dev <= --pair-max-dev
    weight    n_tags / (1 + (spread / sigma_rot)^2)

    python tag_graph_register.py ../../captures/femto_bolt/flora_unit/yellow_pot_09-30_deg_10_speed_5 \\
        --depth-correction ../../captures/femto_bolt/flora_unit/yellow_pot_09-30_deg_10_speed_5/stitch_refined/depth_correction.json

I/O contract
------------
in   <capture>: camera_params.json, manifest.csv, color_*.png, stops/*/avg/, stops/*/mask.png
     --depth-correction  depth_correction.json from tag_stitch.py --refine-depth (optional)
out  <capture>/<--out, default tag_graph>/
        pairs.csv                       every candidate pair: tags, spread, dev, accepted, weight
        <method>/poses.json             T_shot_to_ref per shot (tag_stitch.py format), so
        <method>/tags_per_shot.json     view_tag_corners.py / spread_analysis.py can read it
        <method>/fused.ply              masked clouds of all shots in shot 0's frame
        summary.json                    per-method errors vs turntable + solver residuals
Needs scipy (present in the project venv).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from pupil_apriltags import Detector
from scipy.optimize import least_squares

import tag_stitch as ts
from tag_pose_register import pose_average, tag_pose


# ---------------------------------------------------------------------------
# small SE(3) helpers
# ---------------------------------------------------------------------------
def rv(T):
    return cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))[0].ravel()


def T_of(x):
    return ts.make_T(cv2.Rodrigues(np.asarray(x[:3], float))[0], x[3:6])


def x_of(T):
    return np.r_[rv(T), T[:3, 3]]


def inv(T):
    return np.linalg.inv(T)


# ---------------------------------------------------------------------------
# pair transform from shared tags (pose averaging, as tag_pose_register.py)
# ---------------------------------------------------------------------------
def pair_transform(ta, tb, ids):
    Ts = [tag_pose(tb[i]) @ inv(tag_pose(ta[i])) for i in ids]
    R = pose_average(Ts, np.full(len(ids), 1.0 / len(ids)))
    ca = np.array([ta[i]["center_pnp"] for i in ids])
    cb = np.array([tb[i]["center_pnp"] for i in ids])
    T = ts.make_T(R, (cb - ca @ R.T).mean(0))
    spread = float(np.sqrt(np.mean([ts.rot_deg(R.T @ Ti[:3, :3]) ** 2 for Ti in Ts])))
    return T, spread


# ---------------------------------------------------------------------------
# solvers (vectorised; unknowns are small corrections d around the starting poses:
#          S_i = exp(d_i) @ S_i0, so large rotations never enter the parametrisation)
# ---------------------------------------------------------------------------
def exp_so3(w):
    """(N,3) rotation vectors -> (N,3,3)."""
    th = np.linalg.norm(w, axis=1)[:, None, None]
    K = np.zeros((len(w), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = w[:, 2], -w[:, 1], w[:, 0]
    small = th < 1e-9
    a = np.where(small, 1.0, np.sin(th) / np.where(small, 1.0, th))
    b = np.where(small, 0.5, (1 - np.cos(th)) / np.where(small, 1.0, th ** 2))
    return np.eye(3) + a * K + b * (K @ K)


def log_so3(R):
    """(N,3,3) rotations near identity -> (N,3) rotation vectors."""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.sin(th)
    f = np.where(th < 1e-6, 0.5, th / (2 * np.where(th < 1e-6, 1.0, s)))
    return v * f[:, None]


def compose(d, R0, t0):
    """poses exp(d) @ [R0|t0] for stacked d (N,6)."""
    dR = exp_so3(d[:, :3])
    return dR @ R0, np.einsum("nij,nj->ni", dR, t0) + d[:, 3:]


def solve_pairgraph(n, links, S0, args):
    """links: (i, j, T_ij, weight). Unknowns: corrections for shots 1..n-1 (shot 0 fixed)."""
    sr, st = np.radians(args.sigma_rot), args.sigma_t_mm / 1000.0
    R0 = np.stack([S[:3, :3] for S in S0]); t0 = np.stack([S[:3, 3] for S in S0])
    I = np.array([l[0] for l in links]); Jn = np.array([l[1] for l in links])
    Ti = np.stack([inv(l[2]) for l in links])                    # inv(T_ij)
    W = np.sqrt(np.array([l[3] for l in links]))[:, None]

    def poses(x):
        d = np.vstack([np.zeros(6), x.reshape(-1, 6)])
        return compose(d, R0, t0)

    def resid(x):
        R, t = poses(x)
        # E = inv(T_ij) @ inv(S_j) @ S_i
        RjT = np.transpose(R[Jn], (0, 2, 1))
        Rm = RjT @ R[I]
        tm = np.einsum("nij,nj->ni", RjT, t[I] - t[Jn])
        RE = Ti[:, :3, :3] @ Rm
        tE = np.einsum("nij,nj->ni", Ti[:, :3, :3], tm) + Ti[:, :3, 3]
        return (W * np.hstack([log_so3(RE) / sr, tE / st])).ravel()

    sol = least_squares(resid, np.zeros(6 * (n - 1)), loss="huber", f_scale=args.huber)
    R, t = poses(sol.x)
    return [ts.make_T(R[k], t[k]) for k in range(n)], sol


def solve_tagmap(n, obs, S0, args):
    """obs: (shot, tag, O) with O = tag->camera pose. Unknowns: shots 1..n-1 + one pose per tag."""
    tag_ids = sorted({k for _, k, _ in obs})
    col = {k: c for c, k in enumerate(tag_ids)}
    sr = np.radians(args.sigma_rot)
    slat, sdep = args.sigma_lat_mm / 1000.0, args.sigma_depth_mm / 1000.0
    G0 = []
    for k in tag_ids:                       # start: average of S_i @ O over the tag's sightings
        Gs = [S0[i] @ O for i, kk, O in obs if kk == k]
        R = pose_average(Gs, np.full(len(Gs), 1.0 / len(Gs)))
        G0.append(ts.make_T(R, np.mean([G[:3, 3] for G in Gs], 0)))
    RS0 = np.stack([S[:3, :3] for S in S0]); tS0 = np.stack([S[:3, 3] for S in S0])
    RG0 = np.stack([G[:3, :3] for G in G0]); tG0 = np.stack([G[:3, 3] for G in G0])
    I = np.array([o[0] for o in obs]); K = np.array([col[o[1]] for o in obs])
    RO = np.stack([o[2][:3, :3] for o in obs]); tO = np.stack([o[2][:3, 3] for o in obs])
    u = tO / np.linalg.norm(tO, axis=1, keepdims=True)                 # viewing ray per sighting
    e1 = np.cross(u, [0.0, 1.0, 0.0]); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(u, e1)
    ns = 6 * (n - 1)

    def unpack(x):
        dS = np.vstack([np.zeros(6), x[:ns].reshape(-1, 6)])
        return compose(dS, RS0, tS0), compose(x[ns:].reshape(-1, 6), RG0, tG0)

    def resid(x):
        (RS, tS), (RG, tG) = unpack(x)
        RiT = np.transpose(RS[I], (0, 2, 1))
        RP = RiT @ RG[K]                                             # predicted tag pose in camera i
        tP = np.einsum("nij,nj->ni", RiT, tG[K] - tS[I])
        rot = log_so3(np.transpose(RO, (0, 2, 1)) @ RP)
        d = tP - tO
        return np.hstack([rot / sr, (np.sum(d * e1, 1) / slat)[:, None],
                          (np.sum(d * e2, 1) / slat)[:, None], (np.sum(d * u, 1) / sdep)[:, None]]).ravel()

    sol = least_squares(resid, np.zeros(ns + 6 * len(tag_ids)), loss="huber", f_scale=args.huber)
    (RS, tS), (RG, tG) = unpack(sol.x)
    S = [ts.make_T(RS[k], tS[k]) for k in range(n)]
    return S, {k: ts.make_T(RG[c], tG[c]) for k, c in col.items()}, sol


# ---------------------------------------------------------------------------
# evaluation + output
# ---------------------------------------------------------------------------
def turntable_errors(S, shots, axis):
    """signed rotation of each pose about the turntable axis vs the logged angle (deg)."""
    def ang(T):
        r = rv(T)
        return float(np.degrees(np.linalg.norm(r)) * (np.sign(r @ axis) or 1.0))
    sgn = np.sign(ang(S[1])) or 1.0
    return [((sgn * ang(T) - (s["true_deg"] - shots[0]["true_deg"]) + 180.0) % 360.0) - 180.0
            for T, s in zip(S, shots)]


def write_method(dirpath, name, S, shots, tags_json, args, extra):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "poses.json").write_text(json.dumps({
        "convention": "X_ref = T_shot_to_ref @ X_shot (metres, colour-camera frames)",
        "reference_index": shots[0]["index"], "method": f"tag_graph_register:{name}",
        "loop": {}, "registration_tags_excluded": [],
        "shots": [dict(index=s["index"], true_deg=s["true_deg"], T_shot_to_ref=T.tolist())
                  for s, T in zip(shots, S)], **extra}, indent=2))
    (dirpath / "tags_per_shot.json").write_text(tags_json)
    if args.fused_voxel > 0:
        Pm, Cm, _, _ = ts.voxel_weighted(np.vstack([ts.apply(T, s["P"]) for s, T in zip(shots, S)]),
                                         np.vstack([s["C"] for s in shots]),
                                         np.concatenate([s["w"] for s in shots]), args.fused_voxel)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(Cm, 0, 1))
        o3d.io.write_point_cloud(str(dirpath / "fused.ply"), pcd)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="All-pairs tag registration: chain vs pair graph vs tag map",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--indices", default=None, help="manifest indices (default: all)")
    ap.add_argument("--depth-correction", default=None, help="depth_correction.json (tag_stitch --refine-depth)")
    ap.add_argument("--max-obliquity", type=float, default=55.0)
    ap.add_argument("--tag-size-mm", type=float, default=24.0)
    ap.add_argument("--max-gap", type=int, default=6, help="pairs up to this many shots apart")
    ap.add_argument("--min-tags", type=int, default=2, help="shared tags needed for a pair")
    ap.add_argument("--pair-max-spread", type=float, default=1.5,
                    help="deg; reject a pair whose tags disagree by more (RMS)")
    ap.add_argument("--pair-max-dev", type=float, default=3.0,
                    help="deg; reject a pair whose rotation is this far from the logged gap angle")
    ap.add_argument("--sigma-rot", type=float, default=0.5, help="deg; rotation noise scale")
    ap.add_argument("--sigma-t-mm", type=float, default=2.0, help="pair translation noise scale")
    ap.add_argument("--sigma-lat-mm", type=float, default=0.5, help="tagmap: sideways position noise")
    ap.add_argument("--sigma-depth-mm", type=float, default=3.0, help="tagmap: along-ray (PnP distance) noise")
    ap.add_argument("--huber", type=float, default=2.0, help="robust loss threshold, in sigmas")
    ap.add_argument("--fused-voxel", type=float, default=0.001, help="m; 0 = no fused clouds")
    ap.add_argument("--out", default="tag_graph")
    ap.set_defaults(tag_ids=set(), tag_erode=3, tag_min_pts=30, plane_inlier_mm=1.5, ransac_iters=200,
                    corner_win=6.0, tag_side_tol=0.10, w_min=0.2, mask=True, mask_erode=3)
    args = ap.parse_args()

    capture = Path(args.capture)
    cam = ts.load_camera(capture)
    if args.depth_correction:
        f = Path(args.depth_correction)
        if not f.is_file():
            sys.exit(f"--depth-correction: {f} not found")
        cam = ts.corrected_camera(cam, np.asarray(json.loads(f.read_text())["T"]))
    rows = ts.load_manifest(capture, ts.parse_indices(args.indices) if args.indices else None)
    det = Detector(families="tag36h11", quad_decimate=1.0, refine_edges=True)
    rng = np.random.default_rng(0)

    shots = []
    for r in rows:
        col = cv2.imread(str(capture / r["color"]))
        P, C, w, uv, inb = ts.shot_cloud(capture, r, cam, args, col)
        tags = ts.measure_tags(cv2.cvtColor(col, cv2.COLOR_BGR2GRAY), P, w, uv, inb, cam, args, det, rng)
        m = ts.in_mask(ts.load_mask(capture, r, args), uv, inb)
        shots.append(dict(index=r["index"], true_deg=r["true_deg"], tags=tags, P=P[m], C=C[m], w=w[m]))
    n = len(shots)
    closed = ts.is_closed_loop(rows)
    usable = [{k for k, t in s["tags"].items() if t.get("ok_pnp")} for s in shots]
    step = float(np.median(np.diff([s["true_deg"] for s in shots])))

    # ---- candidate pairs ----------------------------------------------------
    cand = []
    for i in range(n):
        for g in range(1, args.max_gap + 1):
            j = i + g
            if j >= n:
                if not closed:
                    break
                j -= n
            if closed and 2 * g == n and i >= j:
                continue
            ids = sorted(usable[i] & usable[j])
            if len(ids) < args.min_tags:
                continue
            T, spread = pair_transform(shots[i]["tags"], shots[j]["tags"], ids)
            logged = shots[j]["true_deg"] - shots[i]["true_deg"] + (360.0 if j < i else 0.0)
            dev = abs(ts.rot_deg(T[:3, :3]) - logged)
            ok = spread <= args.pair_max_spread and dev <= args.pair_max_dev
            w = len(ids) / (1.0 + (spread / args.sigma_rot) ** 2)
            cand.append(dict(i=i, j=j, gap=g, ids=ids, T=T, spread=spread, dev=dev, logged=logged,
                             accepted=ok, weight=w))
    acc = [c for c in cand if c["accepted"]]
    print(f"{n} shots, {'closed loop, ' if closed else ''}step {step:.3f} deg | candidate pairs "
          f"{len(cand)}, accepted {len(acc)} (max gap {args.max_gap})")
    for g in range(1, args.max_gap + 1):
        cg = [c for c in cand if c["gap"] == g]
        if cg:
            print(f"   gap {g}: {sum(c['accepted'] for c in cg):3d}/{len(cg):3d} accepted, "
                  f"median tags {np.median([len(c['ids']) for c in cg]):.0f}, "
                  f"median dev {np.median([c['dev'] for c in cg]):.2f} deg")

    # ---- chain baseline (adjacent + even loop spread) -------------------------
    adj = {(c["i"], c["j"]): c["T"] for c in cand if c["gap"] == 1}
    if any((i, i + 1) not in adj for i in range(n - 1)):
        missing = [f"{i:03d}->{i + 1:03d}" for i in range(n - 1) if (i, i + 1) not in adj]
        sys.exit(f"adjacent pair(s) without a shared tag: {missing}")
    S_chain = [np.eye(4)]
    for i in range(n - 1):
        S_chain.append(S_chain[-1] @ inv(adj[(i, i + 1)]))
    if closed and (n - 1, 0) in adj:
        D = S_chain[-1] @ inv(adj[(n - 1, 0)])
        rD, tD = rv(D), D[:3, 3]
        S_chain = [ts.make_T(cv2.Rodrigues(-(k / n) * rD)[0], -(k / n) * tD) @ S for k, S in enumerate(S_chain)]
    axis = ts.unit(np.mean([ts.unit(rv(c["T"])) for c in acc], 0))

    # ---- solve ----------------------------------------------------------------
    links = [(c["i"], c["j"], c["T"], c["weight"]) for c in acc]
    S_pg, sol_pg = solve_pairgraph(n, links, S_chain, args)
    obs = [(i, k, tag_pose(s["tags"][k])) for i, s in enumerate(shots) for k in sorted(usable[i])]
    S_tm, G_tm, sol_tm = solve_tagmap(n, obs, S_chain, args)

    # ---- report ---------------------------------------------------------------
    out = capture / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pairs.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["src_index", "tgt_index", "gap", "tags", "rot_deg", "logged_deg", "dev_deg",
                     "tag_spread_deg", "accepted", "weight"])
        for c in cand:
            wr.writerow([shots[c["i"]]["index"], shots[c["j"]]["index"], c["gap"], " ".join(map(str, c["ids"])),
                         f"{ts.rot_deg(c['T'][:3, :3]):.3f}", f"{c['logged']:.3f}", f"{c['dev']:.3f}",
                         f"{c['spread']:.3f}", int(c["accepted"]), f"{c['weight']:.3f}"])

    def link_resid(S):
        rr = [ts.rot_deg((inv(T) @ inv(S[j]) @ S[i])[:3, :3]) for i, j, T, _ in links]
        return float(np.sqrt(np.mean(np.square(rr))))

    tags_json = json.dumps({"frame": "each shot's colour-camera frame, metres",
                            "shots": [dict(index=s["index"], true_deg=s["true_deg"],
                                           tags={str(k): v for k, v in s["tags"].items()}) for s in shots]})
    summary = {}
    print(f"\n  method      pose vs turntable (deg)   RMS disagreement with all      per-shot errors (deg)")
    print(f"              median   max    rms        accepted pairs (deg)")
    for name, S, extra in (("chain", S_chain, {}),
                           ("pairgraph", S_pg, dict(solver_cost=float(sol_pg.cost), links=len(links))),
                           ("tagmap", S_tm, dict(solver_cost=float(sol_tm.cost), observations=len(obs),
                                                 tag_poses={str(k): G.tolist() for k, G in G_tm.items()}))):
        e = np.array(turntable_errors(S, shots, axis))
        a = np.abs(e)
        summary[name] = dict(median_deg=float(np.median(a)), max_deg=float(a.max()),
                             rms_deg=float(np.sqrt(np.mean(e ** 2))), link_rms_deg=link_resid(S),
                             per_shot_deg=e.tolist(), **{k: v for k, v in extra.items() if k != "tag_poses"})
        print(f"  {name:<10} {np.median(a):6.2f} {a.max():6.2f} {np.sqrt(np.mean(e ** 2)):6.2f}        "
              f"{link_resid(S):6.2f}                   " + " ".join(f"{x:+.2f}" for x in e))
        write_method(out / name, name, S, shots, tags_json, args, extra)

    # sightings the tag map fit disagrees with most (after the robust fit)
    S, G = S_tm, G_tm
    worst = sorted(((ts.rot_deg((inv(O) @ inv(S[i]) @ G[k])[:3, :3]),
                     np.linalg.norm((inv(S[i]) @ G[k])[:3, 3] - O[:3, 3]) * 1000, shots[i]["index"], k)
                    for i, k, O in obs), reverse=True)[:5]
    print("\n  tagmap: worst tag sightings (rotation deg / position mm, shot, tag): "
          + "; ".join(f"{r:.2f}/{d:.1f} @{s:03d} tag {k}" for r, d, s, k in worst))
    (out / "summary.json").write_text(json.dumps(dict(args={k: (sorted(v) if isinstance(v, set) else v)
                                                            for k, v in vars(args).items()},
                                                      candidate_pairs=len(cand), accepted_pairs=len(acc),
                                                      methods=summary), indent=2))
    print(f"\n[OUT ] {out}  (pairs.csv, summary.json, chain/ pairgraph/ tagmap/ each with poses.json, "
          f"tags_per_shot.json{', fused.ply' if args.fused_voxel > 0 else ''})")


if __name__ == "__main__":
    main()
