#!/usr/bin/env python3
"""
downsample_view.py  --  view the stitched side sweep + top shot either as the raw overlap of
every cleaned shot (no downsampling at all) or downsampled so that structure seen by many
shots survives and noise seen by one or two shots does not.

Modes
    (default)                    RAW overlap: every cleaned shot moved into the reference
                                 frame and drawn as is -- nothing merged, nothing dropped
    --downsample                 SIGNIFICANT points (--method significant, below)
    --downsample --method voxel  plain voxel-grid mean (what stitch_view.py shows), to compare

Significant-point downsampling, a point-cloud version of the usual mesh clean-up (no
triangles are built):
    1. per shot   incidence   PCA normal of each point in its own shot (--normal-mm); points
                              seen more obliquely than --max-incidence-deg are mixed / flying
                              ToF pixels and are dropped; the rest weigh cos(incidence)
    2. all shots  consensus   the number of distinct shots (top shot included) with a point in
                              the same --support-mm cell or one of its 26 neighbours; side
                              points seen by fewer than --min-views shots are noise (the top
                              shot is one view of mostly unseen-from-the-side plant, so it is
                              kept whatever its support)
    3. fuse       voxel grid  one point per --voxel-mm voxel: incidence-weighted mean position
                              and colour (side and top fused separately)
    4. thin       plane fit   every fused point is projected onto the PCA plane of its
                              neighbours within --smooth-mm when that neighbourhood is flat
                              (surface variation < --max-variation): the shell of overlapping
                              shots collapses onto one surface; edges / stems are left alone
                              (~ the paper's boundary smoothing, 0 = off)
    5. components DBSCAN      clusters at --cluster-mm; clusters under --min-cluster points
                              (suspended fragments) and clusters more than --max-gap-mm from
                              the main plant component (background) go (~ removing small
                              components); outer leaves only the top shot sees stay

Colours (--color): rgb (camera colour), shot (one hue per side shot, top shot white: shows
which shots overlap where), views (support of step 2 as a heat map, blue = 1 shot, red =
many: noise is blue, structure is warm).

    python downsample_view.py ../../../captures/femto_bolt/flora_unit/plant_10-16_deg_5_speed_5
    python downsample_view.py <capture> --color views                    # raw, coloured by support
    python downsample_view.py <capture> --downsample                     # significant points
    python downsample_view.py <capture> --downsample --voxel-mm 3 --min-views 6
    python downsample_view.py <capture> --downsample --method voxel --voxel-mm 2
    python downsample_view.py <capture> --downsample --snapshots --no-view

I/O
---
in   the stitch_view.py inputs (registration.json, side/, top/, the pose-transform file)
out  <capture>/top_view_combine/downsample/       (--downsample only, except the snapshot)
        <method>_side.ply, <method>_top.ply, <method>_combined.ply   reference frame, metres
        <method>.json        settings + point count after every stage
        snap_<mode>_<color>.png   with --snapshots: front / side / top renders
"""

import argparse
import json
from datetime import datetime

import cv2
import numpy as np
import open3d as o3d

import stitch_view as sv
import tv_common as tv
from flora_lib import dataset as ds
from flora_lib import geometry as geo
from flora_lib import registration as reg

OUT_DIR = "downsample"
TOP_WHITE = np.array([0.95, 0.95, 0.95])
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_shots(S, args):
    """Every cleaned shot in the reference frame: dict P, C, shot (side index, top = n_side),
    cos_inc (|normal . view ray| in its own camera, or None when not needed)."""
    need_normals = args.downsample and args.method == "significant"
    P, C, shot, cos = [], [], [], []

    def add(pcd, T, k):
        Q = np.asarray(pcd.points)
        if need_normals:
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(args.normal_mm / 1000.0, 30))
            ray = Q / np.linalg.norm(Q, axis=1, keepdims=True)          # camera at the origin
            cos.append(np.abs((np.asarray(pcd.normals) * ray).sum(1)))
        P.append(geo.apply(T, Q))
        C.append(np.asarray(pcd.colors))
        shot.append(np.full(len(Q), k, np.int32))

    for k, (f, T) in enumerate(zip(S["frames"], S["poses"])):
        add(o3d.io.read_point_cloud(str(ds.clean_cloud_path(S["side"], f["stem"]))), T, k)
    add(o3d.io.read_point_cloud(str(S["top"] / "cloud_top.ply")), S["T_top"], len(S["frames"]))
    return dict(P=np.vstack(P), C=np.vstack(C), shot=np.concatenate(shot),
                cos=np.concatenate(cos) if need_normals else None, n_side=len(S["frames"]))


# ---------------------------------------------------------------------------
# significance
# ---------------------------------------------------------------------------
def view_support(P, shot, n_shots, cell):
    """Per point: the number of distinct shots with a point in its cell or the 26 around it."""
    k = np.floor(P / cell).astype(np.int64)
    k -= k.min(0) - 1                                    # one empty cell of margin: offsets never wrap
    n = k.max(0) + 2
    key = (k[:, 0] * n[1] + k[:, 1]) * n[2] + k[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    pair = np.unique(inv.astype(np.int64) * n_shots + shot)          # (cell, shot) occupied
    cell_i, shot_i = pair // n_shots, (pair % n_shots).astype(np.uint64)
    words = (n_shots + 63) // 64
    bits = np.zeros((len(uk), words), np.uint64)
    np.bitwise_or.at(bits, (cell_i, (shot_i // 64).astype(np.int64)), np.left_shift(np.uint64(1), shot_i % 64))
    acc = bits.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nk = uk + (dx * n[1] + dy) * n[2] + dz
                pos = np.minimum(np.searchsorted(uk, nk), len(uk) - 1)
                hit = uk[pos] == nk
                acc[hit] |= bits[pos[hit]]
    return POPCOUNT[acc.view(np.uint8)].reshape(len(uk), -1).sum(1)[inv]


def fuse(P, cols, voxel, w):
    """Weighted voxel grid: one point per occupied voxel, weighted mean of P and every column
    array in `cols` (each (N, d))."""
    k = np.floor(P / voxel).astype(np.int64)
    k -= k.min(0)
    n = k.max(0) + 1
    _, inv = np.unique((k[:, 0] * n[1] + k[:, 1]) * n[2] + k[:, 2], return_inverse=True)
    inv = inv.ravel()
    ws = np.bincount(inv, weights=w)

    def mean(X):
        return np.column_stack([np.bincount(inv, weights=w * X[:, j]) for j in range(X.shape[1])]) / ws[:, None]
    return mean(P), [mean(X) for X in cols]


def neighbours(P, radius, max_nn):
    """Indices (N, max_nn) of the points within radius (-1 padded, self first) and counts."""
    t = o3d.core.Tensor(P.astype(np.float32))
    nns = o3d.core.nns.NearestNeighborSearch(t)
    nns.hybrid_index(radius)
    idx, _, cnt = nns.hybrid_search(t, radius, max_nn)
    return idx.numpy(), cnt.numpy().astype(np.int64)


def plane_project(P, radius, max_variation, min_nn=6, max_nn=60, chunk=50000):
    """Project every point onto the PCA plane of its neighbourhood where that is flat.
    Returns (new P, moved mask)."""
    idx, cnt = neighbours(P, radius, max_nn)
    out, moved = P.copy(), np.zeros(len(P), bool)
    for s in range(0, len(P), chunk):
        e = min(s + chunk, len(P))
        I, c = idx[s:e], cnt[s:e]
        ok = I >= 0
        Q = P[np.where(ok, I, np.arange(s, e)[:, None])]
        m = (Q * ok[..., None]).sum(1) / np.maximum(c, 1)[:, None]
        D = (Q - m[:, None]) * ok[..., None]
        ev, evec = np.linalg.eigh(np.einsum("bki,bkj->bij", D, D) / np.maximum(c, 1)[:, None, None])
        var = ev[:, 0] / np.maximum(ev.sum(1), 1e-18)                  # surface variation, 0 = flat
        nrm = evec[:, :, 0]
        go = (c >= min_nn) & (var < max_variation)
        off = ((P[s:e] - m) * nrm).sum(1)
        out[s:e][go] = P[s:e][go] - off[go, None] * nrm[go]
        moved[s:e] = go
    return out, moved


def components(P, args, log):
    """DBSCAN at --cluster-mm; keep the largest component and every other one with at least
    --min-cluster points that comes within --max-gap-mm of it (a leaf only one view saw is
    kept; a patch of background far off the plant is not). Returns the keep mask."""
    labels = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).cluster_dbscan(
        args.cluster_mm / 1000.0, 3))
    if not (labels >= 0).any():
        return np.ones(len(P), bool)
    sizes = np.bincount(labels[labels >= 0])
    main = int(np.argmax(sizes))
    main_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[labels == main]))
    good, n_small, n_far = [main], 0, 0
    for k in np.argsort(-sizes)[1:]:
        if sizes[k] < args.min_cluster:
            n_small += 1
            continue
        gap = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[labels == k]))
                         .compute_point_cloud_distance(main_pcd)).min() * 1000
        if args.max_gap_mm > 0 and gap > args.max_gap_mm:
            n_far += 1
        else:
            good.append(int(k))
    keep = np.isin(labels, good)
    log(f"[SIG ] 5 components at {args.cluster_mm:g} mm: {len(sizes)} clusters -> kept {len(good)} "
        f"({n_small} under {args.min_cluster} points, {n_far} more than {args.max_gap_mm:g} mm off the plant), "
        f"{int((~keep).sum()):,} points dropped")
    return keep


def significant(D, args, log):
    """The significance pipeline -> (side P, C, support), (top P, C, support), stage counts."""
    P, C, shot, n_side = D["P"], D["C"], D["shot"], D["n_side"]
    is_top = shot == n_side
    counts = dict(side_in=int((~is_top).sum()), top_in=int(is_top.sum()))

    # 1. incidence
    keep = D["cos"] >= np.cos(np.radians(args.max_incidence_deg))
    counts.update(side_incidence=int((keep & ~is_top).sum()), top_incidence=int((keep & is_top).sum()))
    # 2. consensus (support counted over the incidence-filtered points)
    sup = np.zeros(len(P), np.int64)
    sup[keep] = view_support(P[keep], shot[keep], n_side + 1, args.support_mm / 1000.0)
    keep &= is_top | (sup >= args.min_views)
    counts["side_consensus"] = int((keep & ~is_top).sum())
    log(f"[SIG ] 1 incidence <= {args.max_incidence_deg:g} deg: side {counts['side_in']:,} -> "
        f"{counts['side_incidence']:,}, top {counts['top_in']:,} -> {counts['top_incidence']:,}")
    log(f"[SIG ] 2 consensus >= {args.min_views} shots within {args.support_mm:g} mm cells: side -> "
        f"{counts['side_consensus']:,}  (support median {np.median(sup[keep & ~is_top]):.0f} shots)")

    # 3. weighted voxel fusion, side and top apart
    w = D["cos"]
    parts = []
    for name, sel in (("side", keep & ~is_top), ("top", keep & is_top)):
        Pf, (Cf, Sf) = fuse(P[sel], [C[sel], sup[sel, None].astype(float)], args.voxel_mm / 1000.0, w[sel])
        parts.append([Pf, Cf, Sf[:, 0]])
        counts[f"{name}_voxel"] = len(Pf)
    log(f"[SIG ] 3 {args.voxel_mm:g} mm voxels, incidence-weighted: side -> {counts['side_voxel']:,}, "
        f"top -> {counts['top_voxel']:,}")

    # 4. + 5. on side and top together (they share surfaces)
    Pa = np.vstack([parts[0][0], parts[1][0]])
    src = np.r_[np.zeros(len(parts[0][0]), int), np.ones(len(parts[1][0]), int)]
    if args.smooth_mm > 0:
        Pn, moved = plane_project(Pa, args.smooth_mm / 1000.0, args.max_variation)
        shift = np.linalg.norm(Pn - Pa, axis=1)[moved] * 1000
        Pa = Pn
        log(f"[SIG ] 4 plane fit within {args.smooth_mm:g} mm: {moved.mean():.0%} of the points flat enough, "
            f"moved median {np.median(shift):.2f} / p90 {np.percentile(shift, 90):.2f} mm")
    keep2 = np.ones(len(Pa), bool)
    if args.min_cluster > 0:
        keep2 = components(Pa, args, log)
    out = []
    for i, (Pf, Cf, Sf) in enumerate(parts):
        sel = keep2[src == i]
        out.append((Pa[src == i][sel], Cf[sel], Sf[sel]))
    counts.update(side_out=len(out[0][0]), top_out=len(out[1][0]))
    return out[0], out[1], counts


# ---------------------------------------------------------------------------
# colours
# ---------------------------------------------------------------------------
def heat(v, vmax):
    """Values -> turbo colours (RGB 0-1), 1 = blue ... vmax = red (turbo's dark end skipped)."""
    u = 0.1 + 0.9 * np.clip((np.asarray(v, float) - 1) / max(vmax - 1, 1), 0, 1)
    bgr = cv2.applyColorMap((u * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    return bgr[:, ::-1] / 255.0


def shot_colours(shot, n_side):
    hsv = np.stack([(shot * 180 // max(n_side, 1)) % 180, np.full_like(shot, 220), np.full_like(shot, 255)], 1)
    rgb = cv2.cvtColor(hsv.astype(np.uint8)[:, None], cv2.COLOR_HSV2RGB)[:, 0] / 255.0
    rgb[shot == n_side] = TOP_WHITE
    return rgb


# ---------------------------------------------------------------------------
def add_significance_args(ap):
    """The significance-pipeline settings (shared with overlap_view.py)."""
    ap.add_argument("--voxel-mm", type=float, default=2.0, help="output voxel size")
    ap.add_argument("--normal-mm", type=float, default=5.0, help="per-shot normal estimation radius")
    ap.add_argument("--max-incidence-deg", type=float, default=75.0,
                    help="drop points seen more obliquely than this in their own shot")
    ap.add_argument("--support-mm", type=float, default=2.0, help="consensus cell size (+ its 26 neighbours)")
    ap.add_argument("--min-views", type=int, default=4, help="side points need this many shots in support")
    ap.add_argument("--smooth-mm", type=float, default=6.0, help="plane-fit radius (0 = off)")
    ap.add_argument("--max-variation", type=float, default=0.25,
                    help="plane fit only where the neighbourhood's surface variation is below this")
    ap.add_argument("--cluster-mm", type=float, default=5.0, help="DBSCAN radius for the component filter")
    ap.add_argument("--min-cluster", type=int, default=100, help="drop components with fewer points (0 = off)")
    ap.add_argument("--max-gap-mm", type=float, default=30.0,
                    help="drop components farther than this from the main plant component (0 = off)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Raw overlap of all shots, or significance-downsampled, in 3D",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sv.add_scene_args(ap)
    ap.add_argument("--downsample", action="store_true", help="downsample (else: raw overlap of every shot)")
    ap.add_argument("--method", choices=["significant", "voxel"], default="significant",
                    help="downsampling: significance pipeline, or a plain voxel-grid mean")
    ap.add_argument("--color", choices=["rgb", "shot", "views"], default="rgb",
                    help="camera colour, one colour per shot, or multi-view support heat map")
    add_significance_args(ap)
    ap.add_argument("--snapshots", action="store_true", help="write downsample/snap_<mode>_<color>.png")
    ap.add_argument("--no-view", action="store_true", help="no viewer")
    args = ap.parse_args()

    S = sv.load_scene(args)
    D = load_shots(S, args)
    n_side, A = D["n_side"], S["A"]
    is_top = D["shot"] == n_side
    mode = args.method if args.downsample else "raw"
    print(f"\n[LOAD] {n_side} side shots {int((~is_top).sum()):,} points + top shot {int(is_top.sum()):,} points")
    out = S["work"] / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    if mode == "raw":
        sup = view_support(D["P"], D["shot"], n_side + 1, args.support_mm / 1000.0) if args.color == "views" else None
        side = (D["P"][~is_top], D["C"][~is_top], None if sup is None else sup[~is_top])
        top = (D["P"][is_top], D["C"][is_top], None if sup is None else sup[is_top])
        counts = dict(side_out=len(side[0]), top_out=len(top[0]))
        print("[RAW ] every cleaned point of every shot, nothing merged or dropped")
        if sup is not None:
            for lo, hi in ((1, 1), (2, 3), (4, 9), (10, 10 ** 6)):
                f = ((sup >= lo) & (sup <= hi) & ~is_top).mean() / max((~is_top).mean(), 1e-9)
                print(f"[RAW ] side points seen by {lo}{'' if lo == hi else '+' if hi > 999 else f'-{hi}'} "
                      f"shot(s): {f:.1%}")
    elif mode == "voxel":
        voxel = args.voxel_mm / 1000.0
        parts = []
        for sel in (~is_top, is_top):
            Pf, Cf = geo.voxel_fuse(D["P"][sel], D["C"][sel], voxel)
            parts.append((Pf, Cf, None))
        side, top = parts
        counts = dict(side_out=len(side[0]), top_out=len(top[0]))
        print(f"[VOX ] plain {args.voxel_mm:g} mm voxel mean: side {int((~is_top).sum()):,} -> {counts['side_out']:,}, "
              f"top {int(is_top.sum()):,} -> {counts['top_out']:,}")
    else:
        side, top, counts = significant(D, args, print)

    # display colours
    if args.color == "rgb" or (args.color == "views" and side[2] is None):
        if args.color == "views":
            print("[WARN] --color views needs the support of --method significant (or raw mode); showing rgb")
        Cs, Ct = side[1], top[1]
    elif args.color == "shot":
        if mode != "raw":
            print("[WARN] downsampled points merge shots; --color shot is only meaningful without --downsample")
        Cs = shot_colours(D["shot"][~is_top], n_side) if mode == "raw" else side[1]
        Ct = np.tile(TOP_WHITE, (len(top[0]), 1))
    else:
        vmax = np.percentile(side[2], 95)
        Cs, Ct = heat(side[2], vmax), heat(top[2], vmax)
        print(f"[COL ] support heat map: blue = 1 shot ... red >= {vmax:.0f} shots")
    Cs, Ct = np.clip(Cs, 0, 1), np.clip(Ct, 0, 1)

    Ps, Pt = side[0], top[0]
    F = reg.plant_frame(A, Ps, args.base_percentile)
    print(f"[OUT ] {mode}: side {len(Ps):,} + top {len(Pt):,} = {len(Ps) + len(Pt):,} points "
          f"(raw total {len(D['P']):,}, {(len(Ps) + len(Pt)) / len(D['P']):.1%})")
    if mode != "raw":
        side_pcd, top_pcd = sv.to_pcd(Ps, side[1]), sv.to_pcd(Pt, top[1])
        o3d.io.write_point_cloud(str(out / f"{mode}_side.ply"), side_pcd)
        o3d.io.write_point_cloud(str(out / f"{mode}_top.ply"), top_pcd)
        o3d.io.write_point_cloud(str(out / f"{mode}_combined.ply"),
                                 sv.to_pcd(np.vstack([Ps, Pt]), np.clip(np.vstack([side[1], top[1]]), 0, 1)))
        keys = ("voxel_mm", "normal_mm", "max_incidence_deg", "support_mm", "min_views", "smooth_mm",
                "max_variation", "cluster_mm", "min_cluster", "max_gap_mm") if mode == "significant" else ("voxel_mm",)
        (out / f"{mode}.json").write_text(json.dumps(dict(
            finished=datetime.now().isoformat(timespec="seconds"), method=mode,
            settings={k: getattr(args, k) for k in keys}, counts=counts,
            frame="reference = first side shot's colour camera (C1), metres"), indent=2))
        print(f"[OUT ] {out}  ({mode}_side.ply, {mode}_top.ply, {mode}_combined.ply, {mode}.json)")
    if args.snapshots:
        snap = out / f"snap_{mode}_{args.color}.png"
        sv.snapshots(snap, Ps, Cs, Pt, Ct, F, args.snap_mm_per_px)
        print(f"[OUT ] {snap}")

    if not args.no_view:
        args.info = (f"{mode} | colour {args.color}\nside {len(Ps):,} pts\ntop {len(Pt):,} pts\n"
                     f"({(len(Ps) + len(Pt)) / len(D['P']):.1%} of {len(D['P']):,} raw)")
        print("\n[VIEW ] close the window when done")
        sv.Viewer(sv.to_pcd(Ps, Cs), sv.to_pcd(Pt, Ct), F, S["T_top"], S["cam"], args,
                  title=f"Flora {mode} view ({args.color})").run()


if __name__ == "__main__":
    main()
