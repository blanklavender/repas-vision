#!/usr/bin/env python3
"""
leaf_track.py  --  leaf instance IDs on the 2DGS TSDF mesh from per-shot SAM3 leaf masks.

Each shot's SAM3 masks number the leaves 0..N-1 indepe This script linkndently.s them across
shots through the mesh they all see, and gives every mesh vertex one leaf ID.

    python leaf_track.py \
        --run   ../../captures/femto_bolt/plant_10-16_clean55/results/2dgs_7k \
        --masks ../../captures/femto_bolt/plant_10-16_deg_5_speed_5/leaf_masks

Algorithm
---------
1. Visibility. Every mesh vertex is projected into every shot (cameras.json, K + c2w in metres,
   undistorted frame). It counts as seen if it lands in the image, the 2DGS alpha there is
   > --min-alpha, and its depth is within --depth-tol-mm of the rendered median depth (the
   z-buffer). The vertex then takes the SAM3 instance under it (labels map), or none.
2. Pairwise overlap. A node is one SAM3 instance in one shot. For nodes i (shot a) and
   j (shot b), IoU is measured only on vertices both shots see:
       inter = |vertices labelled i in a AND labelled j in b|
       union = |labelled i in a, seen in b| + |labelled j in b, seen in a| - inter
   Pairs with union < --min-union are "not comparable" (the shots don't share that area).
3. Clustering. Average-linkage agglomeration: repeatedly merge the two clusters with the
   highest mean IoU over their comparable node pairs, while it is >= --min-iou and the two
   clusters have no shot in common (one shot can't show the same leaf as two instances).
   Clusters with fewer than --min-views shots are discarded.
4. Vertex vote. A vertex gets the leaf most of its labelled observations agree on, if that
   leaf got >= --min-votes votes and >= --min-frac of the shots that saw the vertex.
   Otherwise it is 0 (not a leaf: pot, stem, background, or ambiguous). Leaves left with
   fewer than --min-vertices vertices are dropped as noise.

I/O contract
------------
in   <run>/mesh/tsdf_mesh_metres.ply   world = shot-0 colour camera, metres
     <run>/maps/cameras.json           K, wh, views [{stem, c2w_metres}]
     <run>/maps/<stem>.npz             median_depth_mm, alpha  (2DGS renders)
     <run>/maps/<stem>_rgb.png         used only for the check images
     <masks>/<stem>.npz                labels (H,W) uint16, scores (sam3_leaf_masks.py)
out  <out>/vertex_leaf.npy             (V,) int32, 0 = not a leaf, k = leaf k (1 = most vertices)
     <out>/vertex_leaf_frac.npy        (V,) float32, winning votes / shots that saw the vertex
     <out>/leaves_mesh.ply             the mesh in its own RGB, leaf vertices tinted with their
                                       leaf colour (--leaf-alpha; 1 = flat leaf colours)
     <out>/leaves/leaf_<kk>.ply        one surface per leaf (triangles whose 3 vertices are leaf k),
                                       same colours as leaves_mesh.ply
     <out>/tracks.json                 settings, per leaf: shots + instances, per shot:
                                       instance -> leaf map (-1 = discarded)
     <out>/check/<stem>.jpg            each shot's SAM3 masks coloured by GLOBAL leaf ID --
                                       a leaf should keep its colour from shot to shot
     <out>/check/montage.jpg           every --montage-step-th shot in one image
Default <out> is <run>/leaf_track.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import scipy.sparse as sp

PALETTE = np.array([(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
                    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
                    (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
                    (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)],
                   np.uint8)                            # RGB, leaf k -> PALETTE[(k-1) % 20]
GREY = np.array([150, 150, 150], np.uint8)


def leaf_rgb(k):
    return GREY if k <= 0 else PALETTE[(k - 1) % len(PALETTE)]


# ---------------------------------------------------------------------------
# 1. visibility + per-shot labels
# ---------------------------------------------------------------------------
def observe(verts, K, wh, c2w, depth_mm, alpha, labels, args):
    """-> (indices of vertices seen in this shot, their SAM3 label, 0 = none)."""
    T = np.linalg.inv(c2w)
    X = verts @ T[:3, :3].T + T[:3, 3]
    z = X[:, 2]
    w, h = wh
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(K[0, 0] * X[:, 0] / z + K[0, 2])
        v = np.round(K[1, 1] * X[:, 1] / z + K[1, 2])
    idx = np.nonzero((z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h))[0]
    u, v = u[idx].astype(int), v[idx].astype(int)
    d = depth_mm[v, u]
    seen = (alpha[v, u] > args.min_alpha) & (d > 0) & (np.abs(z[idx] * 1000.0 - d) < args.depth_tol_mm)
    return idx[seen], labels[v[seen], u[seen]].astype(np.int64)


# ---------------------------------------------------------------------------
# 3. average-linkage clustering with a same-shot cannot-link
# ---------------------------------------------------------------------------
def cluster(node_view, iou, comparable, n_views, min_iou):
    """node_view (N,), iou / comparable dense (N,N). Returns cluster index per node."""
    n = len(node_view)
    S = np.where(comparable, iou, 0.0)                  # cluster-level sum of IoU
    C = comparable.astype(np.float64)                   # cluster-level number of comparable pairs
    V = np.zeros((n, n_views), bool)                    # shots in each cluster
    V[np.arange(n), node_view] = True
    alive = np.ones(n, bool)
    members = [[i] for i in range(n)]
    while True:
        with np.errstate(divide="ignore", invalid="ignore"):
            link = np.where(C > 0, S / C, -1.0)
        conflict = (V.astype(np.int32) @ V.T.astype(np.int32)) > 0
        link[conflict | ~alive[:, None] | ~alive[None, :]] = -1.0
        np.fill_diagonal(link, -1.0)
        a, b = np.unravel_index(np.argmax(link), link.shape)
        if link[a, b] < min_iou:
            break
        S[a] += S[b]; S[:, a] += S[:, b]                # merge b into a
        C[a] += C[b]; C[:, a] += C[:, b]
        S[b] = S[:, b] = 0; C[b] = C[:, b] = 0
        V[a] |= V[b]; V[b] = False
        alive[b] = False
        members[a] += members[b]; members[b] = []
    out = np.full(n, -1)
    for c, m in enumerate(members):
        out[m] = c
    return out


# ---------------------------------------------------------------------------
# check images
# ---------------------------------------------------------------------------
def check_image(rgb_path, labels, inst_leaf, wh):
    img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR) if rgb_path.is_file() else None
    if img is None:
        img = np.zeros((wh[1], wh[0], 3), np.uint8)
    out = img.copy()
    for i, k in enumerate(inst_leaf):
        m = labels == i + 1
        if not m.any():
            continue
        c = leaf_rgb(k)[::-1].astype(np.float64)        # BGR
        out[m] = (out[m] * 0.45 + c * 0.55).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(x) for x in c), 2)
        ys, xs = np.nonzero(m)
        txt = f"L{k}" if k > 0 else "x"
        p = (int(np.median(xs)) - 10, int(np.median(ys)) + 5)
        cv2.putText(out, txt, p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, txt, p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def crop_box(images):
    """Common crop around the coloured masks of all check images (plus margin)."""
    ys, xs = [], []
    for img, lab, _ in images:
        yy, xx = np.nonzero(lab)
        if len(xx):
            ys += [yy.min(), yy.max()]; xs += [xx.min(), xx.max()]
    h, w = images[0][1].shape
    if not xs:
        return 0, h, 0, w
    m = 30
    return max(min(ys) - m, 0), min(max(ys) + m, h), max(min(xs) - m, 0), min(max(xs) + m, w)


def montage(tiles, cols=6):
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    return sheet


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="leaf instance IDs on the 2DGS TSDF mesh",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--run", required=True, help="2DGS result folder (mesh/, maps/)")
    ap.add_argument("--masks", required=True, help="sam3_leaf_masks.py output folder")
    ap.add_argument("--mesh", default="mesh/tsdf_mesh_metres.ply", help="mesh, relative to --run")
    ap.add_argument("--out", default=None, help="output folder (default <run>/leaf_track)")
    ap.add_argument("--depth-tol-mm", type=float, default=4.0, help="visibility depth tolerance")
    ap.add_argument("--min-alpha", type=float, default=0.5, help="2DGS opacity needed to trust depth")
    ap.add_argument("--min-union", type=int, default=50,
                    help="vertices two instances must jointly cover to be compared")
    ap.add_argument("--min-iou", type=float, default=0.3, help="stop merging below this mean IoU")
    ap.add_argument("--min-views", type=int, default=3, help="drop leaves seen in fewer shots")
    ap.add_argument("--min-votes", type=int, default=2, help="votes a vertex needs for a leaf ID")
    ap.add_argument("--min-frac", type=float, default=0.4,
                    help="winning votes / shots that saw the vertex, needed for a leaf ID")
    ap.add_argument("--min-vertices", type=int, default=200,
                    help="drop leaves that end up with fewer labelled vertices (noise clusters)")
    ap.add_argument("--leaf-alpha", type=float, default=0.55,
                    help="leaf colour blended over the mesh RGB in the output PLYs (0 = RGB only, 1 = flat)")
    ap.add_argument("--montage-step", type=int, default=3, help="every n-th shot in montage.jpg")
    args = ap.parse_args()

    t0 = time.perf_counter()
    run, mdir = Path(args.run), Path(args.masks)
    out = Path(args.out) if args.out else run / "leaf_track"
    for p in (run / args.mesh, run / "maps" / "cameras.json", mdir):
        if not p.exists():
            sys.exit(f"missing {p}")
    (out / "check").mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(run / args.mesh))
    verts = np.asarray(mesh.vertices)
    cams = json.loads((run / "maps" / "cameras.json").read_text())
    K, wh = np.array(cams["K"], float), tuple(cams["wh"])
    mask_info = json.loads((mdir / "leaf_masks.json").read_text()) if (mdir / "leaf_masks.json").exists() else {}
    if mask_info and (not mask_info.get("undistorted", True) or not np.allclose(mask_info["K"], K, atol=1e-3)):
        sys.exit(f"{mdir}: masks are not in the cameras.json frame (distorted, or a different K)")

    views = []
    for vw in cams["views"]:
        if not (mdir / f"{vw['stem']}.npz").is_file():
            print(f"[WARN] no leaf masks for {vw['stem']}; shot skipped")
            continue
        views.append(vw)
    print(f"[LOAD] {len(verts)} vertices | {len(views)} shots with masks")

    # 1. observations -----------------------------------------------------------------
    nv = len(verts)
    vis_rows, vis_cols, obs_rows, obs_cols = [], [], [], []
    node_view, node_inst, node_score, node_offset = [], [], [], []
    shot_labels = []
    for s, vw in enumerate(views):
        m = np.load(run / "maps" / f"{vw['stem']}.npz")
        lm = np.load(mdir / f"{vw['stem']}.npz")
        labels, scores = lm["labels"], lm["scores"]
        if labels.shape != (wh[1], wh[0]):
            sys.exit(f"{vw['stem']}: mask size {labels.shape[::-1]} != {wh}")
        seen, lab = observe(verts, K, wh, np.array(vw["c2w_metres"], float),
                            m["median_depth_mm"], m["alpha"], labels, args)
        node_offset.append(len(node_view))
        node_view += [s] * len(scores)
        node_inst += list(range(len(scores)))
        node_score += [float(x) for x in scores]
        vis_rows.append(seen); vis_cols.append(np.full(len(seen), s))
        on = lab > 0
        obs_rows.append(seen[on]); obs_cols.append(node_offset[-1] + lab[on] - 1)
        shot_labels.append(labels)
    node_view = np.array(node_view)
    n_nodes, n_views = len(node_view), len(views)
    Vis = sp.csr_matrix((np.ones(sum(map(len, vis_rows)), np.float32),
                         (np.concatenate(vis_rows), np.concatenate(vis_cols))), shape=(nv, n_views))
    M = sp.csr_matrix((np.ones(sum(map(len, obs_rows)), np.float32),
                       (np.concatenate(obs_rows), np.concatenate(obs_cols))), shape=(nv, n_nodes))
    print(f"[VIS ] {n_nodes} SAM3 instances | {int((Vis.sum(1) > 0).sum())} vertices seen at least once "
          f"| {int((M.sum(1) > 0).sum())} labelled at least once")

    # 2. pairwise IoU on co-visible vertices ------------------------------------------
    inter = (M.T @ M).toarray()                          # (N,N)
    P = (M.T @ Vis).toarray()                            # (N,views): node i's vertices seen in shot b
    Pi_in_j = P[:, node_view]                            # [i,j] = |node i seen in shot of j|
    union = Pi_in_j + Pi_in_j.T - inter
    same = node_view[:, None] == node_view[None, :]
    comparable = (union >= args.min_union) & ~same
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(comparable, inter / np.maximum(union, 1), 0.0)

    # 3. clustering ---------------------------------------------------------------------
    cl = cluster(node_view, iou, comparable, n_views, args.min_iou)
    ids, counts = np.unique(cl, return_counts=True)
    keep = ids[counts >= args.min_views]
    node_cluster = np.where(np.isin(cl, keep), cl, -1)

    # 4. vertex vote --------------------------------------------------------------------
    keep_list = list(keep)
    col = {c: i for i, c in enumerate(keep_list)}
    A = sp.csr_matrix((np.ones(int((node_cluster >= 0).sum())),
                       (np.nonzero(node_cluster >= 0)[0], [col[c] for c in node_cluster if c >= 0])),
                      shape=(n_nodes, len(keep_list)))
    votes = (M @ A).toarray() if len(keep_list) else np.zeros((nv, 0))   # (V, leaves)
    n_seen = np.asarray(Vis.sum(1)).ravel()
    best = votes.argmax(1) if len(keep_list) else np.zeros(nv, int)
    best_votes = votes[np.arange(nv), best] if len(keep_list) else np.zeros(nv)
    frac = np.where(n_seen > 0, best_votes / np.maximum(n_seen, 1), 0.0)
    ok = (best_votes >= args.min_votes) & (frac >= args.min_frac)

    # drop tiny leaves, then number the rest 1..L by labelled vertex count, largest first
    sizes = np.bincount(best[ok], minlength=len(keep_list))
    big = sizes >= args.min_vertices
    ok &= big[best]
    order = [c for c in np.argsort(-sizes) if big[c]]
    rank = np.full(len(keep_list), -1)
    rank[order] = np.arange(1, len(order) + 1)
    vertex_leaf = np.where(ok, rank[best], 0).astype(np.int32)
    node_leaf = np.array([rank[col[c]] if c >= 0 else -1 for c in node_cluster])
    n_leaves = len(order)

    np.save(out / "vertex_leaf.npy", vertex_leaf)
    np.save(out / "vertex_leaf_frac.npy", frac.astype(np.float32))
    # keep the mesh's own RGB; tint leaf vertices with their leaf colour (--leaf-alpha blend)
    if mesh.has_vertex_colors():
        rgb = np.asarray(mesh.vertex_colors, np.float64)
    else:
        print("[WARN] mesh has no vertex colours; non-leaf vertices will be grey")
        rgb = np.tile(GREY / 255.0, (nv, 1))
    tint = np.array([leaf_rgb(k) for k in range(vertex_leaf.max() + 1)], np.float64)[vertex_leaf] / 255.0
    a = np.where(vertex_leaf > 0, args.leaf_alpha, 0.0)[:, None]
    colored = o3d.geometry.TriangleMesh(mesh)
    colored.vertex_colors = o3d.utility.Vector3dVector((1 - a) * rgb + a * tint)
    o3d.io.write_triangle_mesh(str(out / "leaves_mesh.ply"), colored)

    # one surface per leaf: the triangles whose three vertices all carry that leaf ID
    (out / "leaves").mkdir(exist_ok=True)
    for old in (out / "leaves").glob("leaf_*.ply"):
        old.unlink()
    tris = np.asarray(mesh.triangles)
    tri_leaf = np.where((vertex_leaf[tris] == vertex_leaf[tris[:, :1]]).all(1), vertex_leaf[tris[:, 0]], 0)
    for k in range(1, n_leaves + 1):
        sub = o3d.geometry.TriangleMesh(colored)
        sub.remove_triangles_by_mask(tri_leaf != k)
        sub.remove_unreferenced_vertices()
        o3d.io.write_triangle_mesh(str(out / "leaves" / f"leaf_{k:02d}.ply"), sub)

    # tracks.json ---------------------------------------------------------------------------
    leaves = []
    for k in range(1, n_leaves + 1):
        nodes = np.nonzero(node_leaf == k)[0]
        vk = verts[vertex_leaf == k]
        leaves.append(dict(
            leaf=k, n_vertices=int(len(vk)), n_shots=int(len(nodes)),
            centroid_m=vk.mean(0).round(4).tolist() if len(vk) else None,
            mean_score=round(float(np.mean([node_score[i] for i in nodes])), 3),
            instances=[dict(stem=views[node_view[i]]["stem"], instance=int(node_inst[i]),
                            score=round(node_score[i], 3)) for i in nodes]))
    shots = []
    for s, vw in enumerate(views):
        n0 = node_offset[s]
        n1 = node_offset[s + 1] if s + 1 < n_views else n_nodes
        shots.append(dict(stem=vw["stem"], instance_leaf=node_leaf[n0:n1].tolist()))
    dropped = int((node_leaf < 0).sum())
    (out / "tracks.json").write_text(json.dumps(dict(
        run=str(run), masks=str(mdir), mesh=args.mesh,
        params={k: v for k, v in vars(args).items() if k not in ("run", "masks", "out", "mesh")},
        n_vertices=nv, n_leaf_vertices=int((vertex_leaf > 0).sum()), n_instances=n_nodes,
        n_instances_dropped=dropped, leaves=leaves, shots=shots), indent=2))

    # check images ----------------------------------------------------------------------------
    imgs = []
    for s, vw in enumerate(views):
        n0 = node_offset[s]
        n1 = node_offset[s + 1] if s + 1 < n_views else n_nodes
        img = check_image(run / "maps" / f"{vw['stem']}_rgb.png", shot_labels[s], node_leaf[n0:n1], wh)
        cv2.putText(img, vw["stem"], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out / "check" / f"{vw['stem']}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if s % args.montage_step == 0:
            imgs.append((img, shot_labels[s], vw["stem"]))
    if imgs:
        y0, y1, x0, x1 = crop_box(imgs)
        tiles = []
        for img, _, stem in imgs:
            t = img[y0:y1, x0:x1].copy()
            cv2.putText(t, stem, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(t)
        cv2.imwrite(str(out / "check" / "montage.jpg"), montage(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])

    # summary ---------------------------------------------------------------------------------
    print(f"[TRK ] {len(leaves)} leaves (>= {args.min_views} shots, >= {args.min_vertices} vertices) | "
          f"{dropped}/{n_nodes} instances not assigned to a leaf")
    for L in leaves:
        print(f"       L{L['leaf']:<3} {L['n_shots']:3d} shots  {L['n_vertices']:6d} vertices  "
              f"mean SAM3 score {L['mean_score']:.2f}")
    print(f"[DONE] {int((vertex_leaf > 0).sum())}/{nv} vertices labelled -> {out}  "
          f"({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
