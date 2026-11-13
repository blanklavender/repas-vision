"""
tof_clean  --  the one ToF cleaning used for seed fusion, the depth diagnostic and future captures.

Per view, clean_view() works on the raw (distorted) depth grid, where the gradient and jump tests are
meaningful. Stages run in this order, and each pixel is charged to the first stage that removes it:
    no_depth, depth_range, gradient, occlusion, edge, obliquity

Across views, a capture run then applies:
    view drop      a view whose pose residual exceeds the limit contributes nothing, not even as a witness
    multi-view     a point stays only with enough other views confirming it
    SOR            statistical outlier removal on all remaining points together
    voxel fusion   merges points, removes none: mean point and colour per voxel, as stitch_view.py

Multi-view rule. View j "should see" a point when the point is in front of j, lands inside j's image and
mask, and j's raw depth at that pixel is not more than the tolerance in front of it (not occluded). j is a
witness when it should see the point and j's own cleaned depth there is within the tolerance. Only cleaned
depth can testify. The test is nearest-pixel, on j's raw grid.

Run a capture (writes only to --out; raw data is only read):
    cd ~/Projects/repas-vision/flora_pipeline
    ~/miniconda3/envs/gsplat/bin/python -B -m flora_lib.tof_clean --raw_dir <capture> --out <new folder> \
        --labels_cache <diag>/tof_cache
Change a setting for one run with --set, e.g. --set obliquity_max_deg=55. The run records it as an override.

Depends only on numpy, cv2 and scipy, so it imports from the gsplat env, which has no Open3D.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from . import geometry as geo

CONFIG = dict(
    depth_range_mm=(200.0, 1200.0),     # depth == 0 is dropped as well
    grad_max_mm_per_px=20.0,            # Sobel / 8 on the raw grid, as tv_common.gradient_mask
    grad_hole_guard_px=1,
    occlusion_tol_mm=40.0,              # as clean_side.py + tv_common.near_filter
    occlusion_window_px=24,
    occlusion_mask_erode_px=2,          # clean_side.py erodes the mask by 2 px to form the occluder set
    edge_px=3,                          # distance from the mask boundary or a depth jump
    edge_jump_mm=10.0,                  # between 4-neighbours that both have depth; a hole is not a jump
    obliquity_max_deg=45.0,             # angle between the local plane's normal and the camera ray
    normal_window_px=5,
    normal_min_points=6,                # less support than this: no normal, so dropped at this stage
    view_residual_max_deg=2.0,
    view_residual_window=8,             # neighbours on each side
    multiview_min_witnesses=2,
    multiview_tol_mm=5.0,
    sor_neighbours=20,
    sor_std=2.0,
    voxel_mm=1.0,                       # the existing side_fused.ply used 1 mm
)

VIEW_STAGES = ("no_depth", "depth_range", "gradient", "occlusion", "edge", "obliquity")
FUSION_STAGES = ("view_drop", "multiview", "sor")
STAGES = VIEW_STAGES + FUSION_STAGES
NOT_IN_MASK, KEPT = 0, 255
STAGE_CODE = {name: i + 1 for i, name in enumerate(STAGES)}

# Region labels of diag_depth.py, used only for reporting.
LABELS = {1: "plant", 2: "pot", 3: "tags", 4: "unassigned", 0: "unlabelled"}


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def _ellipse(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _erode(mask, px):
    return mask if px <= 0 else cv2.erode(mask.astype(np.uint8), _ellipse(px)).astype(bool)


def _gradient_ok(depth_mm, grad_max, hole_guard):
    """Same result as tv_common.gradient_mask."""
    if grad_max <= 0:
        return np.ones(depth_mm.shape, bool)
    d = depth_mm.astype(np.float32)
    if hole_guard > 0:
        d = np.where(d > 0, d, cv2.dilate(d, _ellipse(hole_guard)))
    g = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    return g < grad_max


def _occlusion_ok(depth_mm, occluder, tol_mm, window_px):
    """Same result as tv_common.near_filter: at most tol_mm behind the nearest occluder within the window."""
    if tol_mm <= 0:
        return np.ones(depth_mm.shape, bool)
    z = depth_mm.astype(np.float32) / 1000.0
    zk = np.where(occluder & (z > 0), z, np.inf).astype(np.float32)
    k = 2 * int(round(window_px)) + 1
    return z <= cv2.erode(zk, np.ones((k, k), np.uint8)) + tol_mm / 1000.0


def _edge_ok(depth_mm, mask, px, jump_mm):
    """False within px of the mask boundary or of a depth jump."""
    if px <= 0:
        return np.ones(depth_mm.shape, bool)
    d = depth_mm.astype(np.float32)
    jump = np.zeros(d.shape, bool)
    across = (np.abs(d[:, 1:] - d[:, :-1]) > jump_mm) & (d[:, 1:] > 0) & (d[:, :-1] > 0)
    jump[:, 1:] |= across
    jump[:, :-1] |= across
    down = (np.abs(d[1:, :] - d[:-1, :]) > jump_mm) & (d[1:, :] > 0) & (d[:-1, :] > 0)
    jump[1:, :] |= down
    jump[:-1, :] |= down
    near_jump = cv2.dilate(jump.astype(np.uint8), _ellipse(px)) > 0
    return _erode(mask, px) & ~near_jump


def _obliquity_ok(xyz, alive, rays, max_deg, window, min_points):
    """Plane fit over the alive pixels of each window. A pixel without enough support is not ok."""
    if max_deg <= 0:
        return np.ones(alive.shape, bool)
    a = alive.astype(np.float64)
    centre = xyz[alive].mean(0) if alive.any() else np.zeros(3)  # keeps the running sums well conditioned
    X = (xyz - centre) * a[..., None]

    def box(im):
        return cv2.boxFilter(im, cv2.CV_64F, (window, window), normalize=False, borderType=cv2.BORDER_CONSTANT)

    n = box(a)
    sums = [box(X[..., i]) for i in range(3)]
    ok = np.zeros(alive.shape, bool)
    pick = alive & (n >= min_points)
    if not pick.any():
        return ok
    cov = np.zeros((int(pick.sum()), 3, 3))
    for i in range(3):
        for j in range(i, 3):
            c = (box(X[..., i] * X[..., j]) - sums[i] * sums[j] / np.maximum(n, 1))[pick]
            cov[:, i, j] = c
            cov[:, j, i] = c
    normals = np.linalg.eigh(cov)[1][:, :, 0]  # eigenvalues ascend: the smallest is the plane's normal
    r = rays[pick] / np.linalg.norm(rays[pick], axis=1, keepdims=True)
    ok[pick] = np.abs((normals * r).sum(1)) >= np.cos(np.radians(max_deg))
    return ok


def clean_view(depth_mm, K, dist, mask, config=None, labels=None, rays=None):
    """Clean one view.

    depth_mm  raw grid, mm, 0 = no depth
    K, dist   the colour camera the depth is aligned to
    mask      raw grid, bool: the object
    config    overrides for CONFIG (a filter value of 0 turns that filter off)
    labels    optional raw-grid region labels, to split the counts in stats
    rays      optional precomputed geo.ray_grid(K, dist, w, h)

    Returns:
        points      (N, 3) metres, this camera's frame, raster order of valid_mask
        valid_mask  raw grid, bool
        stats       n_mask, removed {stage: n}, kept, stage_map (uint8: NOT_IN_MASK, STAGE_CODE or KEPT),
                    and removed_by_label / kept_by_label when labels are given
    """
    cfg = {**CONFIG, **(config or {})}
    h, w = depth_mm.shape
    if mask.shape != (h, w):
        raise ValueError(f"mask {mask.shape} != depth {(h, w)}")
    rays = geo.ray_grid(K, dist, w, h) if rays is None else rays
    d = depth_mm.astype(np.float32)
    mask = mask.astype(bool)
    alive = mask.copy()
    stage_map = np.where(mask, KEPT, NOT_IN_MASK).astype(np.uint8)
    removed = {}

    def apply(stage, ok):
        nonlocal alive
        out = alive & ~ok
        stage_map[out] = STAGE_CODE[stage]
        removed[stage] = int(out.sum())
        alive = alive & ok

    lo, hi = cfg["depth_range_mm"]
    apply("no_depth", d > 0)
    apply("depth_range", (d >= lo) & (d <= hi))
    grad = _gradient_ok(d, cfg["grad_max_mm_per_px"], cfg["grad_hole_guard_px"])
    apply("gradient", grad)
    occluder = _erode(mask, cfg["occlusion_mask_erode_px"]) & (d >= lo) & (d <= hi) & grad
    apply("occlusion", _occlusion_ok(d, occluder, cfg["occlusion_tol_mm"], cfg["occlusion_window_px"]))
    apply("edge", _edge_ok(d, mask, cfg["edge_px"], cfg["edge_jump_mm"]))
    xyz = rays * (d[..., None].astype(np.float64) / 1000.0)
    apply("obliquity", _obliquity_ok(xyz, alive, rays, cfg["obliquity_max_deg"], cfg["normal_window_px"],
                                     cfg["normal_min_points"]))

    stats = dict(n_mask=int(mask.sum()), removed=removed, kept=int(alive.sum()), stage_map=stage_map)
    if labels is not None:
        stats.update(count_by_label(stage_map, labels))
    return xyz[alive], alive, stats


def count_by_label(stage_map, labels):
    removed = {LABELS[k]: {s: int(((labels == k) & (stage_map == STAGE_CODE[s])).sum()) for s in STAGES} for k in LABELS}
    kept = {LABELS[k]: int(((labels == k) & (stage_map == KEPT)).sum()) for k in LABELS}
    return dict(removed_by_label=removed, kept_by_label=kept)


def labels_to_raw_grid(region, rays, K):
    """Undistorted-grid labels -> raw grid. Each raw pixel takes the label at its undistorted position."""
    h, w = region.shape
    u = np.rint(K[0, 0] * rays[..., 0] + K[0, 2]).astype(int)
    v = np.rint(K[1, 1] * rays[..., 1] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out = np.zeros((h, w), region.dtype)
    out[inside] = region[v[inside], u[inside]]
    return out


def _rotation_deg(Ta, Tb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Tb[:3, :3].T @ Ta[:3, :3]) - 1) / 2, -1, 1))))


def view_pose_residuals(T, true_deg, window=None):
    """For each view: median |pose rotation to a neighbour - recorded turntable step| over the neighbours."""
    window = CONFIG["view_residual_window"] if window is None else window
    n = len(T)
    out = []
    for i in range(n):
        res = []
        for k in range(1, window + 1):
            for j in ((i + k) % n, (i - k) % n):
                gap = abs(true_deg[i] - true_deg[j]) % 360
                res.append(abs(_rotation_deg(T[i], T[j]) - min(gap, 360 - gap)))
        out.append(float(np.median(res)))
    return np.array(out)


def count_witnesses(points_world, source, views, K, dist, tol_mm, deg=None):
    """Per point: how many other views should see it, how many confirm it, and the widest confirming angle.

    views: list of dict(index, T, depth_mm, mask, valid), skipping any dropped views.
    """
    n = len(points_world)
    should, witness = np.zeros(n, np.int32), np.zeros(n, np.int32)
    widest = np.zeros(n, np.float32)
    for v in views:
        if v["index"] == source:
            continue
        cam = geo.apply(v["w2c"], points_world)
        front = cam[:, 2] > 1e-6
        idx = np.nonzero(front)[0]
        if not len(idx):
            continue
        uv = np.rint(geo.project(cam[idx], K, dist)).astype(np.int64)
        h, w = v["mask"].shape
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx, uv = idx[inside], uv[inside]
        ux, uy = uv[:, 0], uv[:, 1]
        z_mm = cam[idx, 2] * 1000.0
        d = v["depth_mm"][uy, ux].astype(np.float64)
        occluded = (d > 0) & (d < z_mm - tol_mm)
        sees = v["mask"][uy, ux] & ~occluded
        confirms = sees & v["valid"][uy, ux] & (np.abs(d - z_mm) <= tol_mm)
        should[idx[sees]] += 1
        witness[idx[confirms]] += 1
        if deg is not None:
            gap = abs(deg[source] - deg[v["index"]]) % 360
            gap = min(gap, 360 - gap)
            hit = idx[confirms]
            widest[hit] = np.maximum(widest[hit], gap)
    return should, witness, widest


def statistical_outliers(points, neighbours, std_ratio):
    """True for inliers: mean distance to the neighbours is at most mean + std_ratio * std (Open3D's rule)."""
    if std_ratio <= 0 or len(points) <= neighbours:
        return np.ones(len(points), bool)
    dist, _ = cKDTree(points).query(points, k=neighbours + 1, workers=-1)
    mean = dist[:, 1:].mean(1)
    return mean <= mean.mean() + std_ratio * mean.std()


def write_ply(path, xyz, rgb=None, colour=None):
    """Binary little-endian PLY in the layout Open3D wrote for the existing clouds (double xyz, uchar rgb)."""
    v = np.zeros(len(xyz), dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = np.asarray(xyz).T
    rgb = np.tile(colour, (len(xyz), 1)) if colour is not None else rgb
    v["red"], v["green"], v["blue"] = np.asarray(rgb, np.uint8).T
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
                 "property double x\nproperty double y\nproperty double z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        v.tofile(f)


# ---------------------------------------------------------------------------
# a whole capture
# ---------------------------------------------------------------------------
def read_capture(raw_dir):
    raw = Path(raw_dir)
    cam = json.loads((raw / "camera_params.json").read_text())
    i = cam["color_intrinsic"]
    K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], float)
    dist = np.array(cam["color_distortion"]["opencv_order"], float)
    shots = json.loads((raw / "top_view_combine/stitch/poses.json").read_text())["shots"]
    return K, dist, (i["width"], i["height"]), float(cam["depth_scale_mm_per_unit"]), shots


def run_capture(raw_dir, out_dir, labels_cache=None, config=None, before_after=("000_ang000", "024_ang120")):
    raw, out = Path(raw_dir), Path(out_dir)
    cfg = {**CONFIG, **(config or {})}
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; write each run to a new folder")
    for sub in ("masks", "clouds_per_view", "clouds_final", "before_after"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    K, dist, wh, depth_scale, shots = read_capture(raw)
    rays = geo.ray_grid(K, dist, *wh)
    stems = [s["stem"] for s in shots]
    T = np.array([s["T_shot_to_ref"] for s in shots], float)
    deg = [s["true_deg"] for s in shots]

    residuals = view_pose_residuals(T, deg, cfg["view_residual_window"])
    dropped = [stems[i] for i in range(len(stems)) if residuals[i] > cfg["view_residual_max_deg"]]
    print(f"[VIEW] residual median {np.median(residuals):.2f} deg, limit {cfg['view_residual_max_deg']} deg; dropped: {dropped or 'none'}")

    views = []
    for i, stem in enumerate(stems):
        depth = cv2.imread(str(raw / f"depth_{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) * depth_scale
        mask = cv2.imread(str(raw / "top_view_combine/side/mask" / f"mask_{stem}.png"), cv2.IMREAD_GRAYSCALE) > 0
        labels = None
        if labels_cache:
            labels = labels_to_raw_grid(np.load(Path(labels_cache) / f"{stem}_region.npy"), rays, K)
        points, valid, stats = clean_view(depth, K, dist, mask, config=cfg, labels=labels, rays=rays)
        colour = cv2.cvtColor(cv2.imread(str(raw / f"color_{stem}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        write_ply(out / "clouds_per_view" / f"cloud_{stem}.ply", points, rgb=colour[valid])
        views.append(dict(index=i, stem=stem, T=T[i], w2c=np.linalg.inv(T[i]), depth_mm=depth, mask=mask, valid=valid,
                          points=geo.apply(T[i], points), rgb=colour[valid], labels=labels, stage_map=stats["stage_map"],
                          stats=stats))
    print(f"[VIEW] cleaned {len(views)} views")

    # Fusion: everything below acts on the views that survive the drop rule.
    used = [v for v in views if v["stem"] not in dropped]
    for v in views:
        if v["stem"] in dropped:
            v["stage_map"][v["stage_map"] == KEPT] = STAGE_CODE["view_drop"]
    report = dict(dropped_views=dropped)
    should_all, witness_all, widest_all = [], [], []
    for v in used:
        should, witness, widest = count_witnesses(v["points"], v["index"], used, K, dist, cfg["multiview_tol_mm"], deg)
        v["keep_mv"] = witness >= cfg["multiview_min_witnesses"]
        should_all.append(should)
        witness_all.append(witness)
        widest_all.append(widest)
    should_all, witness_all, widest_all = (np.concatenate(x) for x in (should_all, witness_all, widest_all))
    keep_mv_all = np.concatenate([v["keep_mv"] for v in used])
    min_w = cfg["multiview_min_witnesses"]
    report["multiview"] = dict(
        points_in=int(len(keep_mv_all)), kept=int(keep_mv_all.sum()),
        removed_too_few_views_should_see=int(((should_all < min_w) & ~keep_mv_all).sum()),
        removed_views_saw_no_surface=int(((should_all >= min_w) & ~keep_mv_all).sum()),
        median_views_should_see=float(np.median(should_all)), median_witnesses=float(np.median(witness_all)),
        kept_widest_witness_deg=dict(zip(("p10", "p50", "p90"), np.percentile(widest_all[keep_mv_all], [10, 50, 90]).tolist())),
        kept_share_no_witness_beyond_10deg=float((widest_all[keep_mv_all] <= 10.0).mean()),
        kept_share_no_witness_beyond_20deg=float((widest_all[keep_mv_all] <= 20.0).mean()))
    print(f"[MV  ] {report['multiview']}")

    survivors = np.vstack([v["points"][v["keep_mv"]] for v in used])
    inliers = statistical_outliers(survivors, cfg["sor_neighbours"], cfg["sor_std"])
    start = 0
    for v in used:
        rows, cols = np.nonzero(v["valid"])
        keep_mv = v["keep_mv"]
        v["stage_map"][rows[~keep_mv], cols[~keep_mv]] = STAGE_CODE["multiview"]
        n = int(keep_mv.sum())
        sor_ok = np.zeros(len(keep_mv), bool)
        sor_ok[keep_mv] = inliers[start:start + n]
        start += n
        out_sor = keep_mv & ~sor_ok
        v["stage_map"][rows[out_sor], cols[out_sor]] = STAGE_CODE["sor"]
        v["final"] = sor_ok
    print(f"[SOR ] {len(survivors):,} points in, {int(inliers.sum()):,} kept")

    all_points = np.vstack([v["points"][v["final"]] for v in used])
    all_colour = np.vstack([v["rgb"][v["final"]] for v in used]) / 255.0
    Pm, Cm = geo.voxel_fuse(all_points, all_colour, cfg["voxel_mm"] / 1000.0)
    write_ply(out / "side_fused_clean.ply", Pm, rgb=np.clip(np.rint(Cm * 255), 0, 255))
    report["fused"] = dict(points_before_voxel=int(len(all_points)), points=int(len(Pm)), voxel_mm=cfg["voxel_mm"])
    print(f"[SEED] {len(all_points):,} points -> {len(Pm):,} in {cfg['voxel_mm']:g} mm voxels")

    # Per-view outputs and counts, from the finished stage maps.
    regions = list(LABELS.values())
    per_view, totals = [], {r: {s: 0 for s in STAGES} | {"in_mask": 0, "kept": 0} for r in regions}
    for v in views:
        sm = v["stage_map"]
        np.savez_compressed(out / "masks" / f"{v['stem']}.npz", stage_map=sm)
        final = sm == KEPT
        if v["stem"] not in dropped:
            rows_final = geo.apply(v["w2c"], v["points"][v["final"]])
            write_ply(out / "clouds_final" / f"cloud_{v['stem']}.ply", rows_final, rgb=v["rgb"][v["final"]])
        row = dict(view=v["stem"], dropped=v["stem"] in dropped, residual_deg=float(residuals[v["index"]]))
        if v["labels"] is not None:
            for k, r in LABELS.items():
                pick = v["labels"] == k
                n_in = int((pick & (sm != NOT_IN_MASK)).sum())
                row[f"{r}_in_mask"] = n_in
                totals[r]["in_mask"] += n_in
                for s in STAGES:
                    c = int((pick & (sm == STAGE_CODE[s])).sum())
                    row[f"{r}_{s}"] = c
                    totals[r][s] += c
                c = int((pick & final).sum())
                row[f"{r}_kept"] = c
                totals[r]["kept"] += c
        per_view.append(row)
        for stem_ba in before_after:
            if v["stem"] == stem_ba:
                _write_before_after(out / "before_after", v, rays)

    with open(out / "stage_counts_per_view.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_view[0]))
        w.writeheader()
        w.writerows(per_view)
    with open(out / "stage_counts_total.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage"] + regions + ["total"])
        for s in ["in_mask"] + list(STAGES) + ["kept"]:
            w.writerow([s] + [totals[r][s] for r in regions] + [sum(totals[r][s] for r in regions)])

    report.update(config=cfg, overrides=config or {}, config_hash=config_hash(cfg), residual_deg=dict(zip(stems, residuals.tolist())), totals=totals)
    (out / "run.json").write_text(json.dumps(report, indent=2))
    return report


def _write_before_after(folder, v, rays):
    """Every masked pixel with depth, in the world frame: raw in red, the finally kept pixels in green."""
    sm = v["stage_map"]
    has_depth = (sm != NOT_IN_MASK) & (v["depth_mm"] > 0)
    world = geo.apply(v["T"], rays * (v["depth_mm"][..., None].astype(np.float64) / 1000.0))
    kept = sm == KEPT
    gone = has_depth & ~kept
    red, green = (255, 0, 0), (0, 255, 0)
    write_ply(folder / f"{v['stem']}_raw_red.ply", world[has_depth], colour=red)
    write_ply(folder / f"{v['stem']}_kept_green.ply", world[kept], colour=green)
    # Green and red coincide on kept pixels, so the merged file holds each point once: removed red, kept green.
    rgb = np.vstack([np.tile(red, (int(gone.sum()), 1)), np.tile(green, (int(kept.sum()), 1))])
    write_ply(folder / f"{v['stem']}_merged_removed_red_kept_green.ply", np.vstack([world[gone], world[kept]]), rgb=rgb)


class CleanRun:
    """A finished run, read back as a ToF reference. A run must equal CONFIG plus the overrides it declared,
    so the diagnostic can't silently score against cleaning that differs from what the run says it used."""

    def __init__(self, clean_dir, K, dist, wh):
        self.dir = Path(clean_dir)
        report = json.loads((self.dir / "run.json").read_text())
        self.overrides = report.get("overrides", {})
        wanted = json.loads(json.dumps({**CONFIG, **self.overrides}))
        if report["config"] != wanted:
            raise SystemExit(f"{self.dir} was cleaned with a different CONFIG than tof_clean.py has now; re-run it")
        self.config_hash = report["config_hash"]
        self.dropped = set(report["dropped_views"])
        self.maps = cv2.initUndistortRectifyMap(K, dist, None, K, tuple(wh), cv2.CV_32FC1)

    def stage_map(self, stem):
        return np.load(self.dir / "masks" / f"{stem}.npz")["stage_map"]

    def undistorted_valid(self, stem):
        """The seed's kept pixels of this view, on the undistorted grid (NEAREST, as the ToF targets)."""
        kept = (self.stage_map(stem) == KEPT).astype(np.uint8)
        return cv2.remap(kept, *self.maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0


def print_totals(totals):
    regions = list(LABELS.values())
    print("\n" + f"{'stage':14s}" + "".join(f"{r:>12s}" for r in regions) + f"{'total':>12s}")
    for s in ["in_mask"] + list(STAGES) + ["kept"]:
        vals = [totals[r][s] for r in regions]
        print(f"{s:14s}" + "".join(f"{x:12,d}" for x in vals) + f"{sum(vals):12,d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="a new folder")
    ap.add_argument("--labels_cache", type=Path, default=None, help="folder of <stem>_region.npy (diag_depth prepare), for per-region counts")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override one CONFIG value for this run")
    args = ap.parse_args()
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        if key not in CONFIG:
            raise SystemExit(f"unknown setting {key}; known: {sorted(CONFIG)}")
        overrides[key] = json.loads(value)
    report = run_capture(args.raw_dir, args.out, args.labels_cache, config=overrides or None)
    print_totals(report["totals"])
    print(f"[OUT ] {args.out}")


if __name__ == "__main__":
    main()
