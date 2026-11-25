#!/usr/bin/env python3
"""
diag_depth.py -- does a Gaussian-splat model's geometry match the measured ToF depth?

Stages:
    prepare   preflight checks, undistorted ToF targets + region labels -> <out>/tof_cache/
    gate      alignment sanity check on a few views (ratio, median signed error)
    metrics   every view, trained and init-only: disagreement with ToF by region, where it
              concentrates, PSNR, per-view CSV, summary.json, 4 figure panels
    export    2DGS: per-view median depth / alpha / normal maps, PLY, normalisation transform
    compare   3DGS vs 2DGS table on held-out views and side-by-side figures

    PY=~/miniconda3/envs/gsplat/bin/python
    $PY diag_depth.py prepare --run_dir <run> --data_dir <colmap> --raw_dir <capture>
    $PY diag_depth.py gate    --run_dir <run> --data_dir <colmap> --raw_dir <capture>
    $PY diag_depth.py metrics --run_dir <run> --data_dir <colmap> --raw_dir <capture>
    $PY diag_depth.py metrics --model 2dgs --tof_cache <3dgs run>/diag/tof_cache --run_dir <2dgs run> ...
    $PY diag_depth.py compare --run_dir <2dgs run> --ref_run_dir <3dgs run> ...

ToF reference:
    --tof_clean off   (default) depth > 0 + gradient test, the cache from the prepare stage
    --tof_clean on    the pixels flora_lib/tof_clean.py kept for the seed, from a run named by --clean_dir,
                      and views that run dropped are not scored. Regions and depth still come from
                      --tof_cache. Outputs go to <run>/diag_tof_clean, so results from off stay as they are.

Conventions (checked against the code and files, see the preflight):
    world units    metres; rendered z is converted by depth_mm = z_norm * 1000 / s, where s is the
                   uniform scale of gsplat Parser's normalisation transform
    depth type     gsplat "ED", the 2DGS median depth and the ToF are all z along the optical axis.
                   --model 3dgs scores ED; --model 2dgs scores the median depth, and ED as a
                   secondary row, because that is the only depth both models can render
    undistortion   initUndistortRectifyMap(K, dist, None, K, ...), as export_colmap_tvc.py;
                   the depth uses NEAREST so that no depth values are invented at leaf edges
    gradient test  tv_common.gradient_mask: Sobel / 8 (true mm/px), hole guard 1 px, computed on
                   the raw grid because NEAREST remapping creates artificial seams
    region labels  derived from colour, the SAM3 mask only knows pot + plant as a union

Nothing outside <out> is written.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from datasets.colmap import Parser
from gsplat import export_splats
from gsplat.rendering import rasterization, rasterization_2dgs

GRAD_MAX_MM_PER_PX = 20.0
HOLE_GUARD_PX = 1
ALPHA_MIN = 0.5
METRES_TO_MM = 1000.0

# Region labels stored in <stem>_region.npy
BACKGROUND, PLANT, POT, TAGS, UNASSIGNED = 0, 1, 2, 3, 4
REGION_NAMES = {PLANT: "plant", POT: "pot_no_tags", TAGS: "tags", UNASSIGNED: "unassigned"}

TAG_GUARD_PX = 3  # pot pixels this close to a tag are left out of the rigid reference
TAG_MIN_AREA_PX = 60
TAG_MAX_ASPECT = 5.0  # drops the thin dark band at the pot rim

GATE_RATIO_TOL = 0.01  # 1% of ~540 mm is ~5 mm
GATE_OFFSET_MM = 5.0

ZBUF_LEAK_WINDOW_PX = 5  # 1 mm seed spacing is ~1.4 px, so back surfaces show through the gaps
ZBUF_LEAK_TOL_MM = 10.0

EDGE_BAND_PX = 5  # ~3.6 mm at the plant's distance
INCIDENCE_BINS = [(0, 20), (20, 40), (40, 60), (60, 90)]
PANEL_VIEWS = [0, 24, 48, 64]  # held-out, spread around the turntable


# ---------------------------------------------------------------------------
# config, paths, small helpers
# ---------------------------------------------------------------------------
class _CfgLoader(yaml.SafeLoader):
    """cfg.yml holds Python tags (the training strategy object); those fields are not needed."""


_CfgLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)


def load_cfg(run_dir):
    """simple_trainer writes cfg.yml, simple_trainer_2dgs writes cfg.json."""
    run_dir = Path(run_dir)
    if (run_dir / "cfg.yml").is_file():
        return yaml.load((run_dir / "cfg.yml").read_text(), Loader=_CfgLoader)
    return json.loads((run_dir / "cfg.json").read_text())


def ellipse(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def read_ply_xyz(path):
    """Binary little-endian PLY as written by Open3D (double xyz, uchar rgb) -> xyz (N, 3)."""
    with open(path, "rb") as f:
        n, props = 0, []
        while True:
            line = f.readline().decode("ascii").split()
            if line[:2] == ["element", "vertex"]:
                n = int(line[2])
            elif line and line[0] == "property":
                props.append((line[2], {"double": "<f8", "float": "<f4", "uchar": "u1"}[line[1]]))
            elif line and line[0] == "end_header":
                break
        v = np.fromfile(f, dtype=np.dtype(props), count=n)
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def diag_folder(args):
    return "diag_tof_clean" if getattr(args, "tof_clean", "off") == "on" else "diag"


class Scene:
    """Parser with the training settings, plus the raw capture's camera and file layout."""

    def __init__(self, args):
        self.cfg = load_cfg(args.run_dir)
        self.model = getattr(args, "model", "3dgs")
        self.clean_run = None
        if not self.cfg["normalize_world_space"]:
            sys.exit("this run did not normalise world space; the depth scale logic does not apply")
        self.parser = Parser(str(args.data_dir), factor=self.cfg["data_factor"], normalize=True,
                             test_every=self.cfg["test_every"])
        p = self.parser
        T = p.transform
        self.scale = float(np.cbrt(np.linalg.det(T[:3, :3])))
        if np.abs(T[:3, :3].T @ T[:3, :3] / self.scale ** 2 - np.eye(3)).max() > 1e-9:
            sys.exit("normalisation transform is not a similarity")
        self.names = p.image_names
        heldout_file = Path(args.data_dir) / "heldout_views.json"
        self.heldout = set(json.loads(heldout_file.read_text())["names"]) if heldout_file.is_file() else None
        self.stems = [n[len("color_"):-len(".png")] for n in self.names]
        self.K = p.Ks_dict[p.camera_ids[0]].copy()
        self.wh = p.imsize_dict[p.camera_ids[0]]
        self.raw = Path(args.raw_dir)
        self.data = Path(args.data_dir)
        tof_clean_on = getattr(args, "tof_clean", "off") == "on"
        self.out = Path(args.out) if args.out else Path(args.run_dir) / diag_folder(args)
        self.cache = Path(args.tof_cache) if getattr(args, "tof_cache", None) else self.out / "tof_cache"
        if tof_clean_on:
            if not (args.clean_dir and getattr(args, "tof_cache", None)):
                sys.exit("--tof_clean on needs --clean_dir and --tof_cache")
            sys.path.insert(0, str(Path(os.environ.get("REPAS_VISION", Path.home() / "Projects/repas-vision")) / "flora_pipeline"))
            from flora_lib import tof_clean
            K, dist, wh, _ = self.raw_camera()
            self.clean_run = tof_clean.CleanRun(args.clean_dir, K, dist, wh)
            print(f"[TOF ] reference = seed cleaning, {args.clean_dir}, config {self.clean_run.config_hash}, "
                  f"overrides {self.clean_run.overrides}, dropped views {sorted(self.clean_run.dropped)}")

    def is_test(self, index):
        """The dataset's recorded held-out views if it has them (views removed from it would shift the index)."""
        if self.heldout is not None:
            return self.names[index] in self.heldout
        return index % self.cfg["test_every"] == 0

    def z_to_mm(self, z_norm):
        return z_norm * METRES_TO_MM / self.scale

    def raw_camera(self):
        c = json.loads((self.raw / "camera_params.json").read_text())
        i = c["color_intrinsic"]
        K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], float)
        return K, np.array(c["color_distortion"]["opencv_order"], float), (i["width"], i["height"]), c

    def mask_path(self, stem):
        return self.raw / "top_view_combine" / "side" / "mask" / f"mask_{stem}.png"

    def load_cached(self, stem):
        valid = self.clean_run.undistorted_valid(stem) if self.clean_run else self.base_valid(stem)
        return (np.load(self.cache / f"{stem}_depth_mm.npy"), valid, np.load(self.cache / f"{stem}_region.npy"))

    def base_valid(self, stem):
        """depth > 0 + gradient test, whatever the reference: where the depth steps and normals are read."""
        return np.load(self.cache / f"{stem}_valid.npy")

    def is_dropped(self, stem):
        return self.clean_run is not None and stem in self.clean_run.dropped


# ---------------------------------------------------------------------------
# prepare: ToF targets and regions
# ---------------------------------------------------------------------------
def gradient_ok(depth_mm):
    """Same test as top_view_combine/tv_common.gradient_mask."""
    d = depth_mm.astype(np.float32)
    d = np.where(d > 0, d, cv2.dilate(d, ellipse(HOLE_GUARD_PX)))
    g = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    return g < GRAD_MAX_MM_PER_PX


def split_regions(bgr, mask):
    """Label the masked pixels as plant / pot without tags / tags / unassigned, from colour."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = (hsv[..., k].astype(int) for k in range(3))
    g, r = bgr[..., 1].astype(int), bgr[..., 2].astype(int)
    yellow = mask & (H >= 18) & (H <= 34) & (S >= 90) & (V >= 70) & (r >= g * 0.95)
    green = mask & (H >= 30) & (H <= 90) & (S >= 40) & (g > r * 1.02) & ~yellow

    # Pot = convex hull of the largest yellow blob: a cylinder's silhouette is convex, and tags
    # cut into the blob's outline at the silhouette, so filling holes would not be enough.
    opened = cv2.morphologyEx(yellow.astype(np.uint8), cv2.MORPH_OPEN, ellipse(1))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n < 2:
        raise ValueError("no yellow pot found")
    blob = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    hull = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(blob)), 1)
    hull = (hull > 0) & mask

    # A leaf in front of the pot is plant, not pot; ignore green specks inside tags.
    n, lab, stats, _ = cv2.connectedComponentsWithStats((green & hull).astype(np.uint8), connectivity=8)
    leaf_on_pot = np.isin(lab, 1 + np.nonzero(stats[1:, cv2.CC_STAT_AREA] >= 30)[0])

    dark_or_white = hull & ~yellow & ~green
    closed = cv2.morphologyEx(dark_or_white.astype(np.uint8), cv2.MORPH_CLOSE, ellipse(3))
    tags = np.zeros(mask.shape, np.uint8)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        if cv2.contourArea(c) >= TAG_MIN_AREA_PX and max(w, h) <= TAG_MAX_ASPECT * max(min(w, h), 1):
            cv2.drawContours(tags, [c], -1, 1, cv2.FILLED)
    tags = (tags > 0) & hull & ~leaf_on_pot

    guard = cv2.dilate((tags | dark_or_white).astype(np.uint8), ellipse(TAG_GUARD_PX)) > 0
    pot = yellow & hull & ~guard & ~leaf_on_pot

    region = np.full(mask.shape, BACKGROUND, np.uint8)
    region[mask] = UNASSIGNED
    region[mask & ~hull & ~yellow] = PLANT
    region[leaf_on_pot] = PLANT
    region[pot] = POT
    region[tags] = TAGS
    return region


def aruco_coverage(bgr, tags_mask, detector):
    """Share of each detected tag's quad inside the colour-derived tag region."""
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    out = []
    for quad in corners if ids is not None else []:
        m = np.zeros(tags_mask.shape, np.uint8)
        cv2.fillConvexPoly(m, quad.astype(np.int32).reshape(-1, 2), 1)
        out.append(float(tags_mask[m > 0].mean()))
    return out


def overlay(bgr, region):
    vis = bgr.copy()
    colours = {PLANT: (60, 200, 60), POT: (0, 220, 255), TAGS: (255, 0, 255), UNASSIGNED: (0, 0, 255)}
    for label, colour in colours.items():
        vis[region == label] = colour
    return vis


def crop_to(im, mask, pad=8):
    y, x = np.nonzero(mask)
    return im[max(y.min() - pad, 0):y.max() + pad, max(x.min() - pad, 0):x.max() + pad]


def stack_rows(rows):
    width = max(r.shape[1] for r in rows)
    return np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows])


def cmd_prepare(args):
    if args.tof_clean == "on":
        sys.exit("prepare builds the uncleaned targets; run it with --tof_clean off")
    s = Scene(args)
    s.cache.mkdir(parents=True, exist_ok=True)
    K, dist, wh, cam = s.raw_camera()
    checks = {}

    def check(name, ok, detail):
        checks[name] = dict(ok=bool(ok), detail=detail)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    check("K matches COLMAP", np.abs(K - s.K).max() == 0 and tuple(wh) == tuple(s.wh),
          f"max diff {np.abs(K - s.K).max()}, size {wh}")
    scale_mm = cam["depth_scale_mm_per_unit"]
    check("depth scale is 1 mm/unit", scale_mm == 1.0, scale_mm)

    raw_parser = Parser(str(s.data), factor=s.cfg["data_factor"], normalize=False, test_every=s.cfg["test_every"])
    poses = {x["stem"]: np.asarray(x["T_shot_to_ref"], float)
             for x in json.loads((s.raw / "top_view_combine/stitch/poses.json").read_text())["shots"]}
    t_err = max(np.abs(poses[st] - c2w).max() for st, c2w in zip(s.stems, raw_parser.camtoworlds))
    check("poses.json matches sparse/0", t_err < 1e-9, f"max entry diff {t_err:.2e}")
    seed = read_ply_xyz(s.raw / "top_view_combine/stitch/side_fused.ply")
    same = len(seed) == len(raw_parser.points)
    seed_err = np.abs(seed - raw_parser.points).max() if same else np.inf
    check("seed points are side_fused.ply", seed_err < 1e-5, f"{len(seed)} points, max diff {seed_err * 1000:.2e} mm")

    maps = cv2.initUndistortRectifyMap(K, dist, None, K, wh, cv2.CV_32FC1)
    aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                    cv2.aruco.DetectorParameters())
    rows, tag_cover, col_diff, mask_diff, self_err, sheet = [], [], [], [], [], []
    t0 = time.time()
    for index, stem in enumerate(s.stems):
        trained_on = cv2.imread(str(s.data / "images" / f"color_{stem}.png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(s.raw / f"depth_{stem}.png"), cv2.IMREAD_UNCHANGED)
        raw_mask = cv2.imread(str(s.mask_path(stem)), cv2.IMREAD_GRAYSCALE)
        raw_col = cv2.imread(str(s.raw / f"color_{stem}.png"), cv2.IMREAD_COLOR)
        if depth is None or depth.dtype != np.uint16 or depth.shape != (wh[1], wh[0]):
            sys.exit(f"bad depth image for {stem}")

        mask = cv2.remap(raw_mask, *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
        col = cv2.remap(raw_col, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        col[~mask] = 0
        col_diff.append(int(np.abs(col.astype(int) - trained_on.astype(int)).max()))
        mask_diff.append(int((mask != (trained_on.sum(2) > 0)).sum()))

        ok_raw = ((depth > 0) & gradient_ok(depth)).astype(np.uint8)
        depth_u = cv2.remap(depth, *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.remap(ok_raw, *maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
        region = split_regions(trained_on, mask)
        cover = aruco_coverage(trained_on, region == TAGS, aruco)
        tag_cover += cover

        # The shot's own cleaned cloud came from this depth, so it must land on it again.
        P = read_ply_xyz(s.raw / "top_view_combine/side/pointcloud" / f"cloud_{stem}.ply")
        u = np.rint(K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]).astype(int)
        v = np.rint(K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]).astype(int)
        keep = (u >= 0) & (u < wh[0]) & (v >= 0) & (v < wh[1])
        d_at = depth_u[v[keep], u[keep]].astype(float)
        e = P[keep, 2] * 1000.0 - d_at
        self_err.append(float(np.median(np.abs(e[d_at > 0]))))

        np.save(s.cache / f"{stem}_depth_mm.npy", depth_u)
        np.save(s.cache / f"{stem}_valid.npy", valid)
        np.save(s.cache / f"{stem}_region.npy", region)

        row = dict(stem=stem, split="test" if s.is_test(index) else "train", mask_px=int(mask.sum()),
                   depth_nonzero_px=int(((depth_u > 0) & mask).sum()), tof_valid_px=int((valid & mask).sum()),
                   tags_detected=len(cover))
        for label, name in REGION_NAMES.items():
            row[f"{name}_px"] = int((region == label).sum())
            row[f"{name}_valid_px"] = int(((region == label) & valid).sum())
        rows.append(row)
        if index % 12 == 0:
            sheet.append(np.hstack([crop_to(trained_on, mask), crop_to(overlay(trained_on, region), mask)]))
    print(f"[TIME] {len(s.stems)} views in {time.time() - t0:.0f} s")

    check("undistorted colour reproduces images/", max(col_diff) == 0, f"max abs diff {max(col_diff)}")
    check("undistorted mask reproduces images/", max(mask_diff) == 0, f"max px diff {max(mask_diff)}")
    check("single-shot clouds land on undistorted depth", max(self_err) < 1.0,
          f"median |z - depth| per view: worst {max(self_err):.3f} mm, median {np.median(self_err):.3f} mm")

    tag_cover = np.array(tag_cover)
    print(f"[TAGS] aruco found {len(tag_cover)} tags in {len(s.stems)} views; share of each quad inside the "
          f"colour tag region: min {tag_cover.min():.3f}, p5 {np.percentile(tag_cover, 5):.3f}, "
          f"median {np.median(tag_cover):.3f}")
    with open(s.out / "regions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cv2.imwrite(str(s.out / "regions_overlay.png"), stack_rows(sheet))
    total = {name: sum(r[f"{name}_px"] for r in rows) for name in REGION_NAMES.values()}
    total_mask = sum(r["mask_px"] for r in rows)
    print("[AREA] share of mask pixels: " + ", ".join(f"{k} {v / total_mask * 100:.2f}%" for k, v in total.items()))
    print(f"[TOF ] share of mask pixels valid after depth>0 and gradient: "
          f"{sum(r['tof_valid_px'] for r in rows) / total_mask * 100:.1f}%")
    (s.out / "prepare.json").write_text(json.dumps(dict(
        checks=checks, scale=s.scale, grad_max_mm_per_px=GRAD_MAX_MM_PER_PX, hole_guard_px=HOLE_GUARD_PX,
        tag_guard_px=TAG_GUARD_PX, aruco=dict(n=int(len(tag_cover)), min=float(tag_cover.min()),
                                              median=float(np.median(tag_cover)))), indent=2))
    if not all(c["ok"] for c in checks.values()):
        sys.exit("preflight failed -- see FAIL lines above")
    print(f"[OUT ] {s.cache}")


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def load_trained(ckpt, device):
    data = torch.load(ckpt, map_location=device, weights_only=False)
    return {k: v.detach() for k, v in data["splats"].items()}


def build_init_splats(scene, device):
    """The Gaussians the trainer starts from, from its own function with this run's cfg."""
    from utils import set_random_seed
    cfg = scene.cfg
    if scene.model == "2dgs":
        import simple_trainer_2dgs  # imported here because it pulls in the viewer and logging stack
        set_random_seed(42)
        splats, _ = simple_trainer_2dgs.create_splats_with_optimizers(
            scene.parser, init_type=cfg["init_type"], init_opacity=cfg["init_opa"], init_scale=cfg["init_scale"],
            scene_scale=scene.parser.scene_scale * 1.1 * cfg["global_scale"], sh_degree=cfg["sh_degree"],
            sparse_grad=cfg["sparse_grad"], batch_size=cfg["batch_size"], device=device)
        return {k: v.detach() for k, v in splats.items()}
    import simple_trainer  # imported here because it pulls in the viewer and logging stack
    set_random_seed(42)
    splats, _ = simple_trainer.create_splats_with_optimizers(
        scene.parser, init_type=cfg["init_type"], init_opacity=cfg["init_opa"], init_scale=cfg["init_scale"],
        scene_scale=scene.parser.scene_scale * 1.1 * cfg["global_scale"], sh_degree=cfg["sh_degree"],
        sparse_grad=cfg["sparse_grad"], visible_adam=cfg["visible_adam"], batch_size=cfg["batch_size"],
        device=device)
    return {k: v.detach() for k, v in splats.items()}


def render(scene, splats, index, device):
    """-> rgb (H, W, 3) in 0-1, the model's primary z-depth in mm, alpha."""
    if scene.model == "2dgs":
        v = render_2dgs(scene, splats, index, device)
        return v["rgb"], v["depth_mm"], v["alpha"]
    return render_3dgs(scene, splats, index, device)


@torch.no_grad()
def render_2dgs(scene, splats, index, device):
    """Same call as simple_trainer_2dgs.eval. Normals are unit vectors in the camera frame, 0 = none."""
    cfg = scene.cfg
    w, h = scene.wh
    c2w = torch.from_numpy(scene.parser.camtoworlds[index]).float().to(device)
    K = torch.from_numpy(scene.K).float().to(device)
    out, alpha, normals, _, _, median, _ = rasterization_2dgs(
        means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]), colors=torch.cat([splats["sh0"], splats["shN"]], 1),
        viewmats=torch.linalg.inv(c2w)[None], Ks=K[None], width=w, height=h, sh_degree=cfg["sh_degree"],
        near_plane=cfg["near_plane"], far_plane=cfg["far_plane"], render_mode="RGB+ED", packed=cfg["packed"])
    out = out[0].cpu().numpy()
    normals_cam = normals[0] @ c2w[:3, :3]  # the renderer returns them in the world frame
    normals_cam = torch.nn.functional.normalize(normals_cam, dim=-1).cpu().numpy()
    return dict(rgb=np.clip(out[..., :3], 0.0, 1.0), depth_mm=scene.z_to_mm(median[0, ..., 0].cpu().numpy()),
                ed_mm=scene.z_to_mm(out[..., 3]), alpha=alpha[0, ..., 0].cpu().numpy(), normals=normals_cam)


@torch.no_grad()
def render_3dgs(scene, splats, index, device):
    cfg = scene.cfg
    w, h = scene.wh
    c2w = torch.from_numpy(scene.parser.camtoworlds[index]).float().to(device)
    K = torch.from_numpy(scene.K).float().to(device)
    out, alpha, _ = rasterization(
        means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]), colors=torch.cat([splats["sh0"], splats["shN"]], 1),
        viewmats=torch.linalg.inv(c2w)[None], Ks=K[None], width=w, height=h, sh_degree=cfg["sh_degree"],
        near_plane=cfg["near_plane"], far_plane=cfg["far_plane"], render_mode="RGB+ED", packed=cfg["packed"],
        rasterize_mode="antialiased" if cfg["antialiased"] else "classic", camera_model=cfg["camera_model"],
        with_ut=cfg["with_ut"], with_eval3d=cfg["with_eval3d"])
    out = out[0].cpu().numpy()
    return np.clip(out[..., :3], 0.0, 1.0), scene.z_to_mm(out[..., 3]), alpha[0, ..., 0].cpu().numpy()


def zbuffer_seed(scene, index):
    """Nearest seed point per pixel in mm (0 = none), with see-through points removed."""
    w, h = scene.wh
    w2c = np.linalg.inv(scene.parser.camtoworlds[index])
    X = scene.parser.points @ w2c[:3, :3].T + w2c[:3, 3]
    front = X[:, 2] > 1e-6
    u = np.rint(scene.K[0, 0] * X[front, 0] / X[front, 2] + scene.K[0, 2]).astype(int)
    v = np.rint(scene.K[1, 1] * X[front, 1] / X[front, 2] + scene.K[1, 2]).astype(int)
    z = scene.z_to_mm(X[front, 2])
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    zbuf = np.full((h, w), np.inf, np.float32)
    np.minimum.at(zbuf, (v[keep], u[keep]), z[keep].astype(np.float32))
    nearest = cv2.erode(zbuf, np.ones((ZBUF_LEAK_WINDOW_PX, ZBUF_LEAK_WINDOW_PX), np.uint8))
    leak = np.isfinite(zbuf) & (zbuf > nearest + ZBUF_LEAK_TOL_MM)
    zbuf[leak | ~np.isfinite(zbuf)] = 0
    return zbuf, int(leak.sum())


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
def gate_row(model, stem, name, pick, rendered, tof, n_candidates):
    n = int(pick.sum())
    row = dict(model=model, view=stem, region=name, n=n, candidates=n_candidates,
               coverage=n / n_candidates if n_candidates else float("nan"))
    if n == 0:
        return dict(row, med_rendered=float("nan"), med_tof=float("nan"), ratio=float("nan"),
                    med_signed=float("nan"), med_abs=float("nan"))
    r, t = rendered[pick].astype(np.float64), tof[pick].astype(np.float64)
    return dict(row, med_rendered=float(np.median(r)), med_tof=float(np.median(t)),
                ratio=float(np.median(r) / np.median(t)), med_signed=float(np.median(r - t)),
                med_abs=float(np.median(np.abs(r - t))))


def signed_error_colour(err_mm, usable, limit=10.0):
    x = np.clip(err_mm / limit, -1, 1)
    vis = np.zeros(err_mm.shape + (3,), np.uint8)
    blue, red = np.clip(x, 0, 1), np.clip(-x, 0, 1)  # BGR: rendered behind ToF = red, in front = blue
    vis[..., 0] = (255 * (1 - blue)).astype(np.uint8)
    vis[..., 1] = (255 * (1 - np.abs(x))).astype(np.uint8)
    vis[..., 2] = (255 * (1 - red)).astype(np.uint8)
    vis[~usable] = 0
    return vis


def depth_colour(depth_mm, usable, lo, hi):
    vis = cv2.applyColorMap((np.clip((depth_mm - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    vis[~usable] = 0
    return vis


def cmd_gate(args):
    s = Scene(args)
    if not s.cache.is_dir():
        sys.exit("run the prepare stage first")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = default_ckpt(args)
    views = [next(i for i, st in enumerate(s.stems) if st.startswith(f"{v:03d}_")) for v in args.views]
    print(f"[RUN ] ckpt {ckpt}\n[RUN ] scale s = {s.scale:.6f}, depth_mm = z_norm * {METRES_TO_MM / s.scale:.4f}")

    trained = load_trained(ckpt, device)
    print(f"[RUN ] trained splats: {len(trained['means'])}")

    # The render path must give back the PSNR recorded at training, or nothing after it can be trusted.
    psnrs = []
    for i in range(len(s.stems)):
        if s.is_test(i):
            rgb, _, _ = render(s, trained, i, device)
            gt = cv2.cvtColor(cv2.imread(str(s.data / "images" / s.names[i])), cv2.COLOR_BGR2RGB) / 255.0
            psnrs.append(-10 * np.log10(np.mean((rgb - gt) ** 2)))
    val_stats = step_stats(args.run_dir, "val", ckpt_step(ckpt))
    recorded = val_stats["psnr"] if val_stats else float("nan")
    render_ok = abs(np.mean(psnrs) - recorded) < 0.1
    print(f"[{'PASS' if render_ok else 'FAIL'}] render path: test-view PSNR {np.mean(psnrs):.3f} dB, "
          f"recorded at training {recorded:.3f} dB")

    init = build_init_splats(s, device)
    print(f"[RUN ] init splats from the trainer's create_splats_with_optimizers: {len(init['means'])}, "
          f"opacity {torch.sigmoid(init['opacities'][0]).item():.3f}, "
          f"median scale {s.z_to_mm(torch.exp(init['scales']).median().item()):.3f} mm")

    rows, panels = [], []
    for i in views:
        stem = s.stems[i]
        if s.is_dropped(stem):
            print(f"[GATE] {stem}: dropped by the cleaning run, not scored")
            continue
        tof, valid, region = s.load_cached(stem)
        mask = region != BACKGROUND
        rgb, d_tr, a_tr = render(s, trained, i, device)
        _, d_in, a_in = render(s, init, i, device)
        d_zb, n_leak = zbuffer_seed(s, i)
        print(f"[ZBUF] {stem}: removed {n_leak} see-through pixels")
        for model, rendered, usable in (("trained", d_tr, a_tr > ALPHA_MIN), ("init_render", d_in, a_in > ALPHA_MIN),
                                        ("init_zbuf", d_zb, d_zb > 0)):
            for name, sel in [("all_mask", mask)] + [(n, region == lab) for lab, n in REGION_NAMES.items() if lab != UNASSIGNED]:
                cand = sel & valid
                rows.append(gate_row(model, stem, name, cand & usable, rendered, tof, int(cand.sum())))
        ok = mask & valid & (a_tr > ALPHA_MIN)
        lo, hi = np.percentile(tof[ok], [1, 99])
        panel = [cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), depth_colour(d_tr, ok, lo, hi),
                 depth_colour(tof.astype(np.float32), mask & valid, lo, hi),
                 signed_error_colour(d_tr - tof.astype(np.float32), ok)]
        panels.append(np.hstack([crop_to(p, mask) for p in panel]))

    s.out.mkdir(parents=True, exist_ok=True)
    with open(s.out / "gate.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cv2.imwrite(str(s.out / "gate_panels.png"), stack_rows(panels))

    print(f"\n{'model':12s} {'view':11s} {'region':12s} {'n':>7s} {'cover':>6s} {'rend':>7s} {'tof':>7s} "
          f"{'ratio':>7s} {'signed':>7s} {'|e|':>6s}")
    for r in rows:
        print(f"{r['model']:12s} {r['view']:11s} {r['region']:12s} {r['n']:7d} {r['coverage']:6.3f} "
              f"{r['med_rendered']:7.1f} {r['med_tof']:7.1f} {r['ratio']:7.4f} {r['med_signed']:7.2f} {r['med_abs']:6.2f}")

    primary = [r for r in rows if r["model"] == "trained" and r["region"] == "all_mask"]
    passed = render_ok and all(abs(r["ratio"] - 1) <= GATE_RATIO_TOL and abs(r["med_signed"]) <= GATE_OFFSET_MM
                               for r in primary)
    print(f"\n[GATE] trained, all mask pixels, ratio within 1 +/- {GATE_RATIO_TOL} and |median signed| <= "
          f"{GATE_OFFSET_MM} mm, render path check: {'PASS' if passed else 'FAIL'}")
    (s.out / "gate.json").write_text(json.dumps(dict(
        passed=bool(passed), scale=s.scale, ckpt=str(ckpt), psnr_rendered=float(np.mean(psnrs)),
        psnr_recorded=recorded, ratio_tol=GATE_RATIO_TOL, offset_tol_mm=GATE_OFFSET_MM, rows=rows), indent=2))
    print(f"[OUT ] {s.out / 'gate.csv'}\n[OUT ] {s.out / 'gate_panels.png'}")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def ckpt_step(path):
    """ckpt_6999.pt or ckpt_6999_rank0.pt -> 6999."""
    return int(Path(path).stem.split("_")[1])


def default_ckpt(args):
    """The latest step; a name sort would put ckpt_29999 before ckpt_6999."""
    return args.ckpt or max((Path(args.run_dir) / "ckpts").glob("*.pt"), key=ckpt_step)


def step_stats(run_dir, kind, step):
    """stats/<kind>_step<step>[_rank0].json, or None."""
    found = sorted((Path(run_dir) / "stats").glob(f"{kind}_step{step:04d}*.json"))
    return json.loads(found[0].read_text()) if found else None


def error_stats(e):
    """e = rendered - ToF in mm; negative means the render is closer to the camera."""
    if len(e) == 0:
        return dict(n=0, median_abs=float("nan"), bias_mean=float("nan"), median_signed=float("nan"),
                    p90_abs=float("nan"), pct_3mm=float("nan"), pct_5mm=float("nan"), pct_10mm=float("nan"))
    a = np.abs(e)
    return dict(n=int(len(e)), median_abs=float(np.median(a)), bias_mean=float(np.mean(e)),
                median_signed=float(np.median(e)), p90_abs=float(np.percentile(a, 90)),
                pct_3mm=float((a <= 3).mean() * 100), pct_5mm=float((a <= 5).mean() * 100),
                pct_10mm=float((a <= 10).mean() * 100))


def psnr(rgb, gt, pick=None):
    d = (rgb - gt) ** 2
    return float(-10 * np.log10(d.mean() if pick is None else d[pick].mean()))


def normals_from_z(z_mm, K, ok):
    """Unit camera-frame normals by central differences, facing the camera; 0 where unknown."""
    h, w = z_mm.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    X = np.stack([(u - K[0, 2]) / K[0, 0] * z_mm, (v - K[1, 2]) / K[1, 1] * z_mm, z_mm], -1)
    n = np.zeros((h, w, 3), np.float32)
    n[1:-1, 1:-1] = np.cross(X[2:, 1:-1] - X[:-2, 1:-1], X[1:-1, 2:] - X[1:-1, :-2])
    known = np.zeros((h, w), bool)
    known[1:-1, 1:-1] = ok[2:, 1:-1] & ok[:-2, 1:-1] & ok[1:-1, 2:] & ok[1:-1, :-2] & ok[1:-1, 1:-1]
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    known &= length[..., 0] > 0
    n = np.divide(n, length, out=np.zeros_like(n), where=length > 0)
    n[~known] = 0
    return n


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0)))


def view_rays(K, wh):
    u, v = np.meshgrid(np.arange(wh[0]), np.arange(wh[1]))
    r = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def normal_colour(n):
    vis = ((n * 0.5 + 0.5) * 255).astype(np.uint8)[..., ::-1].copy()
    vis[(n == 0).all(-1)] = 0
    return vis


def labelled(im, text):
    bar = np.full((22, im.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


def location_masks(mask, valid, region, inc):
    """Plant pixels by where they are, and by incidence angle (taken from the ToF, so shared by all models)."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    edge = mask & (dist <= EDGE_BAND_PX)
    step = (cv2.dilate((mask & ~valid).astype(np.uint8), ellipse(EDGE_BAND_PX)) > 0) & mask & ~edge
    plant = region == PLANT
    out = {"silhouette_band": plant & edge, "near_depth_step": plant & step, "interior": plant & ~edge & ~step}
    for lo, hi in INCIDENCE_BINS:
        out[f"incidence_{lo}_{hi}"] = plant & (inc >= lo) & (inc < hi)
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cmd_metrics(args):
    s = Scene(args)
    if not s.cache.is_dir():
        sys.exit(f"no ToF cache at {s.cache}: run the prepare stage first or pass --tof_cache")
    s.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = default_ckpt(args)
    trained = load_trained(ckpt, device)
    init = build_init_splats(s, device)
    print(f"[RUN ] {s.model}, ckpt {ckpt}: {len(trained['means'])} splats; init {len(init['means'])} splats")
    rays = view_rays(s.K, s.wh)
    panel_views = {f"{v:03d}" for v in PANEL_VIEWS}

    pools, loc_pools, psnr_rows, rows, angles = defaultdict(list), defaultdict(list), [], [], defaultdict(list)
    alpha_bg = defaultdict(list)
    t0 = time.time()
    for i, stem in enumerate(s.stems):
        split = "test" if s.is_test(i) else "train"
        if s.is_dropped(stem):
            continue
        tof, valid, region = s.load_cached(stem)
        base = s.base_valid(stem)
        tof = tof.astype(np.float32)
        mask = region != BACKGROUND
        gt = cv2.cvtColor(cv2.imread(str(s.data / "images" / s.names[i])), cv2.COLOR_BGR2RGB) / 255.0

        tr = render_2dgs(s, trained, i, device) if s.model == "2dgs" else None
        rgb, depth, alpha = (tr["rgb"], tr["depth_mm"], tr["alpha"]) if tr else render(s, trained, i, device)
        rgb_i, depth_i, alpha_i = render(s, init, i, device)
        scored = [("trained", rgb, depth, alpha), ("init", rgb_i, depth_i, alpha_i)]
        if tr:
            scored.append(("trained_ed", rgb, tr["ed_mm"], alpha))

        tof_normals = normals_from_z(cv2.medianBlur(tof, 5), s.K, base & mask)
        inc = angle_deg(tof_normals, -rays)
        inc[(tof_normals == 0).all(-1)] = -1
        locs = location_masks(mask, base, region, inc)
        regions = [("all_mask", mask)] + [(n, region == lab) for lab, n in REGION_NAMES.items() if lab != UNASSIGNED]

        for model, c, d, a in scored:
            usable = a > ALPHA_MIN
            full, inside = psnr(c, gt), psnr(c, gt, mask)
            psnr_rows.append(dict(model=model, view=stem, split=split, psnr_full=full, psnr_mask=inside))
            alpha_bg[(model, split)].append(float(usable[~mask].mean()))
            err = d - tof
            for name, sel in regions:
                cand = sel & valid
                e = err[cand & usable]
                pools[(model, split, name)].append(e)
                rows.append(dict(model=model, view=stem, split=split, region=name, candidates=int(cand.sum()),
                                 coverage=float(usable[cand].mean()) if cand.any() else float("nan"),
                                 **error_stats(e), psnr_full=full, psnr_mask=inside))
            for name, sel in locs.items():
                loc_pools[(model, split, name)].append(err[sel & valid & usable])

        # Normal disagreement with the ToF surface, from depth gradients for every model.
        ok = mask & valid & (alpha > ALPHA_MIN)
        grad_normals = normals_from_z(depth, s.K, mask & base & (alpha > ALPHA_MIN))
        for name, sel in regions:
            both = sel & ok & (tof_normals != 0).any(-1) & (grad_normals != 0).any(-1)
            angles[("depth_gradient", split, name)].append(angle_deg(grad_normals, tof_normals)[both])
            if tr:
                both = sel & ok & (tof_normals != 0).any(-1)
                angles[("rendered", split, name)].append(angle_deg(tr["normals"], tof_normals)[both])

        if stem[:3] in panel_views:
            lo, hi = np.percentile(tof[mask & valid], [1, 99])
            cols = [("RGB render", cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)),
                    (f"{'median' if tr else 'ED'} depth {lo:.0f}-{hi:.0f} mm", depth_colour(depth, ok, lo, hi)),
                    ("ToF", depth_colour(tof, mask & valid, lo, hi)),
                    ("render - ToF, +/-10 mm", signed_error_colour(depth - tof, ok)),
                    ("normals from depth", normal_colour(grad_normals))]
            if tr:
                vis = normal_colour(tr["normals"])
                vis[~ok] = 0
                cols.append(("rendered normals", vis))
            row = np.hstack([labelled(crop_to(c, mask), t) for t, c in cols])
            cv2.imwrite(str(s.out / f"panel_{stem}.png"), row)
    print(f"[TIME] {len(s.stems)} views in {time.time() - t0:.0f} s")

    write_csv(s.out / "per_view_metrics.csv", rows)
    summary = dict(model=s.model, ckpt=str(ckpt), n_gaussians=len(trained["means"]), n_init=len(init["means"]),
                   train_stats=step_stats(args.run_dir, "train", ckpt_step(ckpt)),
                   recorded_val=step_stats(args.run_dir, "val", ckpt_step(ckpt)),
                   regions={}, locations={}, psnr={}, normal_angle_deg={}, alpha_outside_mask={})
    for (model, split, name), v in pools.items():
        summary["regions"].setdefault(model, {}).setdefault(split, {})[name] = error_stats(np.concatenate(v))
    for (model, split, name), v in loc_pools.items():
        summary["locations"].setdefault(model, {}).setdefault(split, {})[name] = error_stats(np.concatenate(v))
    for r in psnr_rows:
        summary["psnr"].setdefault(r["model"], {}).setdefault(r["split"], []).append((r["psnr_full"], r["psnr_mask"]))
    summary["psnr"] = {m: {sp: dict(full=float(np.mean([x[0] for x in v])), mask=float(np.mean([x[1] for x in v])))
                           for sp, v in d.items()} for m, d in summary["psnr"].items()}
    for (kind, split, name), v in angles.items():
        summary["normal_angle_deg"].setdefault(kind, {}).setdefault(split, {})[name] = float(np.median(np.concatenate(v)))
    for (model, split), v in alpha_bg.items():
        summary["alpha_outside_mask"].setdefault(model, {})[split] = float(np.mean(v))
    if s.clean_run:
        summary["tof_clean"] = dict(clean_dir=str(args.clean_dir), config_hash=s.clean_run.config_hash,
                                    overrides=s.clean_run.overrides, dropped_views=sorted(s.clean_run.dropped))
    (s.out / "summary.json").write_text(json.dumps(summary, indent=2))

    recorded = summary["recorded_val"]["psnr"] if summary["recorded_val"] else float("nan")
    got = summary["psnr"]["trained"]["test"]["full"]
    print(f"[{'PASS' if abs(got - recorded) < 0.1 else 'FAIL'}] render path: test PSNR {got:.3f} dB, recorded {recorded:.3f} dB")
    for split in ("test", "train"):
        print(f"\n--- {split} ---   (disagreement with ToF, rendered - ToF; mm)")
        print(f"{'model':11s} {'region':12s} {'|e| med':>8s} {'bias':>7s} {'p90':>7s} {'<=3':>6s} {'<=5':>6s} {'<=10':>6s} {'n':>10s}")
        for model, d in summary["regions"].items():
            for name, r in d[split].items():
                print(f"{model:11s} {name:12s} {r['median_abs']:8.2f} {r['bias_mean']:7.2f} {r['p90_abs']:7.2f} "
                      f"{r['pct_3mm']:6.1f} {r['pct_5mm']:6.1f} {r['pct_10mm']:6.1f} {r['n']:10d}")
    print("\n--- where, plant, test ---")
    for model, d in summary["locations"].items():
        for name, r in d["test"].items():
            print(f"{model:11s} {name:17s} {r['median_abs']:8.2f} {r['bias_mean']:7.2f} {r['p90_abs']:7.2f} {r['n']:9d}")
    print("\n" + json.dumps(dict(psnr=summary["psnr"], normal_angle_deg=summary["normal_angle_deg"],
                                 alpha_outside_mask=summary["alpha_outside_mask"]), indent=1))
    print(f"[OUT ] {s.out}")


# ---------------------------------------------------------------------------
# export (2DGS maps, PLY, transform) and compare
# ---------------------------------------------------------------------------
def cmd_export(args):
    if args.model != "2dgs":
        sys.exit("export is for --model 2dgs (the 3DGS trainer already wrote its PLY)")
    s = Scene(args)
    run = Path(args.run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = default_ckpt(args)
    data = torch.load(ckpt, map_location=device, weights_only=False)
    splats = {k: v.detach() for k, v in data["splats"].items()}

    raw = Parser(str(s.data), factor=s.cfg["data_factor"], normalize=False, test_every=s.cfg["test_every"])
    (run / "normalisation.json").write_text(json.dumps(dict(
        note="x_normalised = transform @ x_metres_homogeneous; depth_mm = z_normalised * mm_per_unit",
        transform=s.parser.transform.tolist(), scale=s.scale, mm_per_unit=METRES_TO_MM / s.scale,
        metric_frame="metres, colour camera of shot 0 (COLMAP sparse/0)"), indent=2))
    (run / "ply").mkdir(exist_ok=True)
    export_splats(means=splats["means"], scales=splats["scales"], quats=splats["quats"], opacities=splats["opacities"],
                  sh0=splats["sh0"], shN=splats["shN"], format="ply",
                  save_to=str(run / "ply" / f"point_cloud_{data['step']}.ply"))

    maps = run / "maps"
    maps.mkdir(exist_ok=True)
    cams = dict(K=s.K.tolist(), wh=list(s.wh), views=[])
    for i, stem in enumerate(s.stems):
        v = render_2dgs(s, splats, i, device)
        np.savez_compressed(maps / f"{stem}.npz", median_depth_mm=v["depth_mm"].astype(np.float32),
                            alpha=v["alpha"].astype(np.float32), normal_cam=v["normals"].astype(np.float16),
                            expected_depth_mm=v["ed_mm"].astype(np.float32))
        cv2.imwrite(str(maps / f"{stem}_rgb.png"), cv2.cvtColor((v["rgb"] * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
        cams["views"].append(dict(stem=stem, split="test" if s.is_test(i) else "train",
                                  c2w_metres=raw.camtoworlds[i].tolist()))
    (maps / "cameras.json").write_text(json.dumps(cams))
    print(f"[OUT ] {run / 'normalisation.json'}, {run / 'ply'}, {maps} ({len(s.stems)} views)")


COMPARE_REGIONS = ["plant", "pot_no_tags", "tags"]


def cmd_compare(args):
    run, ref = Path(args.run_dir), Path(args.ref_run_dir)
    folder = diag_folder(args)
    a = json.loads((ref / folder / "summary.json").read_text())  # 3DGS
    b = json.loads((run / folder / "summary.json").read_text())  # 2DGS
    out = run / ("compare_tof_clean" if args.tof_clean == "on" else "compare")
    out.mkdir(exist_ok=True)
    tag_a, tag_b = (f"{round((ckpt_step(x['ckpt']) + 1) / 1000)}k" for x in (a, b))
    entries = [(f"3DGS {tag_a}, ED", a, "trained", True), ("3DGS init seed, ED", a, "init", False),
               (f"2DGS {tag_b}, median", b, "trained", True), (f"2DGS {tag_b}, ED", b, "trained_ed", True),
               ("2DGS init seed, median", b, "init", False)]
    rows = []
    for label, src, key, is_trained in entries:
        for region in COMPARE_REGIONS:
            r = src["regions"][key]["test"][region]
            p = src["psnr"][key]["test"]
            rows.append(dict(model=label, region=region, psnr_full=p["full"], psnr_mask=p["mask"], median_abs=r["median_abs"],
                             bias_mean=r["bias_mean"], median_signed=r["median_signed"], p90_abs=r["p90_abs"],
                             pct_3mm=r["pct_3mm"], pct_5mm=r["pct_5mm"], pct_10mm=r["pct_10mm"], n=r["n"],
                             n_gaussians=src["n_gaussians"] if is_trained else src["n_init"],
                             train_s=src["train_stats"]["ellipse_time"] if is_trained else 0.0))
    write_csv(out / "comparison.csv", rows)

    lines = ["| Model | Region | PSNR full / mask (dB) | Median abs | Bias | p90 | <=3 mm % | <=5 mm % | <=10 mm % | Gaussians | Train s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['region']} | {r['psnr_full']:.2f} / {r['psnr_mask']:.2f} | {r['median_abs']:.2f} | "
                     f"{r['bias_mean']:.2f} | {r['p90_abs']:.2f} | {r['pct_3mm']:.1f} | {r['pct_5mm']:.1f} | {r['pct_10mm']:.1f} | "
                     f"{r['n_gaussians']:,} | {r['train_s']:.0f} |")
    lines += ["", "Trained minus init (held-out views, same renderer and depth type):", "",
              "| Model | Region | Median abs | Bias | p90 |", "|---|---|---|---|---|"]
    for label, src in (("3DGS", a), ("2DGS median", b)):
        for region in COMPARE_REGIONS:
            t, i = src["regions"]["trained"]["test"][region], src["regions"]["init"]["test"][region]
            lines.append(f"| {label} | {region} | {t['median_abs'] - i['median_abs']:+.2f} | "
                         f"{t['bias_mean'] - i['bias_mean']:+.2f} | {t['p90_abs'] - i['p90_abs']:+.2f} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    for pa in sorted((ref / folder).glob("panel_*.png")):
        ia, ib = cv2.imread(str(pa)), cv2.imread(str(run / folder / pa.name))
        cv2.imwrite(str(out / f"side_by_side_{pa.stem[len('panel_'):]}.png"),
                    stack_rows([labelled(ia, f"3DGS {tag_a}"), labelled(ib, f"2DGS {tag_b}")]))
    print(f"[OUT ] {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "gate", "metrics", "export", "compare"])
    ap.add_argument("--model", choices=["3dgs", "2dgs"], default="3dgs", help="2dgs: score median depth, use rendered normals")
    ap.add_argument("--tof_cache", type=Path, default=None, help="default: <out>/tof_cache; share one so models get identical targets")
    ap.add_argument("--tof_clean", choices=["off", "on"], default="off", help="on: use the seed's cleaning as the ToF reference")
    ap.add_argument("--clean_dir", type=Path, default=None, help="--tof_clean on: a flora_lib.tof_clean run folder")
    ap.add_argument("--ref_run_dir", type=Path, default=None, help="compare: the 3DGS run, after its metrics stage")
    ap.add_argument("--run_dir", type=Path, required=True, help="folder with cfg.yml or cfg.json, and ckpts/")
    ap.add_argument("--data_dir", type=Path, required=True, help="COLMAP dataset (images/, sparse/0)")
    ap.add_argument("--raw_dir", type=Path, required=True, help="raw capture (depth_<stem>.png, camera_params.json, ...)")
    ap.add_argument("--ckpt", type=Path, default=None, help="default: the last file in <run_dir>/ckpts")
    ap.add_argument("--out", type=Path, default=None, help="default: <run_dir>/diag")
    ap.add_argument("--views", type=int, nargs="+", default=[0, 24, 48], help="gate views (image index)")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "gate": cmd_gate, "metrics": cmd_metrics, "export": cmd_export,
     "compare": cmd_compare}[args.stage](args)


if __name__ == "__main__":
    main()
