#!/usr/bin/env python3
"""
simple_trainer_2dgs_tofdepth.py -- simple_trainer_2dgs_masked + a dense ToF depth loss.

Why: RGB-only 2DGS sits ~21 mm in front of the cleaned ToF on the plant (median, held-out views,
diag_clean55), 14 mm on the plain pot, but 1.7 mm on the textured tags: photometric geometry drifts
where there is no texture. The stock --depth_loss can't fix it here: it needs COLMAP point tracks
per image, and the export has none (0 points per image), so it would train with no depth targets.

Change: every training step also renders the 2DGS median depth (the surface the TSDF mesh and the
diag_depth.py scores use) and adds
        tof_lambda * mean |median_depth - tof_depth|   (metres)
over pixels where the ToF is a kept pixel of the seed cleaning (flora_lib.tof_clean, the same
reference as diag_depth.py --tof_clean on) AND the rendered alpha is > 0.5 (median defined).
Targets: <tof_cache>/<stem>_depth_mm.npy, undistorted to the colour grid by diag_depth.py prepare.
The loss is back-propagated inside rasterize_splats (the stock loop has no hook for extra terms),
so its gradient adds to the stock loss's in the same optimiser step. Held-out views are never seen.

Settings are copied from an earlier run's cfg.json (--like), so only the depth loss differs:
    PY=~/miniconda3/envs/gsplat/bin/python
    $PY simple_trainer_2dgs_tofdepth.py \
        --like <data>/results/2dgs_7k/cfg.json --result_dir <data>/results/2dgs_7k_tof \
        --tof_cache <tof_cache> --clean_dir <tof clean run> --raw_dir <raw capture> --tof_lambda 1.0
"""

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

import simple_trainer_2dgs as stock
import simple_trainer_2dgs_masked as masked

REPO = Path(os.environ.get("REPAS_VISION", Path.home() / "Projects/repas-vision"))
sys.path.insert(0, str(REPO / "flora_pipeline"))
from flora_lib import tof_clean  # noqa: E402


def color_camera(raw_dir):
    c = json.loads((Path(raw_dir) / "camera_params.json").read_text())
    i = c["color_intrinsic"]
    K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], float)
    return K, np.array(c["color_distortion"]["opencv_order"], float), (i["width"], i["height"])


class ToFDepthRunner(masked.MaskedRunner):
    def __init__(self, cfg, tof):
        super().__init__(cfg)
        self.tof = tof
        T = self.parser.transform
        self.mm_per_unit = 1000.0 / float(np.cbrt(np.linalg.det(T[:3, :3])))
        K, dist, wh = color_camera(tof.raw_dir)
        clean = tof_clean.CleanRun(tof.clean_dir, K, dist, wh)
        targets, valid, n = [], [], []
        for index in self.trainset.indices:
            stem = Path(self.parser.image_names[index]).stem.removeprefix("color_")
            d = np.load(Path(tof.tof_cache) / f"{stem}_depth_mm.npy").astype(np.float32)
            v = np.load(Path(tof.tof_cache) / f"{stem}_valid.npy") & clean.undistorted_valid(stem) & (d > 0)
            targets.append(torch.from_numpy(d))
            valid.append(torch.from_numpy(v))
            n.append(int(v.sum()))
        self.tof_depth = torch.stack(targets).to(self.device)        # [T,H,W] mm
        self.tof_valid = torch.stack(valid).to(self.device)          # [T,H,W] bool
        self.tof_log = []
        print(f"[TOF ] depth loss lambda {tof.tof_lambda}, from step {tof.tof_start_iter}, "
              f"{len(n)} training views, {int(np.median(n))} supervised px/view (median), "
              f"{self.mm_per_unit:.2f} mm per world unit")

    def rasterize_splats(self, camtoworlds, Ks, width, height, **kwargs):
        image_ids = kwargs.get("image_ids")
        out = super().rasterize_splats(camtoworlds, Ks, width, height, **kwargs)
        if image_ids is None or not torch.is_grad_enabled():
            return out
        alphas, median = out[1], out[5]
        i = int(image_ids.item())
        m = self.tof_valid[i] & (alphas[0, ..., 0].detach() > 0.5)
        if m.sum() < 100:
            return out
        err_m = (median[0, ..., 0][m] * self.mm_per_unit - self.tof_depth[i][m]).abs() / 1000.0
        loss = err_m.mean()
        self.tof_log.append(float(loss.detach()) * 1000.0)
        step = len(self.tof_log)
        if step > self.tof.tof_start_iter and self.tof.tof_lambda > 0:
            (loss * self.tof.tof_lambda).backward(retain_graph=True)
        if step % 500 == 0:
            print(f"\n[TOF ] step {step}: mean |median - ToF| on supervised px, last 500 steps "
                  f"{np.mean(self.tof_log[-500:]):.2f} mm")
        return out


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--like", type=Path, required=True, help="cfg.json of the run to copy settings from")
    ap.add_argument("--tof_cache", type=Path, required=True)
    ap.add_argument("--clean_dir", type=Path, required=True, help="flora_lib.tof_clean run (seed cleaning)")
    ap.add_argument("--raw_dir", type=Path, required=True, help="raw capture (camera_params.json)")
    ap.add_argument("--tof_lambda", type=float, default=1.0)
    ap.add_argument("--tof_start_iter", type=int, default=0)
    tof, rest = ap.parse_known_args()

    like = json.loads(tof.like.read_text())
    cfg = stock.Config()
    fields = {f.name for f in dataclasses.fields(cfg)}
    for k, v in like.items():
        if k in fields:
            setattr(cfg, k, v)
    over = argparse.ArgumentParser()
    over.add_argument("--result_dir", required=True)
    over.add_argument("--max_steps", type=int, default=None)
    o = over.parse_args(rest)
    cfg.result_dir = o.result_dir
    if o.max_steps:
        cfg.max_steps, cfg.eval_steps, cfg.save_steps = o.max_steps, [o.max_steps], [o.max_steps]
    Path(cfg.result_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.result_dir) / "tof_depth.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(tof).items()}, indent=2))
    runner = ToFDepthRunner(cfg, tof)
    runner.train()


if __name__ == "__main__":
    main()
