#!/usr/bin/env python3
"""
simple_trainer_2dgs_masked.py -- simple_trainer_2dgs with the normal-consistency loss inside the object mask.

Why: on images with a black-filled background the stock loss, averaged over the whole frame, has
error 1 on every background pixel. Opaque black planes across the background lower it at no RGB
cost, because the target there is black too. In a 7k run they grew up to 17 mm from the cameras and
blocked whole views (held-out PSNR 27.3 dB vs 38.4 dB for 3DGS).

Change: during training only, rendered normals and depth normals are zeroed where the training image
is black. The error there stays a constant 1, so it has no gradient, and the loss is unchanged on the
object, including its scale. In this dataset non-black pixels equal the object mask exactly
(checked by `diag_depth.py prepare`). Everything else, including the CLI, is the stock trainer.

Split: if the dataset has heldout_views.json (export_colmap_clean.py), those named views are the held-out
set. Removing views from a dataset would otherwise shift gsplat's index-based split onto different views.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro

import simple_trainer_2dgs as stock


class MaskedRunner(stock.Runner):
    def __init__(self, cfg):
        super().__init__(cfg)
        heldout_file = Path(cfg.data_dir) / "heldout_views.json"
        if heldout_file.is_file():
            names = json.loads(heldout_file.read_text())["names"]
            all_names = self.parser.image_names
            if set(names) - set(all_names):
                raise SystemExit(f"held-out views missing from the dataset: {sorted(set(names) - set(all_names))}")
            self.valset.indices = np.array([i for i, n in enumerate(all_names) if n in names])
            self.trainset.indices = np.array([i for i, n in enumerate(all_names) if n not in names])
            print(f"Split from {heldout_file.name}: {len(self.trainset)} train, held out {[all_names[i] for i in self.valset.indices]}")
        masks = []
        for index in self.trainset.indices:
            image = cv2.imread(self.parser.image_paths[index], cv2.IMREAD_COLOR)
            masks.append(image.sum(2) > 0)
        self.train_masks = np.stack(masks)
        print(f"Normal loss masked to non-black pixels: {self.train_masks.mean() * 100:.2f}% of training pixels")

    def rasterize_splats(self, camtoworlds, Ks, width, height, **kwargs):
        image_ids = kwargs.get("image_ids")  # only the training loop passes these
        out = super().rasterize_splats(camtoworlds, Ks, width, height, **kwargs)
        if image_ids is None:
            return out
        colors, alphas, normals, normals_from_depth, distort, median, info = out
        mask = torch.from_numpy(self.train_masks[image_ids.item()]).to(normals.device)[..., None]
        return colors, alphas, normals * mask, normals_from_depth * mask, distort, median, info


def main(cfg):
    runner = MaskedRunner(cfg)
    runner.train()


if __name__ == "__main__":
    cfg = tyro.cli(stock.Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
