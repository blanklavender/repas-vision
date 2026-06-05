import argparse
import os
import sys
import time

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# python .\sam3_plant_extraction.py --image .\391252\color_391252.png --prompt "green plant with white pot"

CHECKPOINT = "./sam3_weights/sam3.pt"

_t0 = time.perf_counter()
def dbg(msg):
    print(f"[DBG +{time.perf_counter() - _t0:6.2f}s] {msg}", flush=True)


def run_sam3(color_path, prompt, confidence):
    dbg(f"run_sam3: opening image {color_path}")
    image = Image.open(color_path).convert("RGB")
    dbg(f"run_sam3: image size (WxH) = {image.size}")
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        dbg(f"run_sam3: CUDA on -> {torch.cuda.get_device_name(0)}")
    else:
        dbg("run_sam3: CUDA not available, running on CPU")

    dbg("run_sam3: building model + processor")
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=CHECKPOINT, load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=confidence)
    dbg("run_sam3: set_image")
    state = processor.set_image(image)
    processor.reset_all_prompts(state)
    dbg(f"run_sam3: set_text_prompt '{prompt}' (conf>={confidence})")
    state = processor.set_text_prompt(state=state, prompt=prompt)
    print(f"[SAM3] {len(state['scores'])} object(s) for '{prompt}'")
    return state


def save_mask_overlay(color_rgb, mask, out_path, alpha=0.5):
    overlay = color_rgb.copy()
    overlay[mask] = (overlay[mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return out_path


def depth_to_pointcloud(depth_img, color_img, mask, fx, fy, cx, cy, depth_scale, min_depth, max_depth):
    h, w = depth_img.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_img.astype(np.float64) / depth_scale
    valid = mask & (z > min_depth) & (z < max_depth)
    dbg(f"  depth_to_pointcloud: mask px={int(mask.sum()):,}  valid depth px={int(valid.sum()):,}")
    z = z[valid]
    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.stack([x, y, z], axis=-1))
    pcd.colors = o3d.utility.Vector3dVector(color_img[valid].astype(np.float64) / 255.0)
    return pcd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--fx", type=float, default=748.982239)
    parser.add_argument("--fy", type=float, default=748.378235)
    parser.add_argument("--cx", type=float, default=633.580566)
    parser.add_argument("--cy", type=float, default=362.705658)
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--min_depth", type=float, default=0.01)
    parser.add_argument("--max_depth", type=float, default=10.0)
    parser.add_argument("--outdir", default="pointclouds")
    args = parser.parse_args()
    dbg(f"args: {vars(args)}")

    os.makedirs(args.outdir, exist_ok=True)

    dbg(f"reading color {args.color}")
    color_bgr = cv2.imread(args.color, cv2.IMREAD_COLOR)
    if color_bgr is None:
        sys.exit(f"ERROR: cannot read color image: {args.color}")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    dbg(f"color shape={color_rgb.shape}")

    dbg(f"reading depth {args.depth}")
    depth_raw = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        sys.exit(f"ERROR: cannot read depth image: {args.depth}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[:, :, 0]
    dbg(f"depth shape={depth_raw.shape}  dtype={depth_raw.dtype}  "
        f"min={depth_raw.min()}  max={depth_raw.max()}")

    h, w = depth_raw.shape[:2]
    if color_rgb.shape[:2] != depth_raw.shape[:2]:
        dbg(f"resizing color {color_rgb.shape[:2]} -> {(h, w)}")
        color_rgb = cv2.resize(color_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    state = run_sam3(args.color, args.prompt, args.confidence)
    n_masks = len(state["scores"])
    if n_masks == 0:
        sys.exit("[SAM3] No objects detected.")

    combined = np.zeros((h, w), dtype=bool)
    for i in range(n_masks):
        mask = state["masks"][i].squeeze(0).cpu().numpy()
        dbg(f"mask {i}: raw shape={mask.shape}")
        if mask.shape != depth_raw.shape:
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            dbg(f"mask {i}: resized -> {mask.shape}")
        else:
            mask = mask.astype(bool)

        score = state["scores"][i]
        if isinstance(score, torch.Tensor):
            score = score.item()

        ys, xs = np.where(mask)
        if len(xs):
            dbg(f"mask {i}: score={score:.2f}  px={len(xs):,}  "
                f"bbox=x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")
        combined |= mask

        overlay_path = os.path.join(args.outdir, f"mask_{i:02d}_score{score:.2f}_overlay.png")
        save_mask_overlay(color_rgb, mask, overlay_path)
        dbg(f"mask {i}: overlay -> {overlay_path}")

        pcd = depth_to_pointcloud(depth_raw, color_rgb, mask, args.fx, args.fy, args.cx, args.cy,
                                  args.depth_scale, args.min_depth, args.max_depth)
        ply_path = os.path.join(args.outdir, f"mask_{i:02d}_score{score:.2f}.ply")
        o3d.io.write_point_cloud(ply_path, pcd)
        print(f"[PLY] mask {i}  score={score:.2f}  points={len(pcd.points):,}  → {ply_path}")

    combined_path = os.path.join(args.outdir, "all_masks_overlay.png")
    save_mask_overlay(color_rgb, combined, combined_path)
    dbg(f"combined overlay ({int(combined.sum()):,} px) -> {combined_path}")
    dbg("done")


if __name__ == "__main__":
    main()