#!/usr/bin/env python3
"""
Canopy Height Detection Pipeline for Femto Bolt Camera
=======================================================
Measures lettuce canopy height using:
- AprilTag-based coordinate frame (directly from depth, bypassing pose estimation)
- EfficientSAM for plant segmentation
- Top 20% of mask points converted to 3D for robust canopy height measurement
  (with mask erosion to remove flying pixels and P99 percentile for outlier rejection)

Coordinate Frame Setup:
- X-axis: from top-left corner of Tag 21 to top-right corner of Tag 16 (horizontal)
- Z-axis: perpendicular to tag face, pointing away from camera (into the system)
- Y-axis: cross product of Z and X (pointing UP)

python canopy_pipeline.py --folder ../../../captures/canopy_test_captures_33.4/capture_20260210_161629 --baseline-cm 65
"""

import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from pupil_apriltags import Detector
import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import argparse
from datetime import datetime
import time


@dataclass
class Config:
    """Pipeline configuration."""
    image_width: int = 1280
    image_height: int = 720

    efficient_sam_repo_path: str = "../../../EfficientSAM"
    model_type: str = "vitt"  # "vitt" or "vits"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    apriltag_family: str = "tag36h11"
    tag_size_meters: float = 0.028

    # Femto Bolt color camera intrinsics
    fx: float = 748.8987426757812
    fy: float = 748.3513793945312
    cx: float = 639.8699951171875
    cy: float = 361.9516906738281

    depth_scale: float = 0.001  # mm → meters for uint16 PNG
    depth_is_meters: bool = False

    left_tag_id: int = 21
    right_tag_id: int = 16

    known_baseline_m: float = None
    scale_factor_warning_threshold: float = 0.05

    output_dir: str = "./pipeline_output"

    canopy_top_fraction: float = 0.20
    canopy_max_samples: int = 2000
    canopy_mask_erode_px: int = 0
    canopy_height_percentile: float = 99.0


# ---------------------------------------------------------------------------
# EfficientSAM segmenter
# ---------------------------------------------------------------------------

class EfficientSAMSegmenter:
    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model()

    def _load_model(self):
        repo_path = Path(self.config.efficient_sam_repo_path).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"EfficientSAM repo not found at {repo_path}. "
                "Clone: git clone https://github.com/yformer/EfficientSAM.git"
            )
        sys.path.insert(0, str(repo_path))
        orig_dir = os.getcwd()
        os.chdir(repo_path)
        try:
            from efficient_sam.build_efficient_sam import (
                build_efficient_sam_vitt, build_efficient_sam_vits
            )
            model = (build_efficient_sam_vitt() if self.config.model_type == "vitt"
                     else build_efficient_sam_vits())
            model = model.to(self.config.device).eval()
            print(f"[OK] EfficientSAM ({self.config.model_type}) on {self.config.device}")
            return model
        finally:
            os.chdir(orig_dir)

    def _run_model(self, image_rgb: np.ndarray, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        img_t = ToTensor()(image_rgb).to(self.config.device)
        pts = torch.tensor(points).reshape(1, 1, -1, 2).float().to(self.config.device)
        lbl = torch.tensor(labels).reshape(1, 1, -1).int().to(self.config.device)
        with torch.no_grad():
            logits, iou = self.model(img_t[None], pts, lbl)
        sorted_ids = torch.argsort(iou, dim=-1, descending=True)
        logits = torch.take_along_dim(logits, sorted_ids[..., None, None], dim=2)
        return torch.ge(logits[0, 0, 0], 0).cpu().numpy()

    def segment_with_points(self, image_rgb: np.ndarray, points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self._run_model(image_rgb, points, labels)

    def segment_with_box(self, image_rgb: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        # Box corners use labels 2 (top-left) and 3 (bottom-right)
        return self._run_model(image_rgb, np.array([[x1, y1], [x2, y2]]), np.array([2, 3]))


# ---------------------------------------------------------------------------
# AprilTag detector
# ---------------------------------------------------------------------------

class AprilTagDetector:
    def __init__(self, config: Config):
        self.config = config
        self.detector = Detector(
            families=config.apriltag_family,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.8,
            refine_edges=1,
            decode_sharpening=0.25,
        )

    def detect(self, image_bgr: np.ndarray) -> List:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return self.detector.detect(gray)

    def get_tag_by_id(self, tags: List, tag_id: int):
        return next((t for t in tags if t.tag_id == tag_id), None)


# ---------------------------------------------------------------------------
# Coordinate frame estimator
# ---------------------------------------------------------------------------

class CoordinateFrameEstimator:
    """
    Establishes world coordinate frame from two AprilTags using depth data
    directly — bypasses solvePnP which is unreliable for small tags at distance.
    """

    # pupil_apriltags corner order: [0]=BL, [1]=BR, [2]=TR, [3]=TL
    TL, TR, BL = 3, 2, 0

    def __init__(self, config: Config):
        self.config = config

    def pixel_to_3d(self, pixel: Tuple[int, int], depth_image: np.ndarray,
                    window_size: int = 5) -> Optional[np.ndarray]:
        """
        Convert pixel + depth to 3D camera-frame point (meters).
        Uses median over expanding windows for robust depth reading.
        """
        x, y = int(pixel[0]), int(pixel[1])
        h, w = depth_image.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        depth_m = None
        for half in [2, 5, 10, 20]:
            patch = depth_image[max(0, y-half):min(h, y+half+1),
                                max(0, x-half):min(w, x+half+1)]
            valid = (patch[(patch > 0.1) & (patch < 10.0)] if self.config.depth_is_meters
                     else patch[patch > 100])
            if len(valid) > 0:
                depth_m = float(np.median(valid))
                if not self.config.depth_is_meters:
                    depth_m *= self.config.depth_scale
                break

        if not depth_m:
            return None

        return np.array([
            (x - self.config.cx) * depth_m / self.config.fx,
            (y - self.config.cy) * depth_m / self.config.fy,
            depth_m
        ])

    def estimate_frame_from_tags(self, tag_left, tag_right,
                                  depth_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Build a right-handed world frame from two co-planar AprilTags:
          X = from tag_left TL → tag_right TR  (horizontal, right)
          Z = normal to tag face (away from camera)
          Y = cross(Z, X)  (up)
        Origin: top-left corner of tag_left.
        """
        cl, cr = tag_left.corners, tag_right.corners

        origin_3d = self.pixel_to_3d(cl[self.TL], depth_image)
        x_end_3d  = self.pixel_to_3d(cr[self.TR], depth_image)
        if origin_3d is None or x_end_3d is None:
            print("[ERROR] No depth at frame anchor corners")
            return None

        # Z-axis from tag plane normal; prefer left tag, fall back to right
        for corners in (cl, cr):
            tl = self.pixel_to_3d(corners[self.TL], depth_image)
            tr = self.pixel_to_3d(corners[self.TR], depth_image)
            bl = self.pixel_to_3d(corners[self.BL], depth_image)
            if tl is not None and tr is not None and bl is not None:
                z_raw = np.cross(tr - tl, bl - tl)
                break
        else:
            print("[ERROR] Cannot estimate Z-axis — insufficient depth data")
            return None

        if z_raw[2] < 0:
            z_raw = -z_raw
        z_axis = z_raw / np.linalg.norm(z_raw)

        x_axis = x_end_3d - origin_3d
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        if y_axis[1] > 0:  # camera Y increases downward; world Y should point up
            y_axis = -y_axis
            z_axis = -z_axis

        # Re-orthogonalise
        z_axis = np.cross(x_axis, y_axis); z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)

        return {
            "origin":    origin_3d,
            "x_axis":    x_axis,
            "y_axis":    y_axis,
            "z_axis":    z_axis,
            "baseline_m": float(np.linalg.norm(x_end_3d - origin_3d)),
            "origin_pixel": cl[self.TL],
            "x_end_pixel":  cr[self.TR],
        }

    def transform_to_world(self, point_camera: np.ndarray, frame: Dict) -> np.ndarray:
        p = point_camera - frame["origin"]
        return np.array([
            np.dot(p, frame["x_axis"]),
            np.dot(p, frame["y_axis"]),
            np.dot(p, frame["z_axis"]),
        ])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class CanopyHeightPipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        os.makedirs(self.config.output_dir, exist_ok=True)

        print("=" * 60)
        print("Canopy Height Detection Pipeline (Femto Bolt)")
        print("=" * 60)

        self.segmenter    = EfficientSAMSegmenter(self.config)
        self.tag_detector = AprilTagDetector(self.config)
        self.frame_est    = CoordinateFrameEstimator(self.config)

        print("[OK] Pipeline ready\n")

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_images(self, color_path: str, depth_path: str):
        image_bgr = cv2.imread(color_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot load color image: {color_path}")

        if depth_path.endswith('.npy'):
            depth_image = np.load(depth_path)
            self.config.depth_is_meters = True
            print("    Depth: NPY (meters)")
        else:
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                raise FileNotFoundError(f"Cannot load depth image: {depth_path}")
            self.config.depth_is_meters = False
            print("    Depth: PNG (uint16 mm)")

        return image_bgr, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), depth_image

    # ------------------------------------------------------------------
    # Canopy top detection
    # ------------------------------------------------------------------

    def find_canopy_top(self, mask: np.ndarray, frame: Dict,
                        depth_image: np.ndarray) -> Tuple[Optional[np.ndarray],
                                                          Optional[Tuple[int, int]],
                                                          Optional[Dict]]:
        """
        Find canopy height from the top fraction of the segmentation mask.

        Robustness measures:
        - Erode mask to strip edge pixels with unreliable depth (flying pixels)
        - Median-windowed depth lookup per point
        - Configurable percentile (default P99) to reject depth outliers
        """
        erode_px   = self.config.canopy_mask_erode_px
        top_frac   = self.config.canopy_top_fraction
        max_pts    = self.config.canopy_max_samples
        height_pct = self.config.canopy_height_percentile

        # Erode
        if erode_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1))
            eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
            print(f"    Eroded {erode_px}px: removed {mask.sum()-eroded.sum()} edge pixels")
            mask = eroded if eroded.any() else mask
        else:
            print("    [OBSERVE] No mask erosion applied")

        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None, None, None

        # Top region
        min_y, max_y = ys.min(), ys.max()
        y_cutoff = min_y + int((max_y - min_y + 1) * top_frac)
        top_filt = ys <= y_cutoff
        top_ys, top_xs = ys[top_filt], xs[top_filt]
        print(f"    Mask Y [{min_y},{max_y}], top {top_frac*100:.0f}% cutoff row {y_cutoff} "
              f"({len(top_ys)} pixels)")

        if len(top_ys) == 0:
            return None, None, None
        
        print(f"    Found {len(top_ys)} top-edge pixels at 20% cutoff; could sample up to {max_pts} points...")
        print(f"    [OBSERVE HERE] Top-edge pixels: {top_ys}")

        # Subsample
        # if len(top_ys) > max_pts:
        #     idx = np.random.default_rng(42).choice(len(top_ys), size=max_pts, replace=False)
        #     top_xs, top_ys = top_xs[idx], top_ys[idx]

        # 3D conversion
        heights, world_pts, cam_pts, pixels_list = [], [], [], []
        no_depth = 0
        t0 = time.perf_counter()
        for xi, yi in zip(top_xs, top_ys):
            pt_cam = self.frame_est.pixel_to_3d((int(xi), int(yi)), depth_image)
            if pt_cam is not None:
                pt_w = self.frame_est.transform_to_world(pt_cam, frame)
                heights.append(pt_w[1])
                world_pts.append(pt_w)
                cam_pts.append(pt_cam)
                pixels_list.append((int(xi), int(yi)))
            else:
                no_depth += 1
        elapsed = time.perf_counter() - t0
        print(f"Depth estimation for {len(top_ys)} points: {elapsed:.3f}s ({elapsed/len(top_ys)*1e3:.4f} ms/point)")

        print(f"    3D: {len(heights)} valid, {no_depth} no depth")
        if not heights:
            return None, None, None

        heights = np.array(heights)
        p_val = np.percentile(heights, height_pct)
        cands = np.where(heights >= p_val)[0]
        best  = cands[np.argmin(heights[cands])] if len(cands) else np.argmax(heights)

        stats = {
            "total_mask_pixels":   int(mask.sum()),
            "top_region_pixels":   len(top_ys),
            "valid_3d_points":     len(heights),
            "erode_px":            erode_px,
            "y_cutoff_row":        int(y_cutoff),
            "height_percentile_used": height_pct,
            "height_max_cm":       float(heights.max() * 100),
            "height_p99_cm":       float(np.percentile(heights, 99) * 100),
            "height_p95_cm":       float(np.percentile(heights, 95) * 100),
            "height_mean_cm":      float(heights.mean() * 100),
            "height_median_cm":    float(np.median(heights) * 100),
            "height_std_cm":       float(heights.std() * 100),
            "height_selected_cm":  float(world_pts[best][1] * 100),
        }
        print(f"    Heights — max:{stats['height_max_cm']:.2f}  "
              f"P99:{stats['height_p99_cm']:.2f}  "
              f"selected:{stats['height_selected_cm']:.2f} cm")

        return world_pts[best], pixels_list[best], stats

    # ------------------------------------------------------------------
    # Main process method
    # ------------------------------------------------------------------

    def process(self, color_path: str, depth_path: str,
                point_prompts: np.ndarray = None, point_labels: np.ndarray = None,
                box_prompt: Tuple[int, int, int, int] = None,
                show: bool = True, save: bool = True,
                capture_folder: str = None) -> Dict[str, Any]:

        print("=" * 60 + "\nPROCESSING\n" + "=" * 60)

        # 1. Load
        print("\n[1] Loading images...")
        image_bgr, image_rgb, depth_image = self.load_images(color_path, depth_path)

        # 2. Tags
        print("\n[2] Detecting AprilTags...")
        tags      = self.tag_detector.detect(image_bgr)
        tag_left  = self.tag_detector.get_tag_by_id(tags, self.config.left_tag_id)
        tag_right = self.tag_detector.get_tag_by_id(tags, self.config.right_tag_id)
        print(f"    Found: {[t.tag_id for t in tags]}")

        if tag_left is None:
            return {"error": f"Tag {self.config.left_tag_id} not detected"}
        if tag_right is None:
            return {"error": f"Tag {self.config.right_tag_id} not detected"}

        # 3. Coordinate frame
        print("\n[3] Establishing coordinate frame...")
        frame = self.frame_est.estimate_frame_from_tags(tag_left, tag_right, depth_image)
        if frame is None:
            return {"error": "Failed to establish coordinate frame"}
        print(f"    Baseline: {frame['baseline_m']*100:.2f} cm")

        # 4. Segmentation
        print("\n[4] Segmenting lettuce...")
        if box_prompt is not None:
            mask = self.segmenter.segment_with_box(image_rgb, box_prompt)
        elif point_prompts is not None and point_labels is not None:
            mask = self.segmenter.segment_with_points(image_rgb, point_prompts, point_labels)
        else:
            return {"error": "No segmentation prompt provided"}

        mask_pct = mask.sum() / mask.size * 100
        ys, xs   = np.where(mask)
        if len(ys) == 0:
            return {"error": "Empty segmentation mask"}
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        print(f"    Coverage: {mask_pct:.1f}%  BBox: {bbox}")

        # 5. Canopy height
        print(f"\n[5] Canopy height (top {self.config.canopy_top_fraction*100:.0f}%)...")
        canopy_world, canopy_pixel, stats = self.find_canopy_top(mask, frame, depth_image)
        if canopy_world is None:
            return {"error": "No depth data in canopy top region"}

        canopy_cam = self.frame_est.pixel_to_3d(canopy_pixel, depth_image)
        height_raw_cm = canopy_world[1] * 100

        # 6. Scale correction
        measured_bl = frame['baseline_m']
        if self.config.known_baseline_m is not None:
            scale = self.config.known_baseline_m / measured_bl
            height_corr_cm = height_raw_cm * scale
            if abs(scale - 1.0) > self.config.scale_factor_warning_threshold:
                print(f"    [WARNING] Scale factor {scale:.4f} deviates > "
                      f"{self.config.scale_factor_warning_threshold} from 1.0")
        else:
            scale = None
            height_corr_cm = None

        print(f"\n{'='*60}")
        print(f"    MEASURED BASELINE: {measured_bl*100:.2f} cm")
        if scale is not None:
            print(f"    KNOWN BASELINE:    {self.config.known_baseline_m*100:.2f} cm")
            print(f"    SCALE FACTOR:      {scale:.4f}")
            print(f"    RAW HEIGHT:        {height_raw_cm:.2f} cm")
            print(f"    CORRECTED HEIGHT:  {height_corr_cm:.2f} cm")
        else:
            print(f"    HEIGHT (uncalibrated): {height_raw_cm:.2f} cm")
        print("=" * 60)

        # 7. Visualize / save
        vis = self.visualize(image_bgr, mask, tags, frame, canopy_pixel, canopy_cam,
                             bbox, height_raw_cm, height_corr_cm, scale)

        out_dir = self.config.output_dir
        if save:
            if capture_folder:
                cap = Path(capture_folder)
                ts  = cap.name.replace("capture_", "")
                out_dir = str(cap / f"viz_output_v2_{ts}")
                Path(out_dir).mkdir(exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, "canopy_height_result.png"), vis)
            print(f"\nSaved visualization → {out_dir}")

        if show:
            disp = vis
            if max(disp.shape[:2]) > 1200:
                s = 1200 / max(disp.shape[:2])
                disp = cv2.resize(disp, None, fx=s, fy=s)
            cv2.imshow("Canopy Height Result", disp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            "height_raw_cm":       height_raw_cm,
            "height_corrected_cm": height_corr_cm,
            "scale_factor":        scale,
            "measured_baseline_m": measured_bl,
            "canopy_pixel":        canopy_pixel,
            "canopy_camera":       canopy_cam.tolist(),
            "canopy_world":        canopy_world.tolist(),
            "bounding_box":        bbox,
            "mask_coverage_pct":   mask_pct,
            "frame":               {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                    for k, v in frame.items()},
            "canopy_stats":        stats,
            "output_dir":          out_dir,
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, image_bgr, mask, tags, frame, canopy_pixel, canopy_cam,
                  bbox, height_raw_cm, height_corr_cm, scale) -> np.ndarray:
        out = image_bgr.copy()

        # Mask overlay
        overlay = np.zeros_like(out)
        overlay[mask] = (0, 200, 0)
        out = cv2.addWeighted(out, 1, overlay, 0.3, 0)

        # Top-region highlight + cutoff line
        ys, xs = np.where(mask)
        if len(ys):
            min_y, max_y = ys.min(), ys.max()
            y_cut = min_y + int((max_y - min_y + 1) * self.config.canopy_top_fraction)
            top_ov = np.zeros_like(out)
            top_mask = mask.copy(); top_mask[y_cut:, :] = False
            top_ov[top_mask] = (0, 100, 255)
            out = cv2.addWeighted(out, 1, top_ov, 0.25, 0)
            cv2.line(out, (bbox[0], y_cut), (bbox[2], y_cut), (0, 165, 255), 1)

        # Tags
        tag_colors = {self.config.left_tag_id: (0, 255, 255),
                      self.config.right_tag_id: (255, 0, 255)}
        for tag in tags:
            c = tag_colors.get(tag.tag_id, (128, 128, 128))
            cv2.polylines(out, [tag.corners.astype(int)], True, c, 2)
            cv2.putText(out, f"ID:{tag.tag_id}",
                        (int(tag.center[0])+10, int(tag.center[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        # Coordinate axes + baseline line
        self._draw_axes(out, frame)
        cv2.line(out, tuple(map(int, frame["origin_pixel"])),
                 tuple(map(int, frame["x_end_pixel"])), (0, 0, 255), 2)

        # Bounding box
        cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)

        # Canopy point
        cv2.circle(out, canopy_pixel, 10, (0, 0, 255), -1)
        cv2.circle(out, canopy_pixel, 12, (255, 255, 255), 2)

        # Height line
        cw = self.frame_est.transform_to_world(canopy_cam, frame)
        base_cam = (frame["origin"]
                    + cw[0]*frame["x_axis"]
                    + 0   *frame["y_axis"]
                    + cw[2]*frame["z_axis"])
        base_px = self._project(base_cam)
        if base_px:
            cv2.line(out, canopy_pixel, base_px, (0, 255, 255), 3)
            cv2.circle(out, base_px, 6, (0, 255, 255), -1)

        # Info box
        lines = [f"Top-{self.config.canopy_top_fraction*100:.0f}% Canopy Height",
                 f"Baseline: {frame['baseline_m']*100:.2f} cm"]
        if scale is not None:
            lines += [f"Known BL: {self.config.known_baseline_m*100:.2f} cm",
                      f"Scale: {scale:.4f}",
                      f"Raw: {height_raw_cm:.2f} cm",
                      f"Corrected: {height_corr_cm:.2f} cm"]
        else:
            lines.append(f"Height: {height_raw_cm:.2f} cm")

        fs, ft, lh = 0.4, 1, 16
        box_h = 10 + lh * len(lines) + 5
        cv2.rectangle(out, (5, 5), (260, box_h), (0, 0, 0), -1)
        cv2.rectangle(out, (5, 5), (260, box_h), (255, 255, 255), 1)
        for i, ln in enumerate(lines):
            clr = ((255, 255, 255) if i == 0
                   else (0, 255, 255) if ln.startswith("Corrected") or ln.startswith("Height")
                   else (200, 200, 200))
            cv2.putText(out, ln, (10, 20 + i*lh), cv2.FONT_HERSHEY_SIMPLEX, fs, clr, ft)

        return out

    def _draw_axes(self, image, frame, length=0.05):
        origin_px = self._project(frame["origin"])
        if origin_px is None:
            return
        for axis, color, label in [(frame["x_axis"], (0,0,255), "X"),
                                    (frame["y_axis"], (0,255,0), "Y"),
                                    (frame["z_axis"], (255,0,0), "Z")]:
            end_px = self._project(frame["origin"] + length * axis)
            if end_px:
                cv2.arrowedLine(image, origin_px, end_px, color, 2, tipLength=0.2)
                cv2.putText(image, label, (end_px[0]+5, end_px[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _project(self, p) -> Optional[Tuple[int, int]]:
        if p[2] <= 0:
            return None
        return (int(self.config.fx * p[0] / p[2] + self.config.cx),
                int(self.config.fy * p[1] / p[2] + self.config.cy))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompts(path: str) -> Dict[str, Any]:
    with open(path) as f:
        d = json.load(f)
    return {
        "mode":          d.get("mode", "points"),
        "point_prompts": np.array(d["point_prompts"]) if d.get("point_prompts") else None,
        "point_labels":  np.array(d["point_labels"])  if d.get("point_labels")  else None,
        "box_prompt":    tuple(d["box_prompt"])        if d.get("box_prompt")    else None,
    }


def parse_capture_folder(folder_path: str) -> Dict[str, str]:
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    if not folder.name.startswith("capture_"):
        raise ValueError(f"Folder must start with 'capture_': {folder.name}")
    ts = folder.name.replace("capture_", "")
    paths = {
        "color_path":   str(folder / f"color_{ts}.png"),
        "depth_path":   str(folder / f"aligned_depth_m_{ts}.npy"),
        "prompts_path": str(folder / f"prompts_{ts}.json"),
        "timestamp":    ts,
    }
    for key in ("color_path", "depth_path", "prompts_path"):
        if not Path(paths[key]).exists():
            raise FileNotFoundError(f"Missing file: {paths[key]}")
    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Canopy Height Detection Pipeline")
    parser.add_argument("--folder", required=True, help="Capture folder path")
    parser.add_argument("--efficient-sam", default="../../../EfficientSAM")
    parser.add_argument("--output", default="./pipeline_output")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--fx", type=float, default=748.8987426757812)
    parser.add_argument("--fy", type=float, default=748.3513793945312)
    parser.add_argument("--cx", type=float, default=639.8699951171875)
    parser.add_argument("--cy", type=float, default=361.9516906738281)
    parser.add_argument("--baseline",    type=float, default=None, help="Known baseline (m)")
    parser.add_argument("--baseline-cm", type=float, default=None, help="Known baseline (cm)")
    parser.add_argument("--top-fraction",     type=float, default=0.20)
    parser.add_argument("--max-samples",      type=int,   default=2000)
    parser.add_argument("--erode-px",         type=int,   default=2)
    parser.add_argument("--height-percentile",type=float, default=99.0)
    args = parser.parse_args()

    paths   = parse_capture_folder(args.folder)
    prompts = load_prompts(paths["prompts_path"])

    known_bl = (args.baseline if args.baseline is not None
                else args.baseline_cm / 100.0 if args.baseline_cm is not None
                else None)

    config = Config(
        efficient_sam_repo_path=args.efficient_sam,
        output_dir=args.output,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
        known_baseline_m=known_bl,
        canopy_top_fraction=args.top_fraction,
        canopy_max_samples=args.max_samples,
        canopy_mask_erode_px=args.erode_px,
        canopy_height_percentile=args.height_percentile,
    )

    pipeline = CanopyHeightPipeline(config)
    results  = pipeline.process(
        color_path=paths["color_path"],
        depth_path=paths["depth_path"],
        point_prompts=prompts["point_prompts"],
        point_labels=prompts["point_labels"],
        box_prompt=prompts["box_prompt"],
        show=not args.no_show,
        save=not args.no_save,
        capture_folder=args.folder,
    )

    if "error" in results:
        print(f"\n[ERROR] {results['error']}")
        return

    # Save JSON results
    ts       = paths["timestamp"]
    out_dir  = results["output_dir"]
    result_json = {
        "metadata": {
            "timestamp": ts,
            "capture_folder": args.folder,
            "processed_at": datetime.now().isoformat(),
            "baseline_calibration_cm": config.known_baseline_m * 100 if config.known_baseline_m else None,
            "canopy_top_fraction": config.canopy_top_fraction,
            "canopy_max_samples": config.canopy_max_samples,
            "canopy_mask_erode_px": config.canopy_mask_erode_px,
            "canopy_height_percentile": config.canopy_height_percentile,
        },
        "summary": {
            "final_height_cm": float(results["height_corrected_cm"]
                                     if results["height_corrected_cm"] is not None
                                     else results["height_raw_cm"]),
            "calibrated": results["scale_factor"] is not None,
            "scale_factor": results["scale_factor"],
            "measured_baseline_cm": results["measured_baseline_m"] * 100,
            "mask_coverage_percent": results["mask_coverage_pct"],
        },
        "measurements": {
            "height_raw_cm":       results["height_raw_cm"],
            "height_corrected_cm": results["height_corrected_cm"],
            "canopy_stats":        results["canopy_stats"],
            "bounding_box":        results["bounding_box"],
        },
        "coordinates": {
            "canopy_pixel":  results["canopy_pixel"],
            "canopy_camera": results["canopy_camera"],
            "canopy_world":  results["canopy_world"],
        },
        "coordinate_frame": results["frame"],
    }

    json_path = os.path.join(out_dir, f"results_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"\nResults saved → {json_path}")


if __name__ == "__main__":
    main()