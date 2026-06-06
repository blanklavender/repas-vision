#!/usr/bin/env python3
"""
Canopy Height -- Two-Method Comparison (Intel RealSense)
========================================================
Reference frame: pose of Tag 16 from pupil_apriltags built-in estimator.
Tag 15 is used ONLY for baseline sanity check (calibration hint).

Method 1 (Tag frame):
    p_tag = R^-1 @ p_cam  -  R^-1 @ t      (R, t are Tag 16's pose)
    -> report p_tag[1]   (Y in Tag 16's frame)

Method 2 (Camera-frame triangle):
    OP - OT = TP
    OP = p_cam     (camera -> canopy point P)
    OT = t         (camera -> Tag 16 origin)
    TP = p_cam - t
    -> report TP[1]      (Y in camera frame)

Usage:
    python canopy_two_methods.py --live
    python canopy_two_methods.py --folder /path/to/capture_YYYYMMDD_HHMMSS
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from pupil_apriltags import Detector
from torchvision.transforms import ToTensor


# =========================================================================
# CONSTANTS  -- edit these here, no CLI flags
# =========================================================================

# --- RealSense stream ---
IMAGE_WIDTH  = 1280
IMAGE_HEIGHT = 720
FPS          = 6

# --- EfficientSAM ---
EFFICIENT_SAM_REPO_PATH = "../../EfficientSAM"
MODEL_TYPE              = "vitt"            # "vitt" or "vits"
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"

# --- AprilTag ---
APRILTAG_FAMILY = "tag36h11"

# Physical dimensions of the printed tag:
#   APRILTAG_BLACK_SIZE_M -- side of the BLACK square (what pupil_apriltags
#                            wants as `tag_size`; pose translation scales with this).
#   APRILTAG_OUTER_SIZE_M -- side of the OUTER WHITE edge (the physical
#                            reference for the baseline tape measure).
APRILTAG_BLACK_SIZE_M = 0.028
APRILTAG_OUTER_SIZE_M = 0.036
APRILTAG_SIZE_M       = APRILTAG_BLACK_SIZE_M       # alias used by the detector
# White-border thickness on each side ( = half the difference, 0.004 m here)
APRILTAG_WHITE_BORDER_M = (APRILTAG_OUTER_SIZE_M - APRILTAG_BLACK_SIZE_M) / 2

REF_TAG_ID      = 16        # pose used as the measurement reference frame
BASELINE_TAG_ID = 15        # used only for baseline sanity check
# pupil_apriltags corner order:  [0]=BL  [1]=BR  [2]=TR  [3]=TL
CORNER_BL, CORNER_BR, CORNER_TR, CORNER_TL = 0, 1, 2, 3

# --- Vertical offset from tag center to reservoir base (meters) ---
# After computing each method's height-above-tag-center, this is subtracted
# to express the result w.r.t. the reservoir base. The tag is mounted with
# its outer top WHITE edge flush with the reservoir rim, so the offset equals
# half the outer white-edge dimension (= 0.036 / 2 = 0.018 m).
TAG_TO_BASE_OFFSET_M = APRILTAG_OUTER_SIZE_M / 2     # 0.018 m

# --- Baseline sanity check ---
# Baseline measured: Tag 16 TOP-LEFT corner -> Tag 15 TOP-RIGHT corner
# (both back-projected to 3D via depth -- matches the physical tape measure).
KNOWN_BASELINE_M     = 0.65     # set to None to skip
SCALE_WARN_THRESHOLD = 0.05     # warn if |measured/known - 1| > this

# --- Output ---
OUTPUT_DIR       = "./pipeline_output"
CAPTURE_SAVE_DIR = "./captures"

# --- Canopy detection ---
NUM_TOP_EDGE_SAMPLES = 20

# --- Fallback intrinsics for --folder mode if intrinsics_<ts>.json is missing.
#     (Leave as None to require the JSON to be present.)
FALLBACK_INTRINSICS = None   # e.g. {"fx": 748.9, "fy": 748.3, "cx": 639.9, "cy": 362.0,
                             #       "depth_scale": 0.001}


# =========================================================================
# Helpers
# =========================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
        return super().default(obj)


def pixel_to_3d(pixel, depth_image, fx, fy, cx, cy,
                depth_scale, depth_is_meters):
    """Back-project a pixel to a 3D point in the CAMERA frame (meters)."""
    x, y = int(pixel[0]), int(pixel[1])
    h, w = depth_image.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    depth_m = None
    for half in [2, 5, 10, 20]:
        patch = depth_image[max(0, y-half):min(h, y+half+1),
                            max(0, x-half):min(w, x+half+1)]
        valid = (patch[(patch > 0.1) & (patch < 10.0)] if depth_is_meters
                 else patch[patch > 100])
        if len(valid) > 0:
            depth_m = float(np.median(valid))
            if not depth_is_meters:
                depth_m *= depth_scale
            break

    if not depth_m:
        return None

    return np.array([
        (x - cx) * depth_m / fx,
        (y - cy) * depth_m / fy,
        depth_m,
    ])


def project(p_cam, fx, fy, cx, cy):
    if p_cam[2] <= 0:
        return None
    return (int(fx * p_cam[0] / p_cam[2] + cx),
            int(fy * p_cam[1] / p_cam[2] + cy))


# =========================================================================
# EfficientSAM
# =========================================================================

class EfficientSAMSegmenter:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        repo_path = Path(EFFICIENT_SAM_REPO_PATH).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"EfficientSAM repo not found at {repo_path}. "
                "Clone: git clone https://github.com/yformer/EfficientSAM.git"
            )
        sys.path.insert(0, str(repo_path))
        orig = os.getcwd()
        os.chdir(repo_path)
        try:
            from efficient_sam.build_efficient_sam import (
                build_efficient_sam_vitt, build_efficient_sam_vits,
            )
            model = (build_efficient_sam_vitt() if MODEL_TYPE == "vitt"
                     else build_efficient_sam_vits())
            model = model.to(DEVICE).eval()
            print(f"[OK] EfficientSAM ({MODEL_TYPE}) on {DEVICE}")
            return model
        finally:
            os.chdir(orig)

    def _run(self, image_rgb, points, labels):
        img_t = ToTensor()(image_rgb).to(DEVICE)
        pts = torch.tensor(points).reshape(1, 1, -1, 2).float().to(DEVICE)
        lbl = torch.tensor(labels).reshape(1, 1, -1).int().to(DEVICE)
        with torch.no_grad():
            logits, iou = self.model(img_t[None], pts, lbl)
        sorted_ids = torch.argsort(iou, dim=-1, descending=True)
        logits = torch.take_along_dim(logits, sorted_ids[..., None, None], dim=2)
        return torch.ge(logits[0, 0, 0], 0).cpu().numpy()

    def segment_with_points(self, image_rgb, points, labels):
        return self._run(image_rgb, points, labels)

    def segment_with_box(self, image_rgb, box):
        x1, y1, x2, y2 = box
        return self._run(image_rgb,
                          np.array([[x1, y1], [x2, y2]]),
                          np.array([2, 3]))


# =========================================================================
# AprilTag detector  (now with pose estimation)
# =========================================================================

class AprilTagDetector:
    def __init__(self):
        self.detector = Detector(
            families=APRILTAG_FAMILY,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.8,
            refine_edges=1,
            decode_sharpening=0.25,
        )

    def detect(self, image_bgr, camera_params=None):
        """
        camera_params: (fx, fy, cx, cy)
            If provided, pose estimation is performed and each detection has
            `pose_R` (3x3) and `pose_t` (3x1) attributes.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        if camera_params is not None:
            return self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=camera_params,
                tag_size=APRILTAG_SIZE_M,
            )
        return self.detector.detect(gray)

    @staticmethod
    def get_by_id(tags, tag_id):
        return next((t for t in tags if t.tag_id == tag_id), None)


# =========================================================================
# RealSense camera
# =========================================================================

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.align    = rs.align(rs.stream.color)
        self.profile  = None
        self.intr     = None        # (fx, fy, cx, cy)
        self.depth_scale = None

    def start(self):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT,
                          rs.format.z16,  FPS)
        cfg.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT,
                          rs.format.bgr8, FPS)
        self.profile = self.pipeline.start(cfg)

        ci = self.profile.get_stream(rs.stream.color) \
                          .as_video_stream_profile().get_intrinsics()
        self.intr = (float(ci.fx), float(ci.fy), float(ci.ppx), float(ci.ppy))

        ds = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(ds.get_depth_scale())

        print(f"[OK] RealSense {IMAGE_WIDTH}x{IMAGE_HEIGHT} @ {FPS}fps")
        print(f"     fx={ci.fx:.2f}  fy={ci.fy:.2f}  cx={ci.ppx:.2f}  cy={ci.ppy:.2f}")
        print(f"     depth_scale = {self.depth_scale} m/unit")
        print("     Warming up (auto-exposure)...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        return self

    def grab(self):
        frames  = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        d = aligned.get_depth_frame()
        c = aligned.get_color_frame()
        if not d or not c:
            raise RuntimeError("Failed to capture aligned frames")
        return np.asanyarray(c.get_data()), np.asanyarray(d.get_data())

    def stop(self):
        try:    self.pipeline.stop()
        except Exception: pass


# =========================================================================
# Live preview
# =========================================================================

def preview_and_capture(camera, tag_detector):
    """Live preview. Returns (color, depth_raw) on 'e', or (None, None) on 'q'."""
    print("\nLive preview -- 'e' to capture, 'q' to quit")
    while True:
        color, depth_raw = camera.grab()
        vis = color.copy()
        try:
            tags = tag_detector.detect(color)
            ids  = [t.tag_id for t in tags]
            for t in tags:
                c = ((0, 255, 255) if t.tag_id == REF_TAG_ID
                     else (255, 0, 255) if t.tag_id == BASELINE_TAG_ID
                     else (128, 128, 128))
                cv2.polylines(vis, [t.corners.astype(int)], True, c, 2)
                cv2.putText(vis, f"ID:{t.tag_id}",
                            (int(t.center[0])+8, int(t.center[1])-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        except Exception:
            ids = []

        ok = (REF_TAG_ID in ids) and (BASELINE_TAG_ID in ids)
        status = (f"Tags: {ids}  "
                  f"[{'BOTH OK' if ok else f'NEED {REF_TAG_ID} AND {BASELINE_TAG_ID}'}]")
        cv2.putText(vis, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if ok else (0, 0, 255), 2)
        cv2.putText(vis, "'e' capture   'q' quit",
                    (10, vis.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("RealSense Preview", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('e'):
            cv2.destroyWindow("RealSense Preview")
            return color, depth_raw
        if k == ord('q'):
            cv2.destroyWindow("RealSense Preview")
            return None, None


# =========================================================================
# Interactive prompt selector  (unchanged from your original)
# =========================================================================

class InteractivePromptSelector:
    """
    Left click  -> foreground point (label 1)
    Right click -> background point (label 0)
    'b' toggles box mode (drag rectangle)
    'c' clears all   |   Enter/Space confirms   |   'q' aborts
    Returns: (points, labels, box) -- at most one of points/box is set.
    """

    def __init__(self, image_bgr):
        self.original = image_bgr.copy()
        self.points: List[List[int]] = []
        self.labels: List[int]       = []
        self.box: Optional[Tuple[int, int, int, int]] = None
        self.box_mode    = False
        self._box_start  = None
        self._drag_end   = None
        self._aborted    = False

    def _on_mouse(self, event, x, y, flags, param):
        if self.box_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._box_start = (x, y); self._drag_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self._box_start is not None:
                self._drag_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self._box_start is not None:
                x0, y0 = self._box_start
                self.box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
                self._box_start = None; self._drag_end = None
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append([x, y]); self.labels.append(1)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.points.append([x, y]); self.labels.append(0)

    def _render(self):
        img = self.original.copy()
        for (x, y), lbl in zip(self.points, self.labels):
            c = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(img, (x, y), 6, c, -1)
            cv2.circle(img, (x, y), 7, (255, 255, 255), 1)
        if self.box is not None:
            cv2.rectangle(img, (self.box[0], self.box[1]),
                          (self.box[2], self.box[3]), (255, 255, 0), 2)
        if self.box_mode and self._box_start is not None and self._drag_end is not None:
            cv2.rectangle(img, self._box_start, self._drag_end, (0, 255, 255), 1)
        hud = [
            f"Mode: {'BOX' if self.box_mode else 'POINTS'}   "
            "(b: toggle  c: clear  ENTER: confirm  q: abort)",
            "Left = foreground  |  Right = background",
            f"Points: {len(self.points)}   Box: {'set' if self.box else 'none'}",
        ]
        for i, line in enumerate(hud):
            cv2.putText(img, line, (10, 25 + 22*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(img, line, (10, 25 + 22*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return img

    def run(self):
        win = "Select prompts"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self._on_mouse)
        try:
            while True:
                cv2.imshow(win, self._render())
                k = cv2.waitKey(20) & 0xFF
                if k == ord('b'):
                    self.box_mode = not self.box_mode
                    self._box_start = None; self._drag_end = None
                elif k == ord('c'):
                    self.points.clear(); self.labels.clear(); self.box = None
                elif k in (13, 32):
                    break
                elif k == ord('q'):
                    self._aborted = True
                    break
        finally:
            cv2.destroyWindow(win)

        if self._aborted:
            return None, None, None
        if self.box is not None:
            return None, None, self.box
        if len(self.points) == 0:
            return None, None, None
        return np.array(self.points), np.array(self.labels), None


# =========================================================================
# Pipeline
# =========================================================================

class CanopyPipeline:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print("=" * 60)
        print("Canopy Height -- Two-Method Comparison (RealSense)")
        print("=" * 60)
        self.segmenter    = EfficientSAMSegmenter()
        self.tag_detector = AprilTagDetector()
        print("[OK] Pipeline ready\n")

    # ---- canopy top: compute both methods per candidate, pick best ----
    def find_canopy_top(self, mask, depth_image, R, t,
                         fx, fy, cx, cy, depth_scale, depth_is_meters):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None

        min_y  = int(ys.min())
        top_xs = xs[ys == min_y]
        print(f"    Found {len(top_xs)} top-edge pixels at y={min_y}; "
              f"sampling up to {NUM_TOP_EDGE_SAMPLES}...")
        if len(top_xs) > NUM_TOP_EDGE_SAMPLES:
            idx    = np.linspace(0, len(top_xs)-1, NUM_TOP_EDGE_SAMPLES, dtype=int)
            top_xs = top_xs[idx]

        R_inv = R.T   # rotation matrix => inverse = transpose
        candidates = []
        for x in top_xs:
            p_cam = pixel_to_3d((int(x), min_y), depth_image,
                                 fx, fy, cx, cy, depth_scale, depth_is_meters)
            if p_cam is None:
                continue

            # --- Method 1: p_tag = R^-1 p_cam - R^-1 t ---
            p_tag = R_inv @ p_cam - R_inv @ t
            # (equivalent: p_tag = R.T @ (p_cam - t))

            # --- Method 2: TP = OP - OT = p_cam - t ---
            TP = p_cam - t

            candidates.append({
                "pixel": (int(x), min_y),
                "p_cam": p_cam,
                "p_tag": p_tag,
                "TP":    TP,
                "m1_y":  float(p_tag[1]),   # Y in tag frame
                "m2_y":  float(TP[1]),      # Y in camera frame
            })

        if not candidates:
            return None

        # "Highest" canopy point = smallest p_cam.y (OpenCV camera frame: Y down).
        # Same physical point reported by both methods -> fair comparison.
        return min(candidates, key=lambda c: c["p_cam"][1])

    # ---- main process ----
    def process(self, image_bgr, depth_image, intr,
                depth_scale, depth_is_meters,
                point_prompts=None, point_labels=None, box_prompt=None,
                show=True, save=True, capture_folder=None):

        print("=" * 60 + "\nPROCESSING\n" + "=" * 60)
        fx, fy, cx, cy = intr

        # 1) Tag detection with pose
        print("\n[1] Detecting AprilTags (with pose)...")
        tags = self.tag_detector.detect(image_bgr, camera_params=(fx, fy, cx, cy))
        tag_ref  = AprilTagDetector.get_by_id(tags, REF_TAG_ID)
        tag_base = AprilTagDetector.get_by_id(tags, BASELINE_TAG_ID)
        print(f"    Detected IDs: {[t.tag_id for t in tags]}")
        if tag_ref is None:
            return {"error": f"Reference tag {REF_TAG_ID} not detected"}

        R = np.asarray(tag_ref.pose_R, dtype=np.float64)
        t = np.asarray(tag_ref.pose_t, dtype=np.float64).flatten()
        print(f"    Tag {REF_TAG_ID} pose_t (camera frame, m): {t}")
        print(f"    Tag {REF_TAG_ID} pose_R:")
        print(R)
        print(f"    det(R)={np.linalg.det(R):.6f}  (should be ~1)")

        # 2) Baseline sanity check (Tag 15 used only here).
        # Measured from Tag 16 TOP-LEFT corner to Tag 15 TOP-RIGHT corner,
        # both back-projected to 3D via depth (matches the physical reference).
        baseline_m = None
        scale = None
        baseline_p1_cam = None
        baseline_p2_cam = None
        if tag_base is not None:
            p_tl16 = tag_ref.corners[CORNER_TL]    # Tag 16 top-left  (pixels)
            p_tr15 = tag_base.corners[CORNER_TR]   # Tag 15 top-right (pixels)
            baseline_p1_cam = pixel_to_3d(p_tl16, depth_image,
                                           fx, fy, cx, cy,
                                           depth_scale, depth_is_meters)
            baseline_p2_cam = pixel_to_3d(p_tr15, depth_image,
                                           fx, fy, cx, cy,
                                           depth_scale, depth_is_meters)
            if baseline_p1_cam is not None and baseline_p2_cam is not None:
                baseline_m = float(np.linalg.norm(baseline_p2_cam - baseline_p1_cam))
                print(f"    Tag {REF_TAG_ID}  TL corner pixel: {tuple(p_tl16)}")
                print(f"    Tag {BASELINE_TAG_ID} TR corner pixel: {tuple(p_tr15)}")
                print(f"    Tag {REF_TAG_ID}  TL corner (cam frame, m): {baseline_p1_cam}")
                print(f"    Tag {BASELINE_TAG_ID} TR corner (cam frame, m): {baseline_p2_cam}")
                print(f"    Baseline (Tag{REF_TAG_ID} TL -> Tag{BASELINE_TAG_ID} TR): "
                      f"{baseline_m*100:.2f} cm")
                if KNOWN_BASELINE_M is not None:
                    scale = KNOWN_BASELINE_M / baseline_m
                    print(f"    Known baseline:  {KNOWN_BASELINE_M*100:.2f} cm")
                    print(f"    Scale factor:    {scale:.4f}  "
                          f"(>1 means measurements are reading short)")
                    if abs(scale - 1.0) > SCALE_WARN_THRESHOLD:
                        print(f"    [WARN] scale factor deviates >"
                              f"{SCALE_WARN_THRESHOLD} from 1.0 -- recalibration suggested")
            else:
                print("    [WARN] Could not back-project both baseline corners "
                      "(missing depth at corner pixels); skipping baseline check.")
        else:
            print(f"    [WARN] Baseline tag {BASELINE_TAG_ID} not detected; "
                  "skipping baseline check.")

        # 3) Segmentation
        print("\n[2] Segmenting...")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if box_prompt is not None:
            mask = self.segmenter.segment_with_box(image_rgb, box_prompt)
        elif point_prompts is not None and point_labels is not None:
            mask = self.segmenter.segment_with_points(image_rgb,
                                                       point_prompts, point_labels)
        else:
            return {"error": "No segmentation prompt provided"}

        ys, xs = np.where(mask)
        if len(ys) == 0:
            return {"error": "Empty mask"}
        bbox     = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        mask_pct = mask.sum() / mask.size * 100
        print(f"    Coverage: {mask_pct:.1f}%   BBox: {bbox}")

        # 4) Canopy top + both methods
        print("\n[3] Computing canopy height (both methods)...")
        best = self.find_canopy_top(mask, depth_image, R, t,
                                     fx, fy, cx, cy,
                                     depth_scale, depth_is_meters)
        if best is None:
            return {"error": "No valid depth at canopy top edge"}

        m1_y_m = best["m1_y"]
        m2_y_m = best["m2_y"]
        m1_cm  = m1_y_m * 100   # signed Y in cm  (negative when canopy is above tag)
        m2_cm  = m2_y_m * 100

        # Height above tag center (positive when canopy is above tag).
        # Y points DOWN in both frames, so negate to flip the sign convention.
        m1_h_tag_cm = -m1_cm
        m2_h_tag_cm = -m2_cm

        # Subtract the tag-to-base offset (1.8 cm = AprilTag side length)
        # to express the result as canopy height ABOVE THE RESERVOIR BASE.
        offset_cm = TAG_TO_BASE_OFFSET_M * 100
        m1_h_base_cm = m1_h_tag_cm - offset_cm
        m2_h_base_cm = m2_h_tag_cm - offset_cm

        # Report
        print("\n" + "=" * 60)
        print(f"  Canopy pixel:           {best['pixel']}")
        print(f"  p_cam (camera frame):   {best['p_cam']}  m")
        print(f"  p_tag (tag frame):      {best['p_tag']}  m")
        print(f"  TP = p_cam - t:         {best['TP']}    m")
        print("  ---")
        print(f"  M1 raw signed Y (tag frame)    : {m1_cm:+.2f} cm   (Y points down)")
        print(f"  M2 raw signed Y (cam frame TP) : {m2_cm:+.2f} cm   (Y points down)")
        print("  ---")
        print(f"  M1 canopy height ABOVE TAG center: {m1_h_tag_cm:+.2f} cm")
        print(f"  M2 canopy height ABOVE TAG center: {m2_h_tag_cm:+.2f} cm")
        print(f"  --- minus offset ({offset_cm:.2f} cm = AprilTag side length) ---")
        print(f"  M1 CANOPY HEIGHT ABOVE RESERVOIR BASE: {m1_h_base_cm:+.2f} cm")
        print(f"  M2 CANOPY HEIGHT ABOVE RESERVOIR BASE: {m2_h_base_cm:+.2f} cm")
        if scale is not None:
            print(f"  --- scale corrected (factor {scale:.4f}) ---")
            print(f"  M1 height above base (scaled): {m1_h_base_cm*scale:+.2f} cm")
            print(f"  M2 height above base (scaled): {m2_h_base_cm*scale:+.2f} cm")
        print("=" * 60)

        # 5) Visualize + save
        vis = self.visualize(image_bgr, mask, tags, best, R, t,
                              intr, baseline_m, scale,
                              m1_h_tag_cm, m2_h_tag_cm,
                              m1_h_base_cm, m2_h_base_cm)

        out_dir = OUTPUT_DIR
        if save:
            if capture_folder:
                cap = Path(capture_folder)
                ts  = cap.name.replace("capture_", "")
                out_dir = str(cap / f"viz_output_{ts}")
                Path(out_dir).mkdir(exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, "canopy_two_methods.png"), vis)
            print(f"\nSaved visualization -> {out_dir}")

        if show:
            disp = vis
            if max(disp.shape[:2]) > 1200:
                s    = 1200 / max(disp.shape[:2])
                disp = cv2.resize(disp, None, fx=s, fy=s)
            cv2.imshow("Canopy -- Two Methods", disp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            "method1_tag_frame": {
                "y_signed_cm":                  m1_cm,           # raw signed Y in tag frame
                "height_above_tag_cm":          m1_h_tag_cm,     # = -y_signed
                "height_above_base_cm":         m1_h_base_cm,    # = above_tag - 1.8 cm
                "height_above_base_cm_scaled":  m1_h_base_cm * scale if scale else None,
            },
            "method2_cam_frame_TP": {
                "y_signed_cm":                  m2_cm,
                "height_above_tag_cm":          m2_h_tag_cm,
                "height_above_base_cm":         m2_h_base_cm,
                "height_above_base_cm_scaled":  m2_h_base_cm * scale if scale else None,
            },
            "tag_to_base_offset_m":  TAG_TO_BASE_OFFSET_M,
            "canopy_pixel":          best["pixel"],
            "canopy_p_cam_m":        best["p_cam"].tolist(),
            "canopy_p_tag_m":        best["p_tag"].tolist(),
            "TP_m":                  best["TP"].tolist(),
            "ref_tag_id":            REF_TAG_ID,
            "ref_tag_pose": {
                "R": R.tolist(),
                "t_m": t.tolist(),
            },
            "baseline_m":            baseline_m,
            "baseline_endpoints_cam_m": {
                f"tag{REF_TAG_ID}_TL":     (baseline_p1_cam.tolist()
                                             if baseline_p1_cam is not None else None),
                f"tag{BASELINE_TAG_ID}_TR":(baseline_p2_cam.tolist()
                                             if baseline_p2_cam is not None else None),
            },
            "known_baseline_m":      KNOWN_BASELINE_M,
            "scale_factor":          scale,
            "bounding_box":          bbox,
            "mask_coverage_pct":     mask_pct,
            "output_dir":            out_dir,
        }

    # ---- visualization ----
    def visualize(self, image_bgr, mask, tags, best, R, t, intr,
                  baseline_m, scale,
                  m1_h_tag_cm, m2_h_tag_cm,
                  m1_h_base_cm, m2_h_base_cm):
        fx, fy, cx, cy = intr
        out = image_bgr.copy()

        # Mask overlay
        overlay = np.zeros_like(out)
        overlay[mask] = (0, 200, 0)
        out = cv2.addWeighted(out, 1, overlay, 0.3, 0)

        # Tag outlines
        tag_colors = {REF_TAG_ID: (0, 255, 255), BASELINE_TAG_ID: (255, 0, 255)}
        for tag in tags:
            c = tag_colors.get(tag.tag_id, (128, 128, 128))
            cv2.polylines(out, [tag.corners.astype(int)], True, c, 2)
            cv2.putText(out, f"ID:{tag.tag_id}",
                        (int(tag.center[0])+10, int(tag.center[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

        # Baseline line: Tag 16 TOP-LEFT corner -> Tag 15 TOP-RIGHT corner
        tag_ref  = AprilTagDetector.get_by_id(tags, REF_TAG_ID)
        tag_base = AprilTagDetector.get_by_id(tags, BASELINE_TAG_ID)
        if tag_ref is not None and tag_base is not None:
            p1 = tuple(map(int, tag_ref.corners[CORNER_TL]))
            p2 = tuple(map(int, tag_base.corners[CORNER_TR]))
            cv2.line(out, p1, p2, (0, 0, 255), 2)
            cv2.circle(out, p1, 5, (0, 0, 255), -1)
            cv2.circle(out, p2, 5, (0, 0, 255), -1)

        # Tag 16 axes from pose (X red, Y green, Z blue)
        self._draw_tag_axes(out, R, t, fx, fy, cx, cy, axis_length=0.05)

        # Canopy point
        cv2.circle(out, best["pixel"], 10, (0, 0, 255), -1)
        cv2.circle(out, best["pixel"], 12, (255, 255, 255), 2)

        # Method-2 visualization: vertical drop in camera frame
        #   from canopy to "same X,Z as canopy but tag's Y level"
        p_cam = best["p_cam"]
        drop_to_tag_y_cam = np.array([p_cam[0], t[1], p_cam[2]])
        drop_px = project(drop_to_tag_y_cam, fx, fy, cx, cy)
        if drop_px is not None:
            cv2.line(out, best["pixel"], drop_px, (0, 255, 255), 2)
            cv2.circle(out, drop_px, 5, (0, 255, 255), -1)

        # Info box
        offset_cm = TAG_TO_BASE_OFFSET_M * 100
        lines = [
            "Canopy height -- 2 methods",
            f"Ref tag: {REF_TAG_ID}   (above-tag = positive)",
        ]
        if baseline_m is not None:
            lines.append(f"Baseline (TL{REF_TAG_ID}->TR{BASELINE_TAG_ID}): {baseline_m*100:.2f} cm")
            if scale is not None:
                lines.append(f"Known: {KNOWN_BASELINE_M*100:.2f} cm  scale: {scale:.4f}")
        lines += [
            f"M1 above tag : {m1_h_tag_cm:+.2f} cm",
            f"M2 above tag : {m2_h_tag_cm:+.2f} cm",
            f"-- minus offset ({offset_cm:.2f} cm) --",
            f"M1 above BASE: {m1_h_base_cm:+.2f} cm",
            f"M2 above BASE: {m2_h_base_cm:+.2f} cm",
        ]
        if scale is not None:
            lines += [
                f"M1 base scaled: {m1_h_base_cm*scale:+.2f} cm",
                f"M2 base scaled: {m2_h_base_cm*scale:+.2f} cm",
            ]

        fs, ft, lh = 0.45, 1, 18
        bw, bh = 320, 10 + lh*len(lines) + 5
        cv2.rectangle(out, (5, 5), (5+bw, bh), (0, 0, 0), -1)
        cv2.rectangle(out, (5, 5), (5+bw, bh), (255, 255, 255), 1)
        for i, ln in enumerate(lines):
            cv2.putText(out, ln, (10, 22 + i*lh),
                        cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0, 255, 255) if ln.startswith("M") else (255, 255, 255),
                        ft)
        return out

    def _draw_tag_axes(self, image, R, t, fx, fy, cx, cy, axis_length=0.05):
        """Draw Tag 16's axes (from its pose) projected into the image."""
        o_px = project(t, fx, fy, cx, cy)
        if o_px is None:
            return
        # Columns of R are the tag's basis vectors expressed in the camera frame
        for axis_vec, color, label in [
            (R @ np.array([axis_length, 0, 0]), (0,   0, 255), "X"),
            (R @ np.array([0, axis_length, 0]), (0, 255,   0), "Y"),
            (R @ np.array([0, 0, axis_length]), (255, 0,   0), "Z"),
        ]:
            end_cam = t + axis_vec
            e_px    = project(end_cam, fx, fy, cx, cy)
            if e_px is not None:
                cv2.arrowedLine(image, o_px, e_px, color, 2, tipLength=0.2)
                cv2.putText(image, label, (e_px[0]+5, e_px[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# =========================================================================
# Capture / load helpers  (folder layout matches the original script)
# =========================================================================

def load_prompts(path):
    with open(path) as f:
        d = json.load(f)
    return {
        "mode":          d.get("mode", "points"),
        "point_prompts": np.array(d["point_prompts"]) if d.get("point_prompts") else None,
        "point_labels":  np.array(d["point_labels"])  if d.get("point_labels")  else None,
        "box_prompt":    tuple(d["box_prompt"])       if d.get("box_prompt")    else None,
    }


def parse_capture_folder(folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder}")
    if not folder.name.startswith("capture_"):
        raise ValueError(f"Folder name must start with 'capture_': {folder.name}")
    ts = folder.name.replace("capture_", "")
    paths = {
        "color_path":      str(folder / f"color_{ts}.png"),
        "depth_path":      str(folder / f"aligned_depth_m_{ts}.npy"),
        "prompts_path":    str(folder / f"prompts_{ts}.json"),
        "intrinsics_path": str(folder / f"intrinsics_{ts}.json"),
        "timestamp":       ts,
    }
    for k in ("color_path", "depth_path", "prompts_path"):
        if not Path(paths[k]).exists():
            raise FileNotFoundError(f"Missing file: {paths[k]}")
    return paths


def save_live_capture(color_bgr, depth_raw, intr, depth_scale,
                       point_prompts, point_labels, box_prompt):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(CAPTURE_SAVE_DIR) / f"capture_{ts}"
    folder.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(folder / f"color_{ts}.png"), color_bgr)
    depth_m = depth_raw.astype(np.float32) * depth_scale
    np.save(folder / f"aligned_depth_m_{ts}.npy", depth_m)

    prompts = {
        "mode":          "box" if box_prompt is not None else "points",
        "point_prompts": (point_prompts.tolist()
                          if isinstance(point_prompts, np.ndarray) else None),
        "point_labels":  (point_labels.tolist()
                          if isinstance(point_labels, np.ndarray) else None),
        "box_prompt":    (list(box_prompt) if box_prompt is not None else None),
    }
    with open(folder / f"prompts_{ts}.json", "w") as f:
        json.dump(prompts, f, indent=2)

    fx, fy, cx, cy = intr
    with open(folder / f"intrinsics_{ts}.json", "w") as f:
        json.dump({"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                   "width": IMAGE_WIDTH, "height": IMAGE_HEIGHT,
                   "depth_scale": depth_scale, "source": "realsense"},
                  f, indent=2)

    print(f"\nSaved capture -> {folder}")
    return str(folder), ts


def save_results_json(results, ts, capture_folder):
    if "error" in results:
        return
    out_dir = results["output_dir"]
    payload = {
        "metadata": {
            "timestamp":        ts,
            "capture_folder":   capture_folder,
            "processed_at":     datetime.now().isoformat(),
            "camera":           "intel_realsense",
            "ref_tag_id":       REF_TAG_ID,
            "baseline_tag_id":  BASELINE_TAG_ID,
            "apriltag_size_m":  APRILTAG_SIZE_M,
            "known_baseline_m": KNOWN_BASELINE_M,
        },
        "results": results,
    }
    json_path = os.path.join(out_dir, f"results_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)
    print(f"Results saved -> {json_path}")


# =========================================================================
# Entry point
# =========================================================================

def main():
    ap = argparse.ArgumentParser(description="Canopy height -- 2 methods (RealSense)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--live",   action="store_true",
                   help="Capture from RealSense + process")
    g.add_argument("--folder", type=str,
                   help="Re-process a previously saved capture folder")
    args = ap.parse_args()

    pipeline = CanopyPipeline()

    # ------------------------------------------------------------------
    # FOLDER MODE
    # ------------------------------------------------------------------
    if args.folder:
        paths = parse_capture_folder(args.folder)

        # Intrinsics: prefer saved JSON, else fall back to constants
        if Path(paths["intrinsics_path"]).exists():
            with open(paths["intrinsics_path"]) as f:
                idict = json.load(f)
            intr        = (idict["fx"], idict["fy"], idict["cx"], idict["cy"])
            depth_scale = idict.get("depth_scale", 0.001)
            print(f"[OK] Loaded intrinsics from {paths['intrinsics_path']}")
        elif FALLBACK_INTRINSICS is not None:
            f_ = FALLBACK_INTRINSICS
            intr        = (f_["fx"], f_["fy"], f_["cx"], f_["cy"])
            depth_scale = f_.get("depth_scale", 0.001)
            print("[OK] Using FALLBACK_INTRINSICS from constants")
        else:
            print(f"[ERROR] No intrinsics_<ts>.json in folder and "
                  "FALLBACK_INTRINSICS is None.")
            sys.exit(1)

        prompts     = load_prompts(paths["prompts_path"])
        image_bgr   = cv2.imread(paths["color_path"])
        depth_image = np.load(paths["depth_path"])     # saved as float meters
        depth_is_meters = True

        results = pipeline.process(
            image_bgr=image_bgr, depth_image=depth_image,
            intr=intr, depth_scale=depth_scale, depth_is_meters=depth_is_meters,
            point_prompts=prompts["point_prompts"],
            point_labels=prompts["point_labels"],
            box_prompt=prompts["box_prompt"],
            show=True, save=True,
            capture_folder=args.folder,
        )
        if "error" in results:
            print(f"\n[ERROR] {results['error']}")
        else:
            save_results_json(results, paths["timestamp"], args.folder)
        return

    # ------------------------------------------------------------------
    # LIVE MODE
    # ------------------------------------------------------------------
    camera = RealSenseCamera().start()
    try:
        while True:
            color, depth_raw = preview_and_capture(camera, pipeline.tag_detector)
            if color is None:
                print("Aborted by user.")
                break

            print("\nSelect prompts (left=fg, right=bg, 'b' for box)...")
            pts, lbls, box = InteractivePromptSelector(color).run()
            if pts is None and box is None:
                print("No prompts selected; skipping.")
                continue

            folder, ts = save_live_capture(color, depth_raw,
                                            camera.intr, camera.depth_scale,
                                            pts, lbls, box)
            depth_m = depth_raw.astype(np.float32) * camera.depth_scale

            results = pipeline.process(
                image_bgr=color, depth_image=depth_m,
                intr=camera.intr, depth_scale=camera.depth_scale,
                depth_is_meters=True,
                point_prompts=pts, point_labels=lbls, box_prompt=box,
                show=True, save=True, capture_folder=folder,
            )
            if "error" in results:
                print(f"\n[ERROR] {results['error']}")
            else:
                save_results_json(results, ts, folder)

            print("\n--- Capture again? Press 'q' in preview to exit ---")
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()