#!/usr/bin/env python3
"""
Canopy Height -- Method 1 only (Intel RealSense + SAM3 text prompt)
===================================================================
Reference frame: pose of Tag 16 from pupil_apriltags built-in estimator.
Tag 15 is used ONLY for baseline sanity check (calibration hint).

Segmentation: SAM3 grounded by the text prompt TEXT_PROMPT ("green plants").
All detected instances are unioned into a single plant mask.

Method 1 (Tag frame):
    p_tag = R^-1 @ p_cam  -  R^-1 @ t      (R, t are Tag 16's pose)
    -> report p_tag[1]   (Y in Tag 16's frame)

Usage:
    python canopy_m1_sam3.py --live
    python canopy_m1_sam3.py --folder /path/to/capture_YYYYMMDD_HHMMSS

python .\canopy_detection_using_sam3.py --folder ..\canopy_detection\captures\capture_20260602_121019
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image
from pupil_apriltags import Detector

from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
from sam3.model.sam3_image_processor import Sam3Processor


# =========================================================================
# CONSTANTS  -- edit these here, no CLI flags
# =========================================================================

# --- RealSense stream ---
IMAGE_WIDTH  = 1280
IMAGE_HEIGHT = 720
FPS          = 6

# --- SAM3 ---
DEVICE                    = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_PROMPT               = "plants"
SAM3_CHECKPOINT_VERSION   = "sam3"      # "sam3" or "sam3.1"
SAM3_CONFIDENCE_THRESHOLD = 0.5         # per-instance score threshold
SAM3_LARGEST_ONLY         = False       # True -> keep only the largest instance
SAM3_MIN_AREA_PX          = 0           # drop instances smaller than this

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
# After computing the height-above-tag-center, this is subtracted to express
# the result w.r.t. the reservoir base. The tag is mounted with its outer top
# WHITE edge flush with the reservoir rim, so the offset equals half the outer
# white-edge dimension (= 0.036 / 2 = 0.018 m).
TAG_TO_BASE_OFFSET_M = APRILTAG_OUTER_SIZE_M / 2     # 0.018 m

# --- Baseline sanity check ---
# Baseline measured: Tag 16 TOP-LEFT corner -> Tag 15 TOP-RIGHT corner
# (both back-projected to 3D via depth -- matches the physical tape measure).
KNOWN_BASELINE_M     = 0.65     # set to None to skip
SCALE_WARN_THRESHOLD = 0.05     # warn if |measured/known - 1| > this

# --- Output ---
OUTPUT_DIR       = "./pipeline_output"
CAPTURE_SAVE_DIR = "./captures"
FALLBACK_INTRINSICS = None      # dict(fx,fy,cx,cy[,depth_scale]) or None

# --- Canopy detection ---
CANOPY_MASK_ERODE_PX     = 0       # strip unreliable edge depth before sampling
CANOPY_TOP_FRACTION      = 0.60    # use top 20% of the mask, not just the min-y row
CANOPY_HEIGHT_PERCENTILE = 100.0    # treat P99 height as canopy top (reject flyers)

# --- Visualization ---
MASK_OVERLAY_BGR       = (255, 0, 255)   # mask highlight (BGR); was green (0,200,0)
MASK_OVERLAY_ALPHA     = 0.3
CANOPY_LINE_BGR        = (0, 255, 255)   # canopy line color (BGR)
CANOPY_LINE_HALF_LEN_M = 0.12            # half-length of the line (m), along tag X
CANOPY_LINE_THICKNESS  = 2
CANOPY_TOP_REGION_BGR = (255, 255, 0)    # top-fraction pixel highlight (BGR = cyan)

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
# SAM3 text-prompt segmenter
# =========================================================================

class Sam3PlantSegmenter:
    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        ckpt = (download_ckpt_from_hf(SAM3_CHECKPOINT_VERSION)
                if SAM3_CHECKPOINT_VERSION == "sam3.1" else None)
        self.model = build_sam3_image_model(device=DEVICE, checkpoint_path=ckpt)
        self.processor = Sam3Processor(
            self.model, device=DEVICE,
            confidence_threshold=SAM3_CONFIDENCE_THRESHOLD)
        print(f"[OK] SAM3 ({SAM3_CHECKPOINT_VERSION}) on {DEVICE}  "
              f"prompt='{TEXT_PROMPT}'")

    def segment(self, image_rgb, prompt=TEXT_PROMPT):
        """RGB ndarray (HxWx3 uint8) -> bool mask (HxW), union of instances."""
        image = Image.fromarray(image_rgb)
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(prompt=prompt, state=state)
        h, w = state["original_height"], state["original_width"]
        masks = state.get("masks")
        if masks is None or masks.shape[0] == 0:
            return np.zeros((h, w), dtype=bool)
        masks = masks.squeeze(1).to(torch.bool).cpu().numpy()
        if SAM3_MIN_AREA_PX > 0:
            areas = masks.reshape(masks.shape[0], -1).sum(1)
            masks = masks[areas >= SAM3_MIN_AREA_PX]
        if masks.shape[0] == 0:
            return np.zeros((h, w), dtype=bool)
        if SAM3_LARGEST_ONLY:
            areas = masks.reshape(masks.shape[0], -1).sum(1)
            return masks[int(areas.argmax())]
        return masks.any(axis=0)


# =========================================================================
# AprilTag detector  (with pose estimation)
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
# Pipeline
# =========================================================================

class CanopyPipeline:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print("=" * 60)
        print("Canopy Height -- Method 1 (RealSense + SAM3)")
        print("=" * 60)
        self.segmenter    = Sam3PlantSegmenter()
        self.tag_detector = AprilTagDetector()
        print("[OK] Pipeline ready\n")

    # ---- canopy top from the TOP REGION of the mask, Method 1 only ----
    def find_canopy_top(self, mask, depth_image, R, t,
                        fx, fy, cx, cy, depth_scale, depth_is_meters):
        # Erode away unreliable silhouette depth (parallax fringe lives there).
        if CANOPY_MASK_ERODE_PX > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2*CANOPY_MASK_ERODE_PX+1, 2*CANOPY_MASK_ERODE_PX+1))
            eroded = cv2.erode(mask.astype(np.uint8), k).astype(bool)
            print(f"    Eroded {CANOPY_MASK_ERODE_PX}px: removed "
                  f"{int(mask.sum()-eroded.sum())} edge pixels")
            mask = eroded if eroded.any() else mask

        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None

        # Top region (fraction of mask height) for meaningful percentile stats.
        min_y, max_y = int(ys.min()), int(ys.max())
        y_cutoff = min_y + int((max_y - min_y + 1) * CANOPY_TOP_FRACTION)
        sel = ys <= y_cutoff
        top_ys, top_xs = ys[sel], xs[sel]
        print(f"    Mask Y [{min_y},{max_y}]  top {CANOPY_TOP_FRACTION*100:.0f}% "
              f"-> row {y_cutoff}  ({len(top_ys)} px)")
        if len(top_ys) == 0:
            return None

        # Back-project candidates, compute Method 1 (tag-frame Y).
        R_inv = R.T
        candidates = []
        no_depth = 0
        for xi, yi in zip(top_xs, top_ys):
            p_cam = pixel_to_3d((int(xi), int(yi)), depth_image,
                                fx, fy, cx, cy, depth_scale, depth_is_meters)
            if p_cam is None:
                no_depth += 1
                continue
            p_tag = R_inv @ p_cam - R_inv @ t
            candidates.append({
                "pixel":  (int(xi), int(yi)),
                "p_cam":  p_cam,
                "p_tag":  p_tag,
                "m1_y":   float(p_tag[1]),
                "height": float(-p_tag[1]),     # up-positive, used for selection
            })
        print(f"    3D: {len(candidates)} valid, {no_depth} no depth")
        if not candidates:
            return None

        # Percentile selection: reject flying-pixel outliers.
        heights = np.array([c["height"] for c in candidates])
        p_val   = np.percentile(heights, CANOPY_HEIGHT_PERCENTILE)
        in_top  = [c for c in candidates if c["height"] >= p_val]
        best = (min(in_top, key=lambda c: c["height"]) if in_top
                else max(candidates, key=lambda c: c["height"]))
        print(f"    Heights (cm) — max:{heights.max()*100:.2f}  "
              f"P{CANOPY_HEIGHT_PERCENTILE:g}:{p_val*100:.2f}  "
              f"selected:{best['height']*100:.2f}")
        best["top_pixels"] = list(zip(top_xs.tolist(), top_ys.tolist()))
        best["heights_cm"] = (heights * 100.0).tolist()
        return best

    # ---- main process ----
    def process(self, image_bgr, depth_image, intr,
                depth_scale, depth_is_meters,
                text_prompt=TEXT_PROMPT,
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

        # 3) Segmentation (SAM3 text prompt)
        print(f"\n[2] Segmenting (SAM3 text prompt: '{text_prompt}')...")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = self.segmenter.segment(image_rgb, text_prompt)

        ys, xs = np.where(mask)
        if len(ys) == 0:
            return {"error": "Empty mask (SAM3 found no matching instances)"}
        bbox     = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        mask_pct = mask.sum() / mask.size * 100
        print(f"    Coverage: {mask_pct:.1f}%   BBox: {bbox}")

        # 4) Canopy top + Method 1
        print("\n[3] Computing canopy height (Method 1)...")
        best = self.find_canopy_top(mask, depth_image, R, t,
                                    fx, fy, cx, cy,
                                    depth_scale, depth_is_meters)
        if best is None:
            return {"error": "No valid depth at canopy top edge"}

        m1_y_m = best["m1_y"]
        m1_cm  = m1_y_m * 100   # signed Y in cm (negative when canopy is above tag)

        # Height above tag center (positive when canopy is above tag).
        # Y points DOWN, so negate to flip the sign convention.
        m1_h_tag_cm = -m1_cm

        # Subtract the tag-to-base offset to express height ABOVE RESERVOIR BASE.
        offset_cm = TAG_TO_BASE_OFFSET_M * 100
        m1_h_base_cm = m1_h_tag_cm - offset_cm

        # Report
        print("\n" + "=" * 60)
        print(f"  Canopy pixel:           {best['pixel']}")
        print(f"  p_cam (camera frame):   {best['p_cam']}  m")
        print(f"  p_tag (tag frame):      {best['p_tag']}  m")
        print("  ---")
        print(f"  M1 raw signed Y (tag frame)      : {m1_cm:+.2f} cm   (Y points down)")
        print(f"  M1 canopy height ABOVE TAG center: {m1_h_tag_cm:+.2f} cm")
        print(f"  --- minus offset ({offset_cm:.2f} cm = half outer tag edge) ---")
        print(f"  M1 CANOPY HEIGHT ABOVE RESERVOIR BASE: {m1_h_base_cm:+.2f} cm")
        if scale is not None:
            print(f"  --- scale corrected (factor {scale:.4f}) ---")
            print(f"  M1 height above base (scaled): {m1_h_base_cm*scale:+.2f} cm")
        print("=" * 60)

        # 5) Visualize + save
        vis = self.visualize(image_bgr, mask, tags, best, R, t,
                             intr, baseline_m, scale,
                             m1_h_tag_cm, m1_h_base_cm)

        out_dir = OUTPUT_DIR
        if save:
            if capture_folder:
                cap = Path(capture_folder)
                ts  = cap.name.replace("capture_", "")
                out_dir = str(cap / f"viz_output_{ts}_sam3_m1_plants")
                Path(out_dir).mkdir(exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, "canopy_m1_sam3.png"), vis)
            print(f"\nSaved visualization -> {out_dir}")

        if show:
            disp = vis
            if max(disp.shape[:2]) > 1200:
                s    = 1200 / max(disp.shape[:2])
                disp = cv2.resize(disp, None, fx=s, fy=s)
            cv2.imshow("Canopy -- Method 1 (SAM3)", disp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Height histogram of the top-region candidates
        heights_cm = best.get("heights_cm", [])
        if heights_cm and (save or show):
            import matplotlib
            if not show:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            p99 = float(np.percentile(heights_cm, CANOPY_HEIGHT_PERCENTILE))
            plt.figure(figsize=(7, 4))
            plt.hist(heights_cm, bins=40, color="green", edgecolor="black", alpha=0.8)
            plt.axvline(best["height"]*100, color="red", ls="--",
                        label=f"selected {best['height']*100:.2f} cm")
            plt.axvline(p99, color="orange", ls=":",
                        label=f"P{CANOPY_HEIGHT_PERCENTILE:g} {p99:.2f} cm")
            plt.xlabel("Height above tag center (cm)")
            plt.ylabel("Pixel count")
            plt.title(f"Top {CANOPY_TOP_FRACTION*100:.0f}% region heights (n={len(heights_cm)})")
            plt.legend(); plt.tight_layout()
            if save:
                hist_path = os.path.join(out_dir, "canopy_heights_hist.png")
                plt.savefig(hist_path, dpi=120)
                print(f"Saved histogram -> {hist_path}")
            if show:
                plt.show()
            plt.close()

        return {
            "method1_tag_frame": {
                "y_signed_cm":                  m1_cm,           # raw signed Y in tag frame
                "height_above_tag_cm":          m1_h_tag_cm,     # = -y_signed
                "height_above_base_cm":         m1_h_base_cm,    # = above_tag - offset
                "height_above_base_cm_scaled":  m1_h_base_cm * scale if scale else None,
            },
            "tag_to_base_offset_m":  TAG_TO_BASE_OFFSET_M,
            "text_prompt":           text_prompt,
            "canopy_pixel":          best["pixel"],
            "canopy_p_cam_m":        best["p_cam"].tolist(),
            "canopy_p_tag_m":        best["p_tag"].tolist(),
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
                  baseline_m, scale, m1_h_tag_cm, m1_h_base_cm):
        fx, fy, cx, cy = intr
        out = image_bgr.copy()

        # Mask overlay
        overlay = np.zeros_like(out)
        overlay[mask] = MASK_OVERLAY_BGR
        out = cv2.addWeighted(out, 1, overlay, MASK_OVERLAY_ALPHA, 0)

        # Highlight the top-fraction candidate region
        top = best.get("top_pixels")
        if top:
            pts = np.asarray(top)
            ys2, xs2 = pts[:, 1], pts[:, 0]
            out[ys2, xs2] = (0.5 * out[ys2, xs2]
                             + 0.5 * np.array(CANOPY_TOP_REGION_BGR)).astype(np.uint8)

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

        # Canopy line: full-image line through the canopy point, parallel to tag X
        dx_cam = R[:, 0]
        P = best["p_cam"]
        p0 = project(P, fx, fy, cx, cy)
        p1 = project(P + 0.05 * dx_cam, fx, fy, cx, cy)
        if p0 is not None and p1 is not None:
            d = np.array(p1, float) - np.array(p0, float)
            if np.linalg.norm(d) > 1e-6:
                d /= np.linalg.norm(d)
                L = 2 * max(out.shape[:2])          # longer than the image diagonal
                pa = np.array(p0, float) - L * d
                pb = np.array(p0, float) + L * d
                ok, q1, q2 = cv2.clipLine(
                    (0, 0, out.shape[1], out.shape[0]),
                    tuple(map(int, pa)), tuple(map(int, pb)))
                if ok:
                    cv2.line(out, q1, q2, CANOPY_LINE_BGR, CANOPY_LINE_THICKNESS)

        # Info box
        offset_cm = TAG_TO_BASE_OFFSET_M * 100
        lines = [
            "Canopy height -- Method 1 (SAM3)",
            f"Ref tag: {REF_TAG_ID}   (above-tag = positive)",
        ]
        if baseline_m is not None:
            lines.append(f"Baseline (TL{REF_TAG_ID}->TR{BASELINE_TAG_ID}): {baseline_m*100:.2f} cm")
            if scale is not None:
                lines.append(f"Known: {KNOWN_BASELINE_M*100:.2f} cm  scale: {scale:.4f}")
        lines += [
            f"M1 above tag : {m1_h_tag_cm:+.2f} cm",
            f"-- minus offset ({offset_cm:.2f} cm) --",
            f"M1 above BASE: {m1_h_base_cm:+.2f} cm",
        ]
        if scale is not None:
            lines.append(f"M1 base scaled: {m1_h_base_cm*scale:+.2f} cm")

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
    for k in ("color_path", "depth_path"):
        if not Path(paths[k]).exists():
            raise FileNotFoundError(f"Missing file: {paths[k]}")
    return paths


def save_live_capture(color_bgr, depth_raw, intr, depth_scale,
                      text_prompt=TEXT_PROMPT):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(CAPTURE_SAVE_DIR) / f"capture_{ts}"
    folder.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(folder / f"color_{ts}.png"), color_bgr)
    depth_m = depth_raw.astype(np.float32) * depth_scale
    np.save(folder / f"aligned_depth_m_{ts}.npy", depth_m)

    with open(folder / f"prompts_{ts}.json", "w") as f:
        json.dump({"mode": "text", "text_prompt": text_prompt}, f, indent=2)

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
            "segmenter":        f"sam3:{SAM3_CHECKPOINT_VERSION}",
            "text_prompt":      results.get("text_prompt"),
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
    ap = argparse.ArgumentParser(description="Canopy height -- Method 1 (RealSense + SAM3)")
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
            print("[ERROR] No intrinsics_<ts>.json in folder and "
                  "FALLBACK_INTRINSICS is None.")
            sys.exit(1)

        # Text prompt: prefer saved prompts JSON, else the TEXT_PROMPT constant
        text_prompt = TEXT_PROMPT
        if Path(paths["prompts_path"]).exists():
            with open(paths["prompts_path"]) as f:
                text_prompt = json.load(f).get("text_prompt", TEXT_PROMPT)

        image_bgr   = cv2.imread(paths["color_path"])
        depth_image = np.load(paths["depth_path"])     # saved as float meters
        depth_is_meters = True

        results = pipeline.process(
            image_bgr=image_bgr, depth_image=depth_image,
            intr=intr, depth_scale=depth_scale, depth_is_meters=depth_is_meters,
            text_prompt=text_prompt, show=True, save=True,
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

            folder, ts = save_live_capture(color, depth_raw,
                                           camera.intr, camera.depth_scale,
                                           TEXT_PROMPT)
            depth_m = depth_raw.astype(np.float32) * camera.depth_scale

            results = pipeline.process(
                image_bgr=color, depth_image=depth_m,
                intr=camera.intr, depth_scale=camera.depth_scale,
                depth_is_meters=True, text_prompt=TEXT_PROMPT,
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