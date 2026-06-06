#!/usr/bin/env python3
"""
Canopy Height -- Method 1, ROS handoff (RealSense + SAM3 + EfficientSAM)
=======================================================================
SAM3 (text prompt) detects plant instances -> interior seed points ->
EfficientSAM refines each -> union mask. Method 1 (Tag 16 frame) gives the
canopy height; the scale-corrected above-base value is written to OUTPUT_TXT
for the ROS side to read. Out-of-bounds values blank the file.

Usage:
    python canopy_depth_based_for_ros_single_run.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image
from pupil_apriltags import Detector

from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
from sam3.model.sam3_image_processor import Sam3Processor

from torchvision.transforms import ToTensor


# =========================================================================
# CONSTANTS
# =========================================================================

# --- RealSense stream ---
IMAGE_WIDTH  = 1280
IMAGE_HEIGHT = 720
FPS          = 6

# --- SAM3 ---
DEVICE                    = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_PROMPT               = "plants"
SAM3_CHECKPOINT_VERSION   = "sam3"
SAM3_CONFIDENCE_THRESHOLD = 0.5
SAM3_LARGEST_ONLY         = False
SAM3_MIN_AREA_PX          = 0

# --- EfficientSAM refinement ---
EFFICIENT_SAM_REPO_PATH  = "../../EfficientSAM"
EFFICIENT_SAM_MODEL_TYPE = "vitt"
ESAM_POINTS_PER_INSTANCE = 2

# --- AprilTag ---
APRILTAG_FAMILY       = "tag36h11"
APRILTAG_BLACK_SIZE_M = 0.028
APRILTAG_OUTER_SIZE_M = 0.036
APRILTAG_SIZE_M       = APRILTAG_BLACK_SIZE_M
REF_TAG_ID      = 16
BASELINE_TAG_ID = 15
CORNER_BL, CORNER_BR, CORNER_TR, CORNER_TL = 0, 1, 2, 3

TAG_TO_BASE_OFFSET_M = APRILTAG_OUTER_SIZE_M / 2     # 0.018 m

# --- Baseline / scale ---
KNOWN_BASELINE_M     = 0.65
SCALE_WARN_THRESHOLD = 0.05

# --- Canopy detection ---
CANOPY_MASK_ERODE_PX     = 1
CANOPY_TOP_FRACTION      = 0.60
CANOPY_HEIGHT_PERCENTILE = 99.0

# --- ROS output ---
OUTPUT_TXT    = "./canopy_heights.txt"
MIN_HEIGHT_CM = 2.0
MAX_HEIGHT_CM = 30.0

# --- Visualization ---
SAVE_VIZ              = True
SHOW_VIZ              = False
VIZ_PATH             = "./canopy_viz.png"
MASK_OVERLAY_BGR     = (0, 165, 255)     # EfficientSAM mask = ORANGE
MASK_OVERLAY_ALPHA   = 0.35
CANOPY_TOP_REGION_BGR = (255, 255, 0)    # top-fraction region = cyan
CANOPY_LINE_BGR      = (0, 255, 255)     # canopy line = yellow
CANOPY_LINE_THICKNESS = 2
SEED_POINT_BGR       = (255, 255, 255)   # seed points = white


# =========================================================================
# Helpers
# =========================================================================

def pixel_to_3d(pixel, depth_image, fx, fy, cx, cy, depth_scale, depth_is_meters):
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
    return np.array([(x - cx) * depth_m / fx,
                     (y - cy) * depth_m / fy,
                     depth_m])


def project(p_cam, fx, fy, cx, cy):
    if p_cam[2] <= 0:
        return None
    return (int(fx * p_cam[0] / p_cam[2] + cx),
            int(fy * p_cam[1] / p_cam[2] + cy))


def mask_interior_point(mask):
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(dt.argmax()), dt.shape)
    return (int(x), int(y))


def sample_fg_points(mask, k):
    if not mask.any():
        return []
    pts = [mask_interior_point(mask)]
    if k > 1:
        ys, xs = np.where(mask)
        idx = np.linspace(0, len(xs) - 1, k - 1, dtype=int)
        pts += [(int(xs[i]), int(ys[i])) for i in idx]
    return pts


# =========================================================================
# SAM3 detector
# =========================================================================

class Sam3PlantSegmenter:
    def __init__(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        ckpt = (download_ckpt_from_hf(SAM3_CHECKPOINT_VERSION)
                if SAM3_CHECKPOINT_VERSION == "sam3.1" else None)
        self.model = build_sam3_image_model(device=DEVICE, checkpoint_path=ckpt)
        self.processor = Sam3Processor(self.model, device=DEVICE,
                                       confidence_threshold=SAM3_CONFIDENCE_THRESHOLD)
        print(f"[OK] SAM3 ({SAM3_CHECKPOINT_VERSION}) on {DEVICE}  prompt='{TEXT_PROMPT}'")

    def detect_instance_masks(self, image_rgb, prompt=TEXT_PROMPT):
        image = Image.fromarray(image_rgb)
        with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            state = self.processor.set_text_prompt(prompt=prompt, state=state)
        masks = state.get("masks")
        if masks is None or masks.shape[0] == 0:
            return []
        masks = masks.squeeze(1).to(torch.bool).cpu().numpy()
        areas = masks.reshape(masks.shape[0], -1).sum(1)
        keep = (areas >= SAM3_MIN_AREA_PX) if SAM3_MIN_AREA_PX > 0 \
               else np.ones(len(areas), bool)
        masks, areas = masks[keep], areas[keep]
        if len(masks) == 0:
            return []
        if SAM3_LARGEST_ONLY:
            masks = masks[[int(areas.argmax())]]
        return list(masks)


# =========================================================================
# EfficientSAM refiner
# =========================================================================

class EfficientSAMSegmenter:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        repo_path = Path(EFFICIENT_SAM_REPO_PATH).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(
                f"EfficientSAM repo not found at {repo_path}. "
                "Clone: git clone https://github.com/yformer/EfficientSAM.git")
        sys.path.insert(0, str(repo_path))
        orig = os.getcwd()
        os.chdir(repo_path)
        try:
            from efficient_sam.build_efficient_sam import (
                build_efficient_sam_vitt, build_efficient_sam_vits)
            model = (build_efficient_sam_vitt() if EFFICIENT_SAM_MODEL_TYPE == "vitt"
                     else build_efficient_sam_vits())
            model = model.to(DEVICE).eval()
            print(f"[OK] EfficientSAM ({EFFICIENT_SAM_MODEL_TYPE}) on {DEVICE}")
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


# =========================================================================
# SAM3 -> EfficientSAM hybrid
# =========================================================================

class Sam3RefinedSegmenter:
    def __init__(self):
        self.sam3 = Sam3PlantSegmenter()
        self.esam = EfficientSAMSegmenter()
        self.last_points = []
        self.last_sam3_mask = None

    def segment(self, image_rgb, prompt=TEXT_PROMPT):
        h, w = image_rgb.shape[:2]
        inst_masks = self.sam3.detect_instance_masks(image_rgb, prompt)
        self.last_points = []
        self.last_sam3_mask = np.zeros((h, w), dtype=bool)
        if not inst_masks:
            return np.zeros((h, w), dtype=bool)
        union = np.zeros((h, w), dtype=bool)
        for m in inst_masks:
            self.last_sam3_mask |= m
            pts = sample_fg_points(m, ESAM_POINTS_PER_INSTANCE)
            if not pts:
                continue
            self.last_points += pts
            refined = self.esam.segment_with_points(
                image_rgb, np.array(pts), np.array([1] * len(pts)))
            union |= refined.astype(bool)
        print(f"    SAM3 instances: {len(inst_masks)}  pts: {len(self.last_points)}"
              f"  -> EfficientSAM union {union.sum()/union.size*100:.1f}%")
        return union


# =========================================================================
# AprilTag detector
# =========================================================================

class AprilTagDetector:
    def __init__(self):
        self.detector = Detector(families=APRILTAG_FAMILY, nthreads=4,
                                 quad_decimate=1.0, quad_sigma=0.8,
                                 refine_edges=1, decode_sharpening=0.25)

    def detect(self, image_bgr, camera_params=None):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        if camera_params is not None:
            return self.detector.detect(gray, estimate_tag_pose=True,
                                        camera_params=camera_params,
                                        tag_size=APRILTAG_SIZE_M)
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
        self.intr     = None
        self.depth_scale = None

    def start(self):
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, FPS)
        self.profile = self.pipeline.start(cfg)
        ci = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intr = (float(ci.fx), float(ci.fy), float(ci.ppx), float(ci.ppy))
        ds = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(ds.get_depth_scale())
        print(f"[OK] RealSense {IMAGE_WIDTH}x{IMAGE_HEIGHT} @ {FPS}fps  depth_scale={self.depth_scale}")
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
# Pipeline
# =========================================================================

class CanopyPipeline:
    def __init__(self):
        print("=" * 60)
        print("Canopy Height -- Method 1 (RealSense + SAM3 + EfficientSAM)")
        print("=" * 60)
        self.segmenter    = Sam3RefinedSegmenter()
        self.tag_detector = AprilTagDetector()
        print("[OK] Pipeline ready\n")

    def find_canopy_top(self, mask, depth_image, R, t,
                        fx, fy, cx, cy, depth_scale, depth_is_meters):
        if CANOPY_MASK_ERODE_PX > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2*CANOPY_MASK_ERODE_PX+1, 2*CANOPY_MASK_ERODE_PX+1))
            eroded = cv2.erode(mask.astype(np.uint8), k).astype(bool)
            mask = eroded if eroded.any() else mask

        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None

        min_y, max_y = int(ys.min()), int(ys.max())
        y_cutoff = min_y + int((max_y - min_y + 1) * CANOPY_TOP_FRACTION)
        sel = ys <= y_cutoff
        top_ys, top_xs = ys[sel], xs[sel]
        if len(top_ys) == 0:
            return None

        R_inv = R.T
        candidates = []
        for xi, yi in zip(top_xs, top_ys):
            p_cam = pixel_to_3d((int(xi), int(yi)), depth_image,
                                fx, fy, cx, cy, depth_scale, depth_is_meters)
            if p_cam is None:
                continue
            p_tag = R_inv @ p_cam - R_inv @ t
            candidates.append({
                "pixel":  (int(xi), int(yi)),
                "p_cam":  p_cam,
                "p_tag":  p_tag,
                "m1_y":   float(p_tag[1]),
                "height": float(-p_tag[1]),
            })
        if not candidates:
            return None

        heights = np.array([c["height"] for c in candidates])
        p_val   = np.percentile(heights, CANOPY_HEIGHT_PERCENTILE)
        in_top  = [c for c in candidates if c["height"] >= p_val]
        best = (min(in_top, key=lambda c: c["height"]) if in_top
                else max(candidates, key=lambda c: c["height"]))
        best["top_pixels"] = list(zip(top_xs.tolist(), top_ys.tolist()))
        return best

    def process(self, image_bgr, depth_image, intr,
                depth_scale, depth_is_meters, text_prompt=TEXT_PROMPT):
        fx, fy, cx, cy = intr

        tags = self.tag_detector.detect(image_bgr, camera_params=(fx, fy, cx, cy))
        tag_ref  = AprilTagDetector.get_by_id(tags, REF_TAG_ID)
        tag_base = AprilTagDetector.get_by_id(tags, BASELINE_TAG_ID)
        if tag_ref is None:
            return {"error": f"Reference tag {REF_TAG_ID} not detected"}, None

        R = np.asarray(tag_ref.pose_R, dtype=np.float64)
        t = np.asarray(tag_ref.pose_t, dtype=np.float64).flatten()

        scale = None
        if tag_base is not None:
            p1 = pixel_to_3d(tag_ref.corners[CORNER_TL], depth_image,
                             fx, fy, cx, cy, depth_scale, depth_is_meters)
            p2 = pixel_to_3d(tag_base.corners[CORNER_TR], depth_image,
                             fx, fy, cx, cy, depth_scale, depth_is_meters)
            if p1 is not None and p2 is not None and KNOWN_BASELINE_M is not None:
                baseline_m = float(np.linalg.norm(p2 - p1))
                scale = KNOWN_BASELINE_M / baseline_m
                print(f"    Baseline {baseline_m*100:.2f} cm  scale {scale:.4f}")
                if abs(scale - 1.0) > SCALE_WARN_THRESHOLD:
                    print(f"    [WARN] scale deviates >{SCALE_WARN_THRESHOLD} from 1.0")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = self.segmenter.segment(image_rgb, text_prompt)
        if mask.sum() == 0:
            return {"error": "Empty mask (no instances)"}, None

        best = self.find_canopy_top(mask, depth_image, R, t,
                                    fx, fy, cx, cy, depth_scale, depth_is_meters)
        if best is None:
            return {"error": "No valid depth at canopy top edge"}, None

        m1_h_tag_cm  = -best["m1_y"] * 100
        offset_cm    = TAG_TO_BASE_OFFSET_M * 100
        m1_h_base_cm = m1_h_tag_cm - offset_cm
        m1_h_base_scaled_cm = m1_h_base_cm * scale if scale is not None else None

        scaled_str = ("%+.2f" % m1_h_base_scaled_cm
                      if m1_h_base_scaled_cm is not None else "n/a")
        print(f"    M1 above tag {m1_h_tag_cm:+.2f} cm | "
              f"above base {m1_h_base_cm:+.2f} cm | scaled {scaled_str} cm")

        vis = self.visualize(image_bgr, mask, tags, best, R, t, intr)
        return {
            "m1_height_above_tag_cm":         m1_h_tag_cm,
            "m1_height_above_base_cm":        m1_h_base_cm,
            "m1_height_above_base_scaled_cm": m1_h_base_scaled_cm,
            "scale_factor":                   scale,
        }, vis

    def visualize(self, image_bgr, mask, tags, best, R, t, intr):
        fx, fy, cx, cy = intr
        out = image_bgr.copy()

        # EfficientSAM mask (orange)
        overlay = np.zeros_like(out)
        overlay[mask] = MASK_OVERLAY_BGR
        out = cv2.addWeighted(out, 1, overlay, MASK_OVERLAY_ALPHA, 0)

        # Top-fraction region (cyan)
        top = best.get("top_pixels")
        if top:
            pts = np.asarray(top)
            ys2, xs2 = pts[:, 1], pts[:, 0]
            out[ys2, xs2] = (0.5 * out[ys2, xs2]
                             + 0.5 * np.array(CANOPY_TOP_REGION_BGR)).astype(np.uint8)

        # Tags + axes
        tag_colors = {REF_TAG_ID: (0, 255, 255), BASELINE_TAG_ID: (255, 0, 255)}
        for tag in tags:
            c = tag_colors.get(tag.tag_id, (128, 128, 128))
            cv2.polylines(out, [tag.corners.astype(int)], True, c, 2)
            cv2.putText(out, f"ID:{tag.tag_id}",
                        (int(tag.center[0])+10, int(tag.center[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        self._draw_tag_axes(out, R, t, fx, fy, cx, cy, 0.05)

        # Canopy point
        cv2.circle(out, best["pixel"], 10, (0, 0, 255), -1)
        cv2.circle(out, best["pixel"], 12, (255, 255, 255), 2)

        # Canopy line (parallel to tag X, full image)
        dx_cam = R[:, 0]
        P = best["p_cam"]
        p0 = project(P, fx, fy, cx, cy)
        p1 = project(P + 0.05 * dx_cam, fx, fy, cx, cy)
        if p0 is not None and p1 is not None:
            d = np.array(p1, float) - np.array(p0, float)
            if np.linalg.norm(d) > 1e-6:
                d /= np.linalg.norm(d)
                L = 2 * max(out.shape[:2])
                pa = np.array(p0, float) - L * d
                pb = np.array(p0, float) + L * d
                ok, q1, q2 = cv2.clipLine((0, 0, out.shape[1], out.shape[0]),
                                          tuple(map(int, pa)), tuple(map(int, pb)))
                if ok:
                    cv2.line(out, q1, q2, CANOPY_LINE_BGR, CANOPY_LINE_THICKNESS)

        # Seed points
        for (px, py) in getattr(self.segmenter, "last_points", []):
            cv2.drawMarker(out, (int(px), int(py)), SEED_POINT_BGR,
                           cv2.MARKER_CROSS, 16, 2)
            cv2.circle(out, (int(px), int(py)), 5, (0, 0, 0), 1)

        return out

    def _draw_tag_axes(self, image, R, t, fx, fy, cx, cy, axis_length=0.05):
        o_px = project(t, fx, fy, cx, cy)
        if o_px is None:
            return
        for axis_vec, color, label in [
            (R @ np.array([axis_length, 0, 0]), (0,   0, 255), "X"),
            (R @ np.array([0, axis_length, 0]), (0, 255,   0), "Y"),
            (R @ np.array([0, 0, axis_length]), (255, 0,   0), "Z"),
        ]:
            e_px = project(t + axis_vec, fx, fy, cx, cy)
            if e_px is not None:
                cv2.arrowedLine(image, o_px, e_px, color, 2, tipLength=0.2)
                cv2.putText(image, label, (e_px[0]+5, e_px[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


# =========================================================================
# Output (ROS handoff)
# =========================================================================

def write_height(result):
    if "error" in result:
        with open(OUTPUT_TXT, "w") as f:
            f.write("")
        print(f"[ERROR] {result['error']}")
        return
    scaled = result.get("m1_height_above_base_scaled_cm")
    base   = result.get("m1_height_above_base_cm")
    value  = scaled if scaled is not None else base
    if value < MIN_HEIGHT_CM or value > MAX_HEIGHT_CM:
        with open(OUTPUT_TXT, "w") as f:
            f.write("")
        print(f"error value out of bounds is {value:.2f}")
        return
    with open(OUTPUT_TXT, "w") as f:
        f.write(f"{value:.2f}\n")
    print(f"[OK] Wrote {OUTPUT_TXT}: {value:.2f} cm")


def show_or_save(vis):
    if vis is None:
        return
    if SAVE_VIZ:
        cv2.imwrite(VIZ_PATH, vis)


# =========================================================================
# Entry point
# =========================================================================

def main():
    pipeline = CanopyPipeline()
    camera = RealSenseCamera().start()
    try:
        color, depth_raw = camera.grab()
        depth_m = depth_raw.astype(np.float32) * camera.depth_scale
        result, vis = pipeline.process(color, depth_m, camera.intr,
                                       camera.depth_scale, True, TEXT_PROMPT)
        write_height(result)
        show_or_save(vis)
        if vis is not None:
            cv2.imshow("canopy", vis)
            cv2.waitKey(0)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        camera.stop()
        cv2.destroyAllWindows()