#!/usr/bin/env python3

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


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FPS = 6

EFFICIENT_SAM_REPO_PATH = "../../EfficientSAM"
MODEL_TYPE = "vitt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

APRILTAG_FAMILY = "tag36h11"
APRILTAG_BLACK_SIZE_M = 0.028
APRILTAG_OUTER_SIZE_M = 0.036
APRILTAG_SIZE_M = APRILTAG_BLACK_SIZE_M

REF_TAG_ID = 16
BASELINE_TAG_ID = 15
CORNER_BL, CORNER_BR, CORNER_TR, CORNER_TL = 0, 1, 2, 3

TAG_TO_BASE_OFFSET_M = APRILTAG_OUTER_SIZE_M / 2

KNOWN_BASELINE_M = 0.65
SCALE_WARN_THRESHOLD = 0.05

NUM_TOP_EDGE_SAMPLES = 20
TOP_PERCENTILE = 1.0
EROSION_AREA_LOSS = 0.1
EROSION_MAX_ITER = 50

OUTPUT_TXT = "./canopy_heights.txt"


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

    return np.array([
        (x - cx) * depth_m / fx,
        (y - cy) * depth_m / fy,
        depth_m,
    ])


def erode_mask_by_area(mask, target_loss=EROSION_AREA_LOSS, max_iter=EROSION_MAX_ITER):
    orig_area = int(mask.sum())
    if orig_area == 0:
        return mask, 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = mask.astype(np.uint8)
    target_area = (1 - target_loss) * orig_area
    iterations = 0
    while int(eroded.sum()) > target_area:
        new_eroded = cv2.erode(eroded, kernel, iterations=1)
        if int(new_eroded.sum()) == 0:
            break
        eroded = new_eroded
        iterations += 1
        if iterations >= max_iter:
            break
    return eroded.astype(bool), iterations


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


class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.intr = None
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

        print(f"[OK] RealSense {IMAGE_WIDTH}x{IMAGE_HEIGHT} @ {FPS}fps")
        print(f"     fx={ci.fx:.2f}  fy={ci.fy:.2f}  cx={ci.ppx:.2f}  cy={ci.ppy:.2f}")
        print(f"     depth_scale = {self.depth_scale} m/unit")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        return self

    def grab(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        d = aligned.get_depth_frame()
        c = aligned.get_color_frame()
        if not d or not c:
            raise RuntimeError("Failed to capture aligned frames")
        return np.asanyarray(c.get_data()), np.asanyarray(d.get_data())

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def preview_and_capture(camera, tag_detector):
    print("\nLive preview -- 'e' to capture, 'q' to quit")
    while True:
        color, depth_raw = camera.grab()
        vis = color.copy()
        try:
            tags = tag_detector.detect(color)
            ids = [t.tag_id for t in tags]
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


class InteractivePromptSelector:
    def __init__(self, image_bgr):
        self.original = image_bgr.copy()
        self.points: List[List[int]] = []
        self.labels: List[int] = []
        self.box: Optional[Tuple[int, int, int, int]] = None
        self.box_mode = False
        self._box_start = None
        self._drag_end = None
        self._aborted = False

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


class CanopyPipeline:
    def __init__(self):
        print("=" * 60)
        print("Canopy Height -- M1 only (RealSense)")
        print("=" * 60)
        self.segmenter = EfficientSAMSegmenter()
        self.tag_detector = AprilTagDetector()
        print("[OK] Pipeline ready\n")

    def find_canopy_top(self, mask, depth_image, R, t,
                         fx, fy, cx, cy, depth_scale, depth_is_meters):
        if int(mask.sum()) == 0:
            return None

        eroded, n_iter = erode_mask_by_area(mask, EROSION_AREA_LOSS)
        print(f"    Erosion: {n_iter} iter, area {int(mask.sum())} -> {int(eroded.sum())}")
        if int(eroded.sum()) == 0:
            print("    [WARN] Erosion emptied mask; falling back to original")
            eroded = mask

        ys, xs = np.where(eroded)
        y_thresh = float(np.percentile(ys, TOP_PERCENTILE))
        sel = ys <= y_thresh
        top_xs = xs[sel]
        top_ys = ys[sel]
        print(f"    Top y range: {int(ys.min())} -> {int(ys.max())}, "
              f"{TOP_PERCENTILE}-th %ile y = {y_thresh:.1f}, "
              f"{len(top_xs)} candidate pixels at/above")

        if len(top_xs) > NUM_TOP_EDGE_SAMPLES:
            idx = np.linspace(0, len(top_xs)-1, NUM_TOP_EDGE_SAMPLES, dtype=int)
            top_xs = top_xs[idx]
            top_ys = top_ys[idx]

        R_inv = R.T
        candidates = []
        for x, y in zip(top_xs, top_ys):
            p_cam = pixel_to_3d((int(x), int(y)), depth_image,
                                 fx, fy, cx, cy, depth_scale, depth_is_meters)
            if p_cam is None:
                continue
            p_tag = R_inv @ p_cam - R_inv @ t
            candidates.append({
                "pixel": (int(x), int(y)),
                "p_cam": p_cam,
                "p_tag": p_tag,
                "m1_y": float(p_tag[1]),
            })

        if not candidates:
            return None

        return min(candidates, key=lambda c: c["p_cam"][1])

    def process(self, image_bgr, depth_image, intr,
                depth_scale, depth_is_meters,
                point_prompts=None, point_labels=None, box_prompt=None):

        print("=" * 60)
        print("PROCESSING")
        print("=" * 60)
        fx, fy, cx, cy = intr

        print("\n[1] Detecting AprilTags (with pose)...")
        tags = self.tag_detector.detect(image_bgr, camera_params=(fx, fy, cx, cy))
        tag_ref = AprilTagDetector.get_by_id(tags, REF_TAG_ID)
        tag_base = AprilTagDetector.get_by_id(tags, BASELINE_TAG_ID)
        print(f"    Detected IDs: {[t.tag_id for t in tags]}")
        if tag_ref is None:
            return {"error": f"Reference tag {REF_TAG_ID} not detected"}

        R = np.asarray(tag_ref.pose_R, dtype=np.float64)
        t = np.asarray(tag_ref.pose_t, dtype=np.float64).flatten()
        print(f"    Tag {REF_TAG_ID} pose_t (m): {t}")
        print(f"    det(R)={np.linalg.det(R):.6f}")

        scale = None
        if tag_base is not None:
            p_tl16 = tag_ref.corners[CORNER_TL]
            p_tr15 = tag_base.corners[CORNER_TR]
            p1 = pixel_to_3d(p_tl16, depth_image, fx, fy, cx, cy, depth_scale, depth_is_meters)
            p2 = pixel_to_3d(p_tr15, depth_image, fx, fy, cx, cy, depth_scale, depth_is_meters)
            if p1 is not None and p2 is not None:
                baseline_m = float(np.linalg.norm(p2 - p1))
                print(f"    Baseline (Tag{REF_TAG_ID} TL -> Tag{BASELINE_TAG_ID} TR): "
                      f"{baseline_m*100:.2f} cm")
                if KNOWN_BASELINE_M is not None:
                    scale = KNOWN_BASELINE_M / baseline_m
                    print(f"    Known: {KNOWN_BASELINE_M*100:.2f} cm  scale: {scale:.4f}")
                    if abs(scale - 1.0) > SCALE_WARN_THRESHOLD:
                        print(f"    [WARN] scale deviates >{SCALE_WARN_THRESHOLD}")
            else:
                print("    [WARN] Could not back-project both baseline corners")
        else:
            print(f"    [WARN] Baseline tag {BASELINE_TAG_ID} not detected")

        print("\n[2] Segmenting...")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if box_prompt is not None:
            mask = self.segmenter.segment_with_box(image_rgb, box_prompt)
        elif point_prompts is not None and point_labels is not None:
            mask = self.segmenter.segment_with_points(image_rgb, point_prompts, point_labels)
        else:
            return {"error": "No segmentation prompt provided"}

        if int(mask.sum()) == 0:
            return {"error": "Empty mask"}
        print(f"    Coverage: {mask.sum() / mask.size * 100:.1f}%")

        print("\n[3] Computing canopy height (M1, percentile + erosion)...")
        best = self.find_canopy_top(mask, depth_image, R, t,
                                     fx, fy, cx, cy,
                                     depth_scale, depth_is_meters)
        if best is None:
            return {"error": "No valid depth at canopy top edge"}

        m1_h_tag_cm = -best["m1_y"] * 100
        offset_cm = TAG_TO_BASE_OFFSET_M * 100
        m1_h_base_cm = m1_h_tag_cm - offset_cm
        m1_h_base_scaled_cm = m1_h_base_cm * scale if scale else None

        print("\n" + "=" * 60)
        print(f"  Canopy pixel:            {best['pixel']}")
        print(f"  p_cam (m):               {best['p_cam']}")
        print(f"  p_tag (m):               {best['p_tag']}")
        print(f"  M1 above tag center:     {m1_h_tag_cm:+.2f} cm")
        print(f"  M1 above reservoir base: {m1_h_base_cm:+.2f} cm  "
              f"(offset {offset_cm:.2f} cm)")
        if m1_h_base_scaled_cm is not None:
            print(f"  M1 above base (scaled):  {m1_h_base_scaled_cm:+.2f} cm  "
                  f"(x{scale:.4f})")
        print("=" * 60)

        return {
            "m1_height_above_tag_cm": m1_h_tag_cm,
            "m1_height_above_base_cm": m1_h_base_cm,
            "m1_height_above_base_scaled_cm": m1_h_base_scaled_cm,
            "scale_factor": scale,
        }


def write_height(results):
    if "error" in results:
        return
    h_base = results["m1_height_above_base_cm"]
    h_scaled = results["m1_height_above_base_scaled_cm"]
    value = h_scaled if h_scaled is not None else h_base
    with open(OUTPUT_TXT, "w") as f:
        f.write(f"{value:.2f}\n")
    print(f"[OK] Wrote {OUTPUT_TXT}: {value:+.2f}")


def main():
    pipeline = CanopyPipeline()
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

            depth_m = depth_raw.astype(np.float32) * camera.depth_scale

            results = pipeline.process(
                image_bgr=color, depth_image=depth_m,
                intr=camera.intr, depth_scale=camera.depth_scale,
                depth_is_meters=True,
                point_prompts=pts, point_labels=lbls, box_prompt=box,
            )
            if "error" in results:
                print(f"\n[ERROR] {results['error']}")
            else:
                write_height(results)

            print("\n--- Capture again? Press 'q' in preview to exit ---")
    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()