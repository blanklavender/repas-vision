import configparser
import glob
import os
from contextlib import nullcontext

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

INPUT_DIR = "../../flora_turntable/captures/yellow_pot_08-25_deg_10_speed_5"
PROMPT_A = "yellow cylinder"
PROMPT_B = "green plant"

CHECKPOINT = "./sam3_weights/sam3.pt"
CONFIDENCE = 0.5

INTRINSICS_FILE = "../scripts/calibration_parameters/CameraParam_Femto_BoltCL8956200ZM_Color1280x720_Depth640x576.ini"
INTRINSIC_SECTION = "ColorIntrinsic"
DISTORTION_SECTION = "ColorDistortion"
UNDISTORT_RAYS = True

DEPTH_SCALE_M = 0.001
X_MIN_M, X_MAX_M = -0.25, 0.25
Y_MIN_M, Y_MAX_M = -0.25, 0.25
Z_MIN_M, Z_MAX_M = 0.30, 0.60

DEPTH_GRAD_MAX_MM_PX = 20.0
HOLE_GUARD_PX = 1
SOR_NEIGHBORS = 20
SOR_STD_RATIO = 2.0

MASK_ERODE_PX = 2
CROP_APPLIES_TO_RGB = False
COLOR_GLOB = "color_*.png"
DEPTH_EXTS = (".npy", ".png")
RGB_SUBDIR = "rgb"
DEPTH_SUBDIR = "depth"
PCD_SUBDIR = "pointcloud"

_RAY_CACHE = {}

import re

ANGLE_INCREMENT_DEG = 10.0          # used only if a file has no _angNNN field
NAME_RE = re.compile(r"^(?P<idx>\d+)(?:_ang(?P<ang>\d+))?$")
_DEPTH_INDEX = {}


def parse_suffix(suffix):
    """'012_ang120' -> (12, 120).  '009' -> (9, 9*ANGLE_INCREMENT_DEG)."""
    m = NAME_RE.match(suffix)
    if not m:
        return None, None
    idx = int(m.group("idx"))
    ang = m.group("ang")
    angle = int(ang) if ang is not None else int(round(idx * ANGLE_INCREMENT_DEG))
    return idx, angle % 360


def load_intrinsics(path):
    cfg = configparser.ConfigParser()
    with open(path) as f:
        cfg.read_string(f.read())
    i, d = cfg[INTRINSIC_SECTION], cfg[DISTORTION_SECTION]
    K = np.array([[float(i["fx"]), 0.0, float(i["cx"])],
                  [0.0, float(i["fy"]), float(i["cy"])],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([float(d["k1"]), float(d["k2"]), float(d["p1"]), float(d["p2"]),
                     float(d["k3"]), float(d["k4"]), float(d["k5"]), float(d["k6"])], dtype=np.float64)
    return K, dist, int(i["width"]), int(i["height"])


def scale_K(K, src_wh, dst_wh):
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    out = K.copy()
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out


def ray_map(K, dist, src_wh, w, h):
    if (w, h) not in _RAY_CACHE:
        Ks = scale_K(K, src_wh, (w, h))
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        if UNDISTORT_RAYS:
            pts = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2).astype(np.float64)
            norm = cv2.undistortPoints(pts, Ks, dist).reshape(h, w, 2)
            xn, yn = norm[..., 0].astype(np.float32), norm[..., 1].astype(np.float32)
        else:
            xn = (xs - Ks[0, 2]) / Ks[0, 0]
            yn = (ys - Ks[1, 2]) / Ks[1, 1]
        _RAY_CACHE[(w, h)] = (xn, yn)
        print(f"[INTR] {w}x{h}  fx={Ks[0,0]:.3f} fy={Ks[1,1]:.3f} cx={Ks[0,2]:.3f} cy={Ks[1,2]:.3f}"
              f"  undistort={UNDISTORT_RAYS}")
    return _RAY_CACHE[(w, h)]


def load_model():
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=CHECKPOINT, load_from_HF=False)
    return Sam3Processor(model, confidence_threshold=CONFIDENCE)


def autocast_ctx():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def union_mask(processor, image, prompts, shape):
    h, w = shape
    combined = np.zeros((h, w), dtype=bool)
    counts = {}
    with autocast_ctx():
        state = processor.set_image(image)
        for prompt in prompts:
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=prompt)
            n = 0
            for i in range(len(state["scores"])):
                m = state["masks"][i].squeeze(0).cpu().numpy()
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                combined |= m.astype(bool)
                n += 1
            counts[prompt] = n
    return combined, counts


def find_depth(folder, suffix):
    if not _DEPTH_INDEX:
        for ext in DEPTH_EXTS:
            for p in sorted(glob.glob(os.path.join(folder, f"depth_*{ext}"))):
                stem = os.path.splitext(os.path.basename(p))[0]
                idx, _ = parse_suffix(stem.split("depth_", 1)[-1])
                # idx, angle = parse_suffix(suffix)
                if idx is not None:
                    _DEPTH_INDEX.setdefault(idx, (p, ext))
    idx, _ = parse_suffix(suffix)
    hit = _DEPTH_INDEX.get(idx)
    if hit is None:
        return None, None
    path, ext = hit
    depth = np.load(path) if ext == ".npy" else cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return depth, ext


def erode(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def box_mask(depth, xn, yn):
    z = depth.astype(np.float32) * DEPTH_SCALE_M
    ok = (z > Z_MIN_M) & (z < Z_MAX_M)
    ok &= (xn * z > X_MIN_M) & (xn * z < X_MAX_M)
    ok &= (yn * z > Y_MIN_M) & (yn * z < Y_MAX_M)
    return ok, z


def gradient_mask(depth):
    if DEPTH_GRAD_MAX_MM_PX <= 0:
        return np.ones(depth.shape[:2], dtype=bool)
    d = depth.astype(np.float32)
    valid = d > 0
    if HOLE_GUARD_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * HOLE_GUARD_PX + 1,) * 2)
        filled = cv2.dilate(d, k)
        d = np.where(valid, d, filled)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    return (np.hypot(gx, gy) / 8.0) < DEPTH_GRAD_MAX_MM_PX


def backproject(color_rgb, z, mask, xn, yn):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    zz = z[ys, xs]
    pts = np.stack([xn[ys, xs] * zz, yn[ys, xs] * zz, zz], axis=1)
    cols = color_rgb[ys, xs].astype(np.float64) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def main():
    prompts = [PROMPT_A, PROMPT_B]
    rgb_out = os.path.join(INPUT_DIR, RGB_SUBDIR)
    depth_out = os.path.join(INPUT_DIR, DEPTH_SUBDIR)
    pcd_out = os.path.join(INPUT_DIR, PCD_SUBDIR)
    for d in (rgb_out, depth_out, pcd_out):
        os.makedirs(d, exist_ok=True)

    K, dist, src_w, src_h = load_intrinsics(INTRINSICS_FILE)

    color_paths = sorted(glob.glob(os.path.join(INPUT_DIR, COLOR_GLOB)))
    if not color_paths:
        raise SystemExit(f"no images matching {COLOR_GLOB} in {INPUT_DIR}")
    print(f"[SAM3] {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[RUN ] {len(color_paths)} frame(s) | prompts: {prompts}")
    print(f"[CROP] x[{X_MIN_M:+.3f},{X_MAX_M:+.3f}] y[{Y_MIN_M:+.3f},{Y_MAX_M:+.3f}] "
          f"z[{Z_MIN_M:.3f},{Z_MAX_M:.3f}] m | grad<{DEPTH_GRAD_MAX_MM_PX} mm/px | "
          f"sor={SOR_NEIGHBORS}/{SOR_STD_RATIO}")

    processor = load_model()
    kept = 0

    for path in color_paths:
        name = os.path.basename(path)
        suffix = os.path.splitext(name)[0].split("color_", 1)[-1]

        color_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if color_bgr is None:
            print(f"[SKIP] unreadable: {name}")
            continue
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        h, w = color_rgb.shape[:2]

        depth, depth_ext = find_depth(INPUT_DIR, suffix)
        if depth is None:
            print(f"[SKIP] no depth for: {name}")
            continue
        if depth.shape[:2] != (h, w):
            print(f"[SKIP] depth {depth.shape[:2]} != color {(h, w)}: {name}")
            continue

        xn, yn = ray_map(K, dist, (src_w, src_h), w, h)

        seg, counts = union_mask(processor, Image.fromarray(color_rgb), prompts, (h, w))
        if not seg.any():
            print(f"[SKIP] empty union mask: {name}  {counts}")
            continue

        box, z = box_mask(depth, xn, yn)
        grad = gradient_mask(depth)
        clean = box & grad

        rgb_mask = seg & clean if CROP_APPLIES_TO_RGB else seg
        masked_rgb = np.where(rgb_mask[..., None], color_rgb, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(rgb_out, name), cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))

        depth_mask = erode(seg, MASK_ERODE_PX) & clean
        masked_depth = np.where(depth_mask, depth, 0).astype(depth.dtype)
        depth_name = f"depth_{suffix}{depth_ext}"
        if depth_ext == ".npy":
            np.save(os.path.join(depth_out, depth_name), masked_depth)
        else:
            cv2.imwrite(os.path.join(depth_out, depth_name), masked_depth)

        pcd = backproject(color_rgb, z, depth_mask, xn, yn)
        if pcd is None:
            print(f"[WARN] nothing survives cropping: {name}  {counts}")
            continue
        n_raw = len(pcd.points)
        if SOR_STD_RATIO > 0 and n_raw > SOR_NEIGHBORS:
            pcd, _ = pcd.remove_statistical_outlier(SOR_NEIGHBORS, SOR_STD_RATIO)
        o3d.io.write_point_cloud(os.path.join(pcd_out, f"cloud_{suffix}.ply"), pcd)

        n_seg = int(seg.sum())
        n_box = int((seg & box).sum())
        n_grad = int((seg & clean).sum())
        kept += 1
        print(f"[OK  ] {name}  seg={n_seg:,} -box-> {n_box:,} -grad-> {n_grad:,} "
              f"-erode-> {n_raw:,} -sor-> {len(pcd.points):,}  {counts}")

    print(f"[DONE] {kept}/{len(color_paths)} frames written to "
          f"{INPUT_DIR}/{{{RGB_SUBDIR},{DEPTH_SUBDIR},{PCD_SUBDIR}}}")


if __name__ == "__main__":
    main()