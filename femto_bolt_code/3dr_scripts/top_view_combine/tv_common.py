"""
tv_common.py  --  what the top_view_combine scripts share: the work-folder layout, SAM3,
the depth filters, the robot-arm pose-transform file and the top-view camera.

Work folder (everything this folder writes stays inside it; the capture itself is only read):

<capture>/top_view_combine/
    registration.json           register_side.py   side shots' tag poses + turntable axis
    side/mask|rgb|depth|pointcloud/, side/cleaning.json
                                clean_side.py      SAM3 + radial crop, one cloud per side shot
    top/                        clean_top.py       SAM3 "green plant" cloud of the top shot
        mask_top.png, color_top.png, depth_top.png, cloud_top.ply, cleaning_top.json
    stitch/                     stitch_view.py     side_fused.ply, top_in_ref.ply, combined.ply,
                                                   poses.json, plant_frame.json, snap_*.png
    pipeline.log                main_pipeline.py

The capture provides (sweep_capture.py + the robot-arm top shot):
    manifest.csv, camera_params.json, color_<stem>.png, depth_<stem>.png    side sweep
    robot_rgbd/color.png (or color_bgr.npy), depth_mm.npy (or depth_raw.npy / .png),
    robot_rgbd/camera_params.json, robot_rgbd/camera_pose_transforms_<date>.txt
"""

import json
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "flora_pipeline"))
from flora_lib import dataset as ds  # noqa: E402
from flora_lib import geometry as geo  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
CHECKPOINT = str(REPO / "sam3_weights" / "sam3.pt")
CONFIDENCE = 0.5

WORK_DIR = "top_view_combine"
SIDE_DIR, TOP_DIR, STITCH_DIR = "side", "top", "stitch"
ROBOT_DIR = "robot_rgbd"
TOP_CLEANING = "cleaning_top.json"
TRANSFORMS_GLOB = "camera_pose_transforms*"


def work_dir(capture, name=WORK_DIR):
    return Path(capture) / name


# ---------------------------------------------------------------------------
# SAM3
# ---------------------------------------------------------------------------
def load_model(checkpoint=None, confidence=None):
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=checkpoint or CHECKPOINT,
                                   load_from_HF=False)
    return Sam3Processor(model, confidence_threshold=CONFIDENCE if confidence is None else confidence)


def autocast_ctx():
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def device_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (slow)"


def union_mask(processor, image, prompts, shape):
    """Union of every detected instance of every prompt -> (bool mask, detections per prompt)."""
    h, w = shape
    combined = np.zeros((h, w), dtype=bool)
    counts = {}
    with autocast_ctx():
        state = processor.set_image(image)
        for prompt in prompts:
            processor.reset_all_prompts(state)
            state = processor.set_text_prompt(state=state, prompt=prompt)
            for i in range(len(state["scores"])):
                m = state["masks"][i].squeeze(0).cpu().numpy()
                if m.shape != (h, w):
                    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                combined |= m.astype(bool)
            counts[prompt] = len(state["scores"])
    return combined, counts


# ---------------------------------------------------------------------------
# depth filters
# ---------------------------------------------------------------------------
def erode(mask, px):
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)


def gradient_mask(depth_mm, grad_max, hole_guard):
    """False on depth edges steeper than grad_max mm per pixel (Sobel / 8): flying pixels."""
    if grad_max <= 0:
        return np.ones(depth_mm.shape, bool)
    d = depth_mm.astype(np.float32)
    valid = d > 0
    if hole_guard > 0:                 # holes would make every neighbour an "edge"
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * hole_guard + 1,) * 2)
        d = np.where(valid, d, cv2.dilate(d, k))
    g = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    return g < grad_max


def near_filter(z, occluder, tol_mm, window_px):
    """True where z (metres, image grid, 0/inf = empty) lies at most tol_mm behind the nearest
    occluder pixel within window_px (a min filter over the occluder pixels only)."""
    if tol_mm <= 0:
        return np.ones(z.shape, bool)
    zk = np.where(occluder & (z > 0), z, np.inf).astype(np.float32)
    k = 2 * int(round(window_px)) + 1
    znear = cv2.erode(zk, np.ones((k, k), np.uint8))
    return z <= znear + tol_mm / 1000.0


# ---------------------------------------------------------------------------
# robot-arm pose transforms
# ---------------------------------------------------------------------------
_MAT_NAME = re.compile(r"^\s*([A-Za-z0-9]+_T_[A-Za-z0-9_]+)\s*:?\s*$")


def find_transforms_file(capture, robot_dir=ROBOT_DIR, explicit=None):
    """--transforms if given, else the newest camera_pose_transforms* in robot_rgbd/ (or the
    capture root). Names sort by their date suffix, so the last one wins."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise ds.DatasetError(f"--transforms {p} does not exist")
        return p
    for d in (Path(capture) / robot_dir, Path(capture)):
        found = sorted(p for p in d.glob(TRANSFORMS_GLOB) if p.suffix.lower() in (".txt", ".json"))
        if found:
            return found[-1]
    raise ds.DatasetError(f"no {TRANSFORMS_GLOB}.txt/.json in {Path(capture) / robot_dir} or {capture}")


def read_transforms(path):
    """Every named 4x4 matrix in the file -> {name: array}.

    .txt: a line holding only a name like 'B_T_C1' / 'C1_T_C2', followed (after any text
    lines) by four lines of four numbers. .json: {name: 4x4 nested list}."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
        return {k: np.asarray(v, float) for k, v in raw.items()
                if isinstance(v, list) and np.asarray(v).shape == (4, 4)}
    mats, name, rows = {}, None, []
    for line in path.read_text().splitlines():
        m = _MAT_NAME.match(line)
        if m:
            name, rows = m.group(1), []
            continue
        if name is None:
            continue
        try:
            vals = [float(x) for x in line.replace(",", " ").split()]
        except ValueError:
            continue
        if len(vals) == 4:
            rows.append(vals)
            if len(rows) == 4:
                mats[name], name = np.array(rows), None
    return mats


def top_to_side(mats, side="C1", top="C2", base="B"):
    """T with X_side = T @ X_top (both colour-optical frames), from the file's matrices.

    Uses C1_T_C2 directly if present, else inv(C2_T_C1), else inv(B_T_C1) @ B_T_C2.
    Returns (T, how, check) -- check: rotation / translation disagreement in degrees / mm
    between the direct matrix and the one through the base, when both are in the file."""
    direct, inverse = f"{side}_T_{top}", f"{top}_T_{side}"
    b1, b2 = f"{base}_T_{side}", f"{base}_T_{top}"
    via_base = np.linalg.inv(mats[b1]) @ mats[b2] if b1 in mats and b2 in mats else None
    if direct in mats:
        T, how = mats[direct], direct
    elif inverse in mats:
        T, how = np.linalg.inv(mats[inverse]), f"inv({inverse})"
    elif via_base is not None:
        T, how = via_base, f"inv({b1}) @ {b2}"
    else:
        raise ds.DatasetError(f"need {direct}, {inverse} or {b1} + {b2}; the file has {sorted(mats)}")
    check = None
    if via_base is not None and how != f"inv({b1}) @ {b2}":
        D = np.linalg.inv(via_base) @ T
        check = dict(rot_deg=geo.rot_deg(D[:3, :3]), trans_mm=float(np.linalg.norm(D[:3, 3]) * 1000))
    U, _, Vt = np.linalg.svd(T[:3, :3])                 # re-orthonormalise the printed rotation
    T = T.copy()
    T[:3, :3] = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return T, how, check


# ---------------------------------------------------------------------------
# top-view camera + images
# ---------------------------------------------------------------------------
def _K(i):
    return np.array([[i["fx"], 0.0, i["cx"]], [0.0, i["fy"], i["cy"]], [0.0, 0.0, 1.0]], float)


def _size(i):
    return int(i["width"]), int(i["height"])


def load_top_camera(capture, robot_dir=ROBOT_DIR):
    """Intrinsics of the top shot. robot_rgbd/camera_params.json holds the depth sensor
    ("intrinsics" or "depth_intrinsic") and the depth scale; what it lacks (colour
    intrinsics, depth distortion, depth->colour extrinsic) comes from the capture's own
    camera_params.json -- it is the same Femto Bolt.

    Returns dict color (K, dist, size), depth (K, dist, size), R_dc, t_dc (metres,
    X_color = R_dc @ X_depth + t_dc), depth_scale_mm, sources."""
    rp = Path(capture) / robot_dir / ds.CAMERA_PARAMS
    cp = Path(capture) / ds.CAMERA_PARAMS
    r = json.loads(rp.read_text()) if rp.is_file() else {}
    c = json.loads(cp.read_text()) if cp.is_file() else {}
    if not r and not c:
        raise ds.DatasetError(f"neither {rp} nor {cp} exists")

    def pick(key):
        return r.get(key) or c.get(key)

    ci, cd = pick("color_intrinsic"), pick("color_distortion")
    if ci is None:
        raise ds.DatasetError(f"no colour intrinsics in {rp} or {cp}")
    color = dict(K=_K(ci), dist=np.array(cd["opencv_order"], float) if cd else np.zeros(8), size=_size(ci))

    di = r.get("depth_intrinsic") or r.get("intrinsics") or c.get("depth_intrinsic")
    if di is None:
        raise ds.DatasetError(f"no depth intrinsics in {rp} or {cp}")
    dd = r.get("depth_distortion")
    if dd is None and c.get("depth_intrinsic") and _size(c["depth_intrinsic"]) == _size(di):
        dd = c.get("depth_distortion")                  # same sensor mode -> same distortion
    depth = dict(K=_K(di), dist=np.array(dd["opencv_order"], float) if dd else np.zeros(8), size=_size(di))

    dc = pick("depth_to_color")
    R_dc = np.asarray(dc["R"], float) if dc else np.eye(3)
    t_dc = np.asarray(dc["t_mm"], float) / 1000.0 if dc else np.zeros(3)
    scale = float(r.get("depth_scale_mm_per_unit", c.get("depth_scale_mm_per_unit", 1.0)))
    return dict(color=color, depth=depth, R_dc=R_dc, t_dc=t_dc, depth_scale_mm=scale,
                has_extrinsic=dc is not None, sources=[str(p) for p in (rp, cp) if p.is_file()])


def read_top_images(capture, robot_dir=ROBOT_DIR, color_name=None, depth_name=None, scale_mm=1.0):
    """(colour BGR uint8, depth in mm float32, colour path, depth path) of the top shot.

    Colour: color.png, else color_bgr.npy. Depth: depth_mm.npy (already mm), else
    depth_raw.npy / depth_raw.png / depth.png (raw units, x scale_mm)."""
    d = Path(capture) / robot_dir
    cands = [color_name] if color_name else ["color.png", "color_bgr.npy"]
    cpath = next((d / n for n in cands if (d / n).is_file()), None)
    if cpath is None:
        raise ds.DatasetError(f"no colour image ({cands}) in {d}")
    color = np.load(cpath) if cpath.suffix == ".npy" else cv2.imread(str(cpath), cv2.IMREAD_COLOR)
    cands = [depth_name] if depth_name else ["depth_mm.npy", "depth_raw.npy", "depth_raw.png", "depth.png"]
    dpath = next((d / n for n in cands if (d / n).is_file()), None)
    if dpath is None:
        raise ds.DatasetError(f"no depth image ({cands}) in {d}")
    depth = np.load(dpath) if dpath.suffix == ".npy" else cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise ds.DatasetError(f"cannot read {cpath if color is None else dpath}")
    depth_mm = depth.astype(np.float32) * (1.0 if dpath.stem.endswith("_mm") else scale_mm)
    return color, depth_mm, cpath, dpath
