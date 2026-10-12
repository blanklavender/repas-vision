"""
Capture-folder layout: the contract between the pipeline steps.

<capture>/
    manifest.csv              sweep_capture.py   one row per turntable stop
    camera_params.json        sweep_capture.py   colour + depth intrinsics, depth->colour extrinsic
    color_<stem>.png          sweep_capture.py   8-bit BGR 1280x720
    depth_<stem>.png          sweep_capture.py   16-bit depth, D2C-aligned to the colour image
    cloud_<stem>.ply          sweep_capture.py   SDK point cloud (optional, not used downstream)
    mask/mask_<stem>.png      sam3_data_cleaning SAM3 union mask (prompts), colour resolution
    rgb/color_<stem>.png      sam3_data_cleaning colour image, black outside the mask
    depth/depth_<stem>.png    sam3_data_cleaning aligned depth, 0 outside the cleaned mask
    pointcloud/cloud_<stem>.ply  sam3_data_cleaning cleaned cloud, colour-camera frame, metres
    cleaning.json             sam3_data_cleaning settings + per-frame result
    tag_pick/                 tag_pose_pick.py   stitch cache, plant frame, picks

<stem> = <index:03d>_ang<commanded deg:03d>, e.g. 012_ang120.
"""

import configparser
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

MANIFEST = "manifest.csv"
CAMERA_PARAMS = "camera_params.json"
CLEANING = "cleaning.json"
MASK_DIR, RGB_DIR, DEPTH_DIR, CLOUD_DIR = "mask", "rgb", "depth", "pointcloud"
MANIFEST_COLUMNS = ["index", "commanded_deg", "true_deg", "color", "depth", "cloud",
                    "valid_points", "total_points", "timestamp"]
_STEM = re.compile(r"^(?P<idx>\d+)(?:_ang(?P<ang>\d+))?$")


class DatasetError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------
def stem(index, commanded_deg):
    return f"{int(index):03d}_ang{int(commanded_deg) % 360:03d}"


def stem_of(filename):
    """'color_012_ang120.png' -> '012_ang120'."""
    return Path(filename).stem.split("_", 1)[1]


def color_name(s):
    return f"color_{s}.png"


def depth_name(s):
    return f"depth_{s}.png"


def mask_path(capture, s):
    return Path(capture) / MASK_DIR / f"mask_{s}.png"


def clean_rgb_path(capture, s):
    return Path(capture) / RGB_DIR / color_name(s)


def clean_depth_path(capture, s):
    return Path(capture) / DEPTH_DIR / depth_name(s)


def clean_cloud_path(capture, s):
    return Path(capture) / CLOUD_DIR / f"cloud_{s}.ply"


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------
def write_manifest(capture, rows):
    """Rewrite manifest.csv from `rows` (dicts with MANIFEST_COLUMNS keys)."""
    with open(Path(capture) / MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_manifest(capture):
    """Stops sorted by index: dicts with index, true_deg, color, depth, stem.

    Captures without a manifest.csv (very old ones) fall back to the color_*.png names;
    their angle is read from the _angNNN part of the name."""
    capture = Path(capture)
    path = capture / MANIFEST
    if path.is_file():
        with open(path, newline="") as f:
            raw = list(csv.DictReader(f))
        need = {"index", "true_deg", "color", "depth"}
        if not raw or not need <= set(raw[0]):
            raise DatasetError(f"{path} needs columns {sorted(need)}")
        rows = [dict(index=int(r["index"]), true_deg=float(r["true_deg"]), color=r["color"],
                     depth=r["depth"], stem=stem_of(r["color"])) for r in raw]
    else:
        rows = []
        for p in sorted(capture.glob("color_*.png")):
            s = stem_of(p.name)
            m = _STEM.match(s)
            if m is None:
                continue
            depth = next((f"depth_{s}{e}" for e in (".png", ".npy") if (capture / f"depth_{s}{e}").is_file()), "")
            rows.append(dict(index=int(m["idx"]), true_deg=float(m["ang"] or 0), color=p.name, depth=depth, stem=s))
        if not rows:
            raise DatasetError(f"no {MANIFEST} and no color_*.png in {capture}")
    rows.sort(key=lambda r: r["index"])
    return rows


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------
def _K(i):
    return np.array([[i["fx"], 0.0, i["cx"]], [0.0, i["fy"], i["cy"]], [0.0, 0.0, 1.0]], float)


def load_color_camera(capture, ini=None):
    """Colour intrinsics: dict K, dist (OpenCV order k1 k2 p1 p2 k3 k4 k5 k6), size (w, h),
    depth_scale_mm. From <capture>/camera_params.json, or an Orbbec CameraParam .ini."""
    path = Path(capture) / CAMERA_PARAMS
    if ini is None and path.is_file():
        c = json.loads(path.read_text())
        i = c["color_intrinsic"]
        return dict(K=_K(i), dist=np.array(c["color_distortion"]["opencv_order"], float),
                    size=(int(i["width"]), int(i["height"])),
                    depth_scale_mm=float(c.get("depth_scale_mm_per_unit", 1.0)), source=str(path))
    if ini is None:
        raise DatasetError(f"missing {path} (written by sweep_capture.py); pass an intrinsics .ini instead")
    cfg = configparser.ConfigParser()
    cfg.read_string(Path(ini).read_text())
    i, d = cfg["ColorIntrinsic"], cfg["ColorDistortion"]
    K = _K({k: float(i[k]) for k in ("fx", "fy", "cx", "cy")})
    dist = np.array([float(d[k]) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")])
    return dict(K=K, dist=dist, size=(int(i["width"]), int(i["height"])), depth_scale_mm=1.0, source=str(ini))


# ---------------------------------------------------------------------------
# cleaning summary
# ---------------------------------------------------------------------------
def read_cleaning(capture):
    path = Path(capture) / CLEANING
    if not path.is_file():
        raise DatasetError(f"missing {path} -- run sam3_data_cleaning.py on this capture first")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# picked points (the pipeline's final output)
# ---------------------------------------------------------------------------
def write_points_file(path, points, capture, frame_note):
    """Rewrite the picked-points text file: comment header + 'P<k> x y z' (mm) per point."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Flora pipeline picked points (rewritten on every pick)",
             f"# capture  {Path(capture).resolve()}",
             f"# frame    {frame_note}",
             f"# written  {datetime.now().isoformat(timespec='seconds')}",
             "# id  x_mm  y_mm  z_mm"]
    lines += [f"P{k} {x:.2f} {y:.2f} {z:.2f}" for k, (x, y, z) in enumerate(points, 1)]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)                       # readers never see a half-written file
