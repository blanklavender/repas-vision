"""
femto_bolt_pointcloud_custom_calib.py

Build a colored point cloud from a single Femto Bolt raw RGB + raw depth pair,
using a user calibration loaded from JSON. This is the manual D2C path that
bypasses the SDK's AlignFilter.

Pipeline:
  1. Load calibration (intrinsics + depth->color extrinsics, rational distortion)
  2. Undistort depth pixel grid to normalized rays (cv2.undistortPoints)
  3. Back-project each valid depth pixel:  P_depth = ray * z
  4. Project P_depth into the color image (cv2.projectPoints applies extrinsics
     AND the color-camera distortion in one shot)
  5. Bilinear-sample color at the subpixel coordinates
  6. Save .ply and render an alignment overlay

Cells are separated with `# %%` for interactive use in VS Code / Jupyter.
"""

# %% ==================== IMPORTS ====================
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


# %% ==================== CONFIG ====================
CALIB_JSON  = "./femto_bolt_factory_calib.json"   # output of calibration_param_test.py
DEPTH_PATH  = "./raw_captures/session_20260520_114042/depth_f000.png"   # uint16, 1 count = 1 mm
COLOR_PATH  = "./raw_captures/session_20260520_114042/color_f000.png"   # uint8 BGR

OUT_PLY     = "cloud_custom_calib.ply"
OUT_OVERLAY = "alignment_overlay.png"

Z_MIN_MM = 100.0
Z_MAX_MM = 2000.0


# %% ==================== CALIBRATION ====================
@dataclass
class Calibration:
    K_depth: np.ndarray
    dist_depth: np.ndarray      # OpenCV rational order: [k1,k2,p1,p2,k3,k4,k5,k6]
    K_color: np.ndarray
    dist_color: np.ndarray
    R_d2c: np.ndarray           # X_color = R_d2c @ X_depth + t_d2c
    t_d2c_mm: np.ndarray        # shape (3,), mm
    depth_size: tuple           # (W, H)
    color_size: tuple           # (W, H)

def load_calibration(path) -> Calibration:
    """Load user calibration JSON. Assumes the `opencv_order` distortion fields
    are pre-arranged for OpenCV (length 8 = rational model)."""
    with open(path) as f:
        c = json.load(f)["user_calibrated"]

    di, ci = c["depth_intrinsics"], c["color_intrinsics"]
    dd, dc = c["depth_distortion"],  c["color_distortion"]
    ext    = c["depth_to_color"]

    return Calibration(
        K_depth    = np.asarray(di["K"], dtype=np.float64),
        dist_depth = np.asarray(dd["opencv_order"], dtype=np.float64),
        K_color    = np.asarray(ci["K"], dtype=np.float64),
        dist_color = np.asarray(dc["opencv_order"], dtype=np.float64),
        R_d2c      = np.asarray(ext["R"],    dtype=np.float64),
        t_d2c_mm   = np.asarray(ext["t_mm"], dtype=np.float64),
        depth_size = (di["width"], di["height"]),
        color_size = (ci["width"], ci["height"]),
    )

calib = load_calibration(CALIB_JSON)
print(f"Calibration loaded:")
print(f"  depth {calib.depth_size}  color {calib.color_size}")
print(f"  ||baseline|| = {np.linalg.norm(calib.t_d2c_mm):.2f} mm")
print(f"  distortion model: rational (8 coeffs per camera)")


# %% ==================== LOAD IMAGES ====================
depth_mm  = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)
color_bgr = cv2.imread(COLOR_PATH, cv2.IMREAD_COLOR)
assert depth_mm  is not None, f"Failed to load {DEPTH_PATH}"
assert color_bgr is not None, f"Failed to load {COLOR_PATH}"
assert depth_mm.dtype == np.uint16, f"Expected uint16 depth, got {depth_mm.dtype}"

depth_mm  = depth_mm.astype(np.float32)
color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

H_d, W_d = depth_mm.shape
H_c, W_c = color_rgb.shape[:2]
assert (W_d, H_d) == calib.depth_size, \
    f"Depth image {(W_d, H_d)} != calibration {calib.depth_size}"
assert (W_c, H_c) == calib.color_size, \
    f"Color image {(W_c, H_c)} != calibration {calib.color_size}"

valid = depth_mm > 0
print(f"Depth {W_d}x{H_d}  valid={valid.sum()}  "
      f"range=[{depth_mm[valid].min():.0f}, {depth_mm.max():.0f}] mm")
print(f"Color {W_c}x{H_c}")


# %% ==================== BACK-PROJECT DEPTH -> 3D ====================
def backproject_depth(depth_mm, K, dist, z_min_mm, z_max_mm):
    """Undistort the pixel grid and back-project to 3D points in the depth-camera
    frame (mm). cv2.undistortPoints handles 5/8/12/14-element distortion arrays
    transparently, so the rational model just works."""
    H, W = depth_mm.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([u.ravel(), v.ravel()], -1).astype(np.float32).reshape(-1, 1, 2)

    rays = cv2.undistortPoints(pix, K, dist).reshape(-1, 2)  # normalized (X/Z, Y/Z)

    d = depth_mm.ravel()
    keep = (d > z_min_mm) & (d < z_max_mm)
    d, rays = d[keep], rays[keep]

    P = np.empty((d.size, 3), dtype=np.float64)
    P[:, 0] = rays[:, 0] * d
    P[:, 1] = rays[:, 1] * d
    P[:, 2] = d
    return P, keep

P_depth_mm, _ = backproject_depth(depth_mm, calib.K_depth, calib.dist_depth,
                                  Z_MIN_MM, Z_MAX_MM)
print(f"Back-projected {len(P_depth_mm)} points")


# %% ==================== PROJECT INTO COLOR AND SAMPLE ====================
def project_into_color(P_depth_mm, calib):
    """Project depth-frame points into the color image. cv2.projectPoints applies
    the extrinsic (R, t) and the color-camera distortion in one shot."""
    rvec, _ = cv2.Rodrigues(calib.R_d2c)
    tvec = calib.t_d2c_mm.reshape(3, 1)
    uv, _ = cv2.projectPoints(P_depth_mm.reshape(-1, 1, 3), rvec, tvec,
                              calib.K_color, calib.dist_color)
    return uv.reshape(-1, 2)

def bilinear_sample(img, uv):
    """Bilinear color sampling. Returns colors in [0,1] (N,3) and an in-bounds mask (N,)."""
    H, W = img.shape[:2]
    u, v = uv[:, 0], uv[:, 1]
    inb = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)

    u_in, v_in = u[inb], v[inb]
    u0 = np.floor(u_in).astype(np.int32)
    v0 = np.floor(v_in).astype(np.int32)
    a = (u_in - u0)[:, None]
    b = (v_in - v0)[:, None]

    c00 = img[v0,     u0    ].astype(np.float32)
    c01 = img[v0,     u0 + 1].astype(np.float32)
    c10 = img[v0 + 1, u0    ].astype(np.float32)
    c11 = img[v0 + 1, u0 + 1].astype(np.float32)
    col = (1 - a) * (1 - b) * c00 + a * (1 - b) * c01 \
        + (1 - a) * b       * c10 + a * b       * c11
    return col / 255.0, inb

uv_color = project_into_color(P_depth_mm, calib)
colors, in_bounds = bilinear_sample(color_rgb, uv_color)

P_final_m = P_depth_mm[in_bounds] / 1000.0   # meters, depth-camera frame
print(f"Colored points: {len(P_final_m)} "
      f"({100*len(P_final_m)/len(P_depth_mm):.1f}% of back-projected)")


# %% ==================== SAVE POINT CLOUD ====================
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(P_final_m)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(OUT_PLY, pcd)
print(f"Saved -> {OUT_PLY}")


# %% ==================== 2D OVERLAY ====================
def render_depth_at_color(P_depth_mm, uv_color, color_shape):
    """Splat depth values into the color image grid with a near-wins z-buffer."""
    H, W = color_shape[:2]
    z = P_depth_mm[:, 2]
    u = np.round(uv_color[:, 0]).astype(np.int32)
    v = np.round(uv_color[:, 1]).astype(np.int32)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[inb], v[inb], z[inb]

    order = np.argsort(-z)              # far -> near; later writes (nearer) overwrite
    out = np.zeros((H, W), dtype=np.float32)
    out[v[order], u[order]] = z[order]
    return out

depth_on_color = render_depth_at_color(P_depth_mm, uv_color, color_rgb.shape)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(color_rgb);                       axes[0].set_title("Color")
axes[1].imshow(depth_on_color, cmap="turbo");    axes[1].set_title("Depth aligned to color")
axes[2].imshow(color_rgb)
axes[2].imshow(np.ma.masked_where(depth_on_color == 0, depth_on_color),
               cmap="turbo", alpha=0.55)
axes[2].set_title("Overlay (custom calib)")
for ax in axes: ax.set_axis_off()
plt.tight_layout()
plt.savefig(OUT_OVERLAY, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved -> {OUT_OVERLAY}")


# %% ==================== 3D VIEWER ====================
o3d.visualization.draw_geometries(
    [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)],
    window_name="Custom-calibrated point cloud",
)