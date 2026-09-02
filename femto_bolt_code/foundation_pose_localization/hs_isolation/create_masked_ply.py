#!/usr/bin/env python3
"""
masked_pointcloud.py - Generate a masked point cloud from RGB, depth, and segmentation mask
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d

# -------------------- CONFIG --------------------
COLOR_PATH      = Path(r"../../captures/captures_colorworld_stable/capture_20251121_162722/color_20251121_162722.png")
DEPTH_NPY       = Path(r"../../captures/captures_colorworld_stable/capture_20251121_162722/aligned_depth_m_20251121_162722.npy")
MASK_PATH       = Path(r"../../captures/captures_colorworld_stable/capture_20251121_162722/segmented_mask_20251121_162722.png")  
INTRINSICS_JSON = Path(r"./calibration_parameters/factory_color_intrinsics_2025-09-08T143506.json")

OUTPUT_PLY      = Path(r"../../captures/captures_colorworld_stable/capture_20251121_162722/masked_pointcloud_20251121_162722.ply")

# Options
INVERT_MASK     = False    # True if object=black, background=white
VOXEL_SIZE      = 0.0      # meters (0 to disable downsampling)
REMOVE_OUTLIERS = True     # statistical outlier removal
VISUALIZE       = True     # show result before saving

# -------------------- Intrinsics I/O --------------------
def load_color_intrinsics(json_path: Path):
    """Load COLOR intrinsics from JSON. Supports lean or master bundle format."""
    if not json_path.exists():
        raise FileNotFoundError(f"Intrinsics JSON not found: {json_path}")
    data = json.loads(json_path.read_text())

    intr = data["color_intrinsics"] if (isinstance(data, dict) and "color_intrinsics" in data) else data

    required = ("fx", "fy", "cx", "cy")
    for k in required:
        if k not in intr:
            raise KeyError(f"Missing '{k}' in intrinsics JSON: {json_path}")

    fx = float(intr["fx"]); fy = float(intr["fy"])
    cx = float(intr["cx"]); cy = float(intr["cy"])
    iw = int(intr.get("width", 0)); ih = int(intr.get("height", 0))
    return fx, fy, cx, cy, iw, ih


def scale_intrinsics(fx, fy, cx, cy, src_w, src_h, dst_w, dst_h):
    """Scale intrinsics from (src_w,src_h) to (dst_w,dst_h)."""
    if src_w <= 0 or src_h <= 0:
        return fx, fy, cx, cy
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    return fx * sx, fy * sy, cx * sx, cy * sy


# -------------------- Point Cloud Generation --------------------
def create_masked_pointcloud(rgb: np.ndarray, 
                              depth_m: np.ndarray, 
                              mask: np.ndarray,
                              fx: float, fy: float, cx: float, cy: float,
                              invert_mask: bool = False) -> o3d.geometry.PointCloud:
    """
    Generate a colored point cloud from masked regions.
    
    Args:
        rgb: HxWx3 uint8 BGR image
        depth_m: HxW float depth in meters (aligned to color)
        mask: HxW uint8 mask (255=object, 0=background by default)
        fx, fy, cx, cy: camera intrinsics
        invert_mask: if True, treat 0 as object and 255 as background
    
    Returns:
        Open3D PointCloud
    """
    h, w = depth_m.shape
    
    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Build mask condition
    if invert_mask:
        mask_bool = mask == 0
    else:
        mask_bool = mask > 0
    
    # Valid = masked AND positive finite depth
    valid = mask_bool & np.isfinite(depth_m) & (depth_m > 0)
    
    # Extract valid pixels
    u_valid = u[valid].astype(np.float64)
    v_valid = v[valid].astype(np.float64)
    z = depth_m[valid].astype(np.float64)
    
    # Back-project to 3D (pinhole model)
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy
    
    points = np.stack([x, y, z], axis=-1)
    
    # Get colors (convert BGR -> RGB and normalize)
    colors = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)[valid] / 255.0
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


# -------------------- Main --------------------
def main():
    # --- Validate paths ---
    for p, name in [(COLOR_PATH, "Color"), (DEPTH_NPY, "Depth"), 
                    (MASK_PATH, "Mask"), (INTRINSICS_JSON, "Intrinsics")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found: {p}")
    
    # --- Load images ---
    rgb = cv2.imread(str(COLOR_PATH), cv2.IMREAD_COLOR)
    depth_m = np.load(str(DEPTH_NPY))
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    
    if rgb is None:
        raise RuntimeError(f"Failed to read color image: {COLOR_PATH}")
    if mask is None:
        raise RuntimeError(f"Failed to read mask image: {MASK_PATH}")
    
    h, w = rgb.shape[:2]
    print(f"[INFO] Color image: {w}x{h}")
    print(f"[INFO] Depth shape: {depth_m.shape}")
    print(f"[INFO] Mask shape: {mask.shape}")
    
    # --- Check alignment ---
    if depth_m.shape != (h, w):
        raise RuntimeError(f"Depth size {depth_m.shape} != color size {(h, w)}")
    if mask.shape != (h, w):
        # Try resizing mask if dimensions differ
        print(f"[WARN] Resizing mask from {mask.shape} to {(h, w)}")
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # --- Load & scale intrinsics ---
    fx0, fy0, cx0, cy0, iw, ih = load_color_intrinsics(INTRINSICS_JSON)
    
    if iw > 0 and ih > 0 and (iw != w or ih != h):
        fx, fy, cx, cy = scale_intrinsics(fx0, fy0, cx0, cy0, iw, ih, w, h)
        print(f"[INTR] Scaled {iw}x{ih} -> {w}x{h}")
    else:
        fx, fy, cx, cy = fx0, fy0, cx0, cy0
    
    print(f"[INTR] fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    
    # --- Mask statistics ---
    mask_pixels = np.sum(mask > 0) if not INVERT_MASK else np.sum(mask == 0)
    print(f"[MASK] {mask_pixels:,} pixels selected ({100*mask_pixels/(h*w):.1f}%)")
    
    # --- Generate point cloud ---
    pcd = create_masked_pointcloud(rgb, depth_m, mask, fx, fy, cx, cy, 
                                   invert_mask=INVERT_MASK)
    
    print(f"[PCD] Generated {len(pcd.points):,} points")
    
    # --- Optional: voxel downsampling ---
    if VOXEL_SIZE > 0:
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        print(f"[PCD] After voxel downsampling ({VOXEL_SIZE*1000:.1f}mm): {len(pcd.points):,} points")
    
    # --- Optional: outlier removal ---
    if REMOVE_OUTLIERS and len(pcd.points) > 100:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"[PCD] After outlier removal: {len(pcd.points):,} points")
    
    # --- Compute normals (useful for mesh reconstruction later) ---
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    
    # --- Save ---
    o3d.io.write_point_cloud(str(OUTPUT_PLY), pcd)
    print(f"[SAVE] Wrote {OUTPUT_PLY}")
    
    # --- Visualize ---
    if VISUALIZE:
        # Add coordinate frame at origin for reference
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, axes], 
                                          window_name="Masked Point Cloud",
                                          point_show_normal=False)


if __name__ == "__main__":
    main()