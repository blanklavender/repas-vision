import numpy as np
import open3d as o3d
import cv2

FX, FY = 748.8987, 748.3514
CX, CY = 639.8700, 361.9517
PIVOT_Z = 0.5125
PIVOT_X = -0.007

def depth_to_points_colors(depth_path, color_path):
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    color = cv2.imread(str(color_path))
    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64) / 1000.0
    valid = z > 0
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    colors = color_rgb[valid].astype(np.float64) / 255.0
    return points, colors

def undo_plant_rotation(points, angle_deg):
    pivot = np.array([PIVOT_X, 0.0, PIVOT_Z])
    pts = points - pivot
    theta = np.radians(angle_deg)  # positive sign confirmed
    c, s = np.cos(theta), np.sin(theta)
    Ry = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])
    pts_rotated = (Ry @ pts.T).T
    return pts_rotated + pivot

# 4-view test
view_colors = {0: [1,0,0], 90: [0,1,0], 180: [0,0,1], 270: [1,1,0]}
all_pcds = []

for ang in [0, 90, 180, 270]:
    pts, cols = depth_to_points_colors(f"../../../captures/2026-03-21_flora_01_led/masked/depth_{ang:03d}.png", f"../../../captures/2026-03-21_flora_01_led/masked/color_{ang:03d}.png")
    pts_aligned = undo_plant_rotation(pts, ang)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_aligned)
    pcd.paint_uniform_color(view_colors[ang])
    all_pcds.append(pcd)
    print(f"{ang:>3}°: {len(pts)} points")

combined = o3d.geometry.PointCloud()
for p in all_pcds:
    combined += p
o3d.io.write_point_cloud("debug_4view_colored.ply", combined)

# Also save a version with real colors
all_pcds_real = []
for ang in [0, 90, 180, 270]:
    pts, cols = depth_to_points_colors(f"../../../captures/2026-03-21_flora_01_led/masked/depth_{ang:03d}.png", f"../../../captures/2026-03-21_flora_01_led/masked/color_{ang:03d}.png")
    pts_aligned = undo_plant_rotation(pts, ang)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_aligned)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    all_pcds_real.append(pcd)

combined_real = o3d.geometry.PointCloud()
for p in all_pcds_real:
    combined_real += p
o3d.io.write_point_cloud("debug_4view_realcolor.ply", combined_real)

print(f"\nSaved debug_4view_colored.ply (Red=0° Green=90° Blue=180° Yellow=270°)")
print(f"Saved debug_4view_realcolor.ply (actual plant colors)")