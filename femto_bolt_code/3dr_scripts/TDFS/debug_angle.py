import numpy as np
import open3d as o3d
import cv2

FX, FY = 748.8987, 748.3514
CX, CY = 639.8700, 361.9517

# Estimated from data instead of tape measure
PIVOT_Z = 0.5125
PIVOT_X = (-0.0168 + 0.0019) / 2  # average X centroid ≈ -0.007

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

    theta = np.radians(-angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    Ry = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

    pts_rotated = (Ry @ pts.T).T
    return pts_rotated + pivot

# Test with 0° and 90° — sign matters here
for sign_label, sign in [("negative", -1), ("positive", 1)]:
    all_pcds = []
    for ang, col in [(0, [1,0,0]), (90, [0,1,0])]:
        pts, cols = depth_to_points_colors(f"../../../captures/2026-03-21_flora_01_led/masked/depth_{ang:03d}.png", f"../../../captures/2026-03-21_flora_01_led/masked/color_{ang:03d}.png")
        # Apply rotation with current sign
        effective_angle = sign * ang
        pts_aligned = undo_plant_rotation(pts, effective_angle)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_aligned)
        pcd.paint_uniform_color(col)
        all_pcds.append(pcd)

    combined = all_pcds[0] + all_pcds[1]
    o3d.io.write_point_cloud(f"debug_0_90_{sign_label}.ply", combined)
    print(f"Saved debug_0_90_{sign_label}.ply")

print("\nOpen both in MeshLab. Red=0° Green=90°")
print("The correct one will have green filling in the side of the plant.")
print("The wrong one will have green overlapping or on the wrong side.")