import numpy as np
import open3d as o3d
import cv2
from pathlib import Path

session_dir = Path("../../../captures/2026-03-21_flora_01_led/masked")

RADIUS    = 0.448
CAM_HEIGHT = 0.16
FX, FY = 748.8987, 748.3514
CX, CY = 639.8700, 361.9517

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
    """
    All points are in camera space. The plant rotated by angle_deg CCW
    (viewed from above). Camera Y points down, so 'above' = -Y direction.

    To undo: rotate points around the turntable axis by -angle_deg.
    Turntable center in camera coords: (0, CAM_HEIGHT, RADIUS)
    Turntable axis in camera coords: -Y direction (world up)
    """
    pivot = np.array([0.0, CAM_HEIGHT, RADIUS])

    # Translate to pivot
    pts = points - pivot

    # Rotate around Y axis
    # CCW around -Y by theta  =  CW around +Y by theta  =  Ry(-theta)
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    Ry = np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

    pts_rotated = (Ry @ pts.T).T
    return pts_rotated + pivot


# ============================================================
# Test with 0° and 180° first
# ============================================================
all_pcds = []
view_colors_debug = {0: [1,0,0], 180: [0,0,1]}

for ang in [0, 180]:
    astr = f"{ang:03d}"
    pts, cols = depth_to_points_colors(
        session_dir / f"depth_{astr}.png",
        session_dir / f"color_{astr}.png"
    )

    pts_aligned = undo_plant_rotation(pts, ang)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_aligned)
    pcd.paint_uniform_color(view_colors_debug[ang])
    all_pcds.append(pcd)
    print(f"{ang:>3}°: {len(pts)} points")

combined = all_pcds[0] + all_pcds[1]
o3d.io.write_point_cloud("debug_0_180_aligned.ply", combined)
print("Saved debug_0_180_aligned.ply  (Red=0°, Blue=180°)")