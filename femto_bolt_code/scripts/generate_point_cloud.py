import numpy as np
import cv2
import open3d as o3d

# Femto Bolt intrinsics (1280x720)
FX, FY = 748.8987, 748.3514
CX, CY = 639.87, 361.9517
W, H = 1280, 720
DEPTH_SCALE = 0.001  # 1 count = 1 mm -> meters
Z_MIN, Z_MAX = 0.25, 8.0

def generate_pointcloud(color_path, depth_path, out_ply):
    # Load images
    color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # uint16, mm

    assert color.shape[:2] == (H, W), f"color shape {color.shape[:2]} != ({H},{W})"
    assert depth.shape[:2] == (H, W), f"depth shape {depth.shape[:2]} != ({H},{W})"

    # Convert depth to meters, apply valid-range mask
    Z = depth.astype(np.float32) * DEPTH_SCALE
    valid = (Z > Z_MIN) & (Z < Z_MAX)

    # Also treat fully-black color pixels as masked-out (common masking convention)
    color_mask = np.any(color > 0, axis=2)
    valid &= color_mask

    # Pixel grid -> 3D back-projection (OpenCV convention: X right, Y down, Z forward)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - CX) * Z / FX
    Y = (v - CY) * Z / FY

    pts = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)
    cols = color[valid].astype(np.float32) / 255.0

    # Build and save PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.io.write_point_cloud(out_ply, pcd, write_ascii=False)
    print(f"Saved {len(pts)} points -> {out_ply}")

if __name__ == "__main__":
    generate_pointcloud(
        color_path="..\\captures\\04-17_flora_matte_test\\capture_20260417_114924\\masked_calibrated\\color_000.png",
        depth_path="..\\captures\\04-17_flora_matte_test\\capture_20260417_114924\\masked_calibrated\\depth_000.png",
        out_ply="..\\captures\\04-17_flora_matte_test\\capture_20260417_114924\\masked_calibrated\\pointcloud_000.ply",
    )