import numpy as np
import open3d as o3d
import cv2
from pathlib import Path

# === Config ===
data_dir = Path("../../../captures/2026-03-21_flora_01_led/masked")  # folder with masked color/depth
output_dir = Path("./TSDF_output")
output_dir.mkdir(exist_ok=True)

FX, FY = 748.8987, 748.3514
CX, CY = 639.8700, 361.9517
W, H = 1280, 720

PIVOT_Z = 0.5125
PIVOT_X = -0.007
ANGLES = list(range(0, 360, 5))

# === Validated rotation ===
def undo_rotation_matrix(angle_deg):
    theta = np.radians(-angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

# ============================================================
# METHOD 1: Direct point cloud fusion (VALIDATED, guaranteed)
# ============================================================
def run_pointcloud_fusion():
    print("=" * 60)
    print("POINT CLOUD FUSION — all 72 frames")
    print("=" * 60)

    all_points = []
    all_colors = []

    for ang in ANGLES:
        astr = f"{ang:03d}"
        depth_path = data_dir / f"depth_{astr}.png"
        color_path = data_dir / f"color_{astr}.png"
        if not depth_path.exists():
            continue

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        color = cv2.imread(str(color_path))
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float64) / 1000.0
        valid = z > 0
        x = (u - CX) * z / FX
        y = (v - CY) * z / FY

        pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
        cols = color_rgb[valid].astype(np.float64) / 255.0

        # Undo plant rotation
        pivot = np.array([PIVOT_X, 0.0, PIVOT_Z])
        Ry = undo_rotation_matrix(ang)
        pts_aligned = (Ry @ (pts - pivot).T).T + pivot

        all_points.append(pts_aligned)
        all_colors.append(cols)
        print(f"  {ang:>3}°: {len(pts)} points")

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=0.001)  # 1mm resolution
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

    # Save point cloud
    o3d.io.write_point_cloud(str(output_dir / "plant_pointcloud.ply"), pcd_down)
    print(f"\nPoint cloud: {len(pcd_down.points)} points → plant_pointcloud.ply")

    # Generate mesh via Poisson reconstruction
    print("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_down, depth=9)

    # Remove low-density vertices (noisy outliers)
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(str(output_dir / "plant_mesh_poisson.ply"), mesh)
    print(f"Mesh: {len(mesh.vertices)} vertices → plant_mesh_poisson.ply")

    return pcd_down

# ============================================================
# METHOD 2: TSDF fusion
# ============================================================
def run_tsdf_fusion():
    print("\n" + "=" * 60)
    print("TSDF FUSION — all 72 frames")
    print("=" * 60)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, FX, FY, CX, CY)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.001,     # 1mm voxels
        sdf_trunc=0.005,        # 5mm truncation
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated = 0
    for ang in ANGLES:
        astr = f"{ang:03d}"
        color_path = data_dir / f"color_{astr}.png"
        depth_path = data_dir / f"depth_{astr}.png"
        if not color_path.exists() or not depth_path.exists():
            continue

        color = o3d.io.read_image(str(color_path))
        depth = o3d.io.read_image(str(depth_path))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=1000.0,
            depth_trunc=0.6,
            convert_rgb_to_intensity=False,
        )

        # Camera-to-world transform (same math as point cloud fusion)
        pivot = np.array([PIVOT_X, 0.0, PIVOT_Z])
        Ry = undo_rotation_matrix(ang)
        t = pivot - Ry @ pivot

        c2w = np.eye(4)
        c2w[:3, :3] = Ry
        c2w[:3, 3] = t

        # Open3D integrate expects world-to-camera = inverse of c2w
        w2c = np.linalg.inv(c2w)
        volume.integrate(rgbd, intrinsic, w2c)

        integrated += 1
        print(f"  [{integrated:>3}/{len(ANGLES)}]  {ang:>3}° integrated")

    # Extract mesh
    print("Extracting TSDF mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output_dir / "plant_mesh_tsdf.ply"), mesh)
    print(f"TSDF mesh: {len(mesh.vertices)} vertices → plant_mesh_tsdf.ply")

    # Extract point cloud
    pcd = volume.extract_point_cloud()
    o3d.io.write_point_cloud(str(output_dir / "plant_pcd_tsdf.ply"), pcd)
    print(f"TSDF pcd: {len(pcd.points)} points → plant_pcd_tsdf.ply")

    return mesh

# ============================================================
# RUN BOTH
# ============================================================
pcd = run_pointcloud_fusion()
mesh = run_tsdf_fusion()

print("\n" + "=" * 60)
print("DONE — open these in MeshLab to compare:")
print(f"  {output_dir / 'plant_pointcloud.ply'}")
print(f"  {output_dir / 'plant_mesh_poisson.ply'}")
print(f"  {output_dir / 'plant_mesh_tsdf.ply'}")
print(f"  {output_dir / 'plant_pcd_tsdf.ply'}")
print("=" * 60)