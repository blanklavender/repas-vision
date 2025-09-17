import cv2
import numpy as np
import open3d as o3d
from pupil_apriltags import Detector

# === Configuration === [FILES DONT EXIST]
IMAGE_PATH = "../02_ply_generation/ply_generation_scripts/out_20250728_145935.png"
PLY_PATH   = "../02_ply_generation/ply_generation_scripts/out_20250728_145935_v2.ply"
CAD_PATH   = "../cad_model/Structure2.PLY"

# Camera intrinsics & tag size
# Checkerboard calibration values for D415i @ 640x480
K = np.array([
    [605.2845686, 0.0,           309.95995203],
    [0.0,          605.44233933,  229.79166863],
    [0.0,          0.0,            1.0         ]
], dtype=np.float32)
DIST_COEFFS = np.array([0.04344582, 0.32076285, -0.00060687, -0.0004814, -1.40593456], dtype=np.float32)
TAG_SIZE = 0.0303  # updated tag size in meters

# Convert OpenCV (x→right,y→down,z→forward) to Open3D (x→right,y→up,z→into-screen)
CV2O3D = np.diag([1.0, -1.0, -1.0])

# Local AprilTag detector (robust fallback)
local_detector = Detector(
    families='tag36h11',
    nthreads=8,
    quad_decimate=1.0,
    quad_sigma=0.8,
    refine_edges=1,
    decode_sharpening=0.5
)


def preprocess_gray(img: np.ndarray) -> np.ndarray:
    """
    Enhance a BGR image by converting to grayscale, applying CLAHE, and gamma correction.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    gamma = 1.5
    inv_gamma = 1.0 / gamma
    lut = np.array([(i/255.0)**inv_gamma * 255 for i in range(256)], dtype='uint8')
    return cv2.LUT(equalized, lut)


def detect_tag_pose(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect an AprilTag using local detection and return its pose (R, t) in Open3D coords.
    """
    # Load and preprocess
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    gray = preprocess_gray(frame)

    # Detect tag corners
    tags = local_detector.detect(gray)
    if not tags:
        raise RuntimeError("No AprilTags detected.")
    tag = tags[0]
    corners = tag.corners.astype(np.float32)

    # Define tag corner points in tag frame
    half = TAG_SIZE / 2.0
    obj_pts = np.array([
        [-half, -half, 0],
        [ half, -half, 0],
        [ half,  half, 0],
        [-half,  half, 0]
    ], dtype=np.float32)

    # Solve PnP: camera_R_tag and camera_t_tag in OpenCV coords
    success, rvec, tvec = cv2.solvePnP(
        obj_pts, corners, K, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("solvePnP failed to recover pose.")
    R_cv, _ = cv2.Rodrigues(rvec)  # rotation (3x3)
    t_cv = tvec.flatten()         # translation (3,)

    # Transform into Open3D frame:
    # R_o3d = M_cv2o3d * R_cv * M_o3d2cv  (M_o3d2cv == M_cv2o3d)
    R_o3d = CV2O3D @ R_cv @ CV2O3D
    # t_o3d = M_cv2o3d * t_cv
    t_o3d = CV2O3D @ t_cv

    # Print distance for sanity check
    dist = np.linalg.norm(t_o3d)
    print(f"Camera→Tag distance: {dist:.3f} m")
    return R_o3d, t_o3d


def load_point_cloud(ply_path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"Failed to load point cloud: {ply_path}")
    return pcd


def load_cad_mesh(cad_path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(cad_path)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load CAD mesh: {cad_path}")
    mesh.compute_vertex_normals()
    return mesh


def transform_cad_mesh(mesh: o3d.geometry.TriangleMesh, R: np.ndarray, t: np.ndarray) -> o3d.geometry.TriangleMesh:
    verts = np.asarray(mesh.vertices)
    # Apply rotation then translation
    verts_cam = (R @ verts.T).T + t
    mesh_cam = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_cam),
        triangles=mesh.triangles
    )
    mesh_cam.compute_vertex_normals()
    return mesh_cam


def visualize(pcd: o3d.geometry.PointCloud, mesh_cam: o3d.geometry.TriangleMesh, R: np.ndarray, t: np.ndarray) -> None:
    """
    Visualize the point cloud, CAD mesh, camera origin, and AprilTag axes.
    """
    mesh_cam.paint_uniform_color([1.0, 0.2, 0.2])

    extent = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())
    cam_axis = extent * 0.002
    tag_axis = extent * 0.005

    # Camera frame at origin
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cam_axis, origin=[0, 0, 0])

    # Tag frame at detected pose
    tag_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=tag_axis, origin=t.tolist())
    tag_frame.rotate(R, center=t.tolist())

    o3d.visualization.draw_geometries([pcd, mesh_cam, cam_frame, tag_frame])


def main():
    R, t = detect_tag_pose(IMAGE_PATH)
    pcd = load_point_cloud(PLY_PATH)
    cad = load_cad_mesh(CAD_PATH)
    mesh_cam = transform_cad_mesh(cad, R, t)
    visualize(pcd, mesh_cam, R, t)


if __name__ == "__main__":
    main()
