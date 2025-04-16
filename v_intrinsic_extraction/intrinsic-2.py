import numpy as np
import cv2
import trimesh
import plotly.graph_objects as go
import os

# --- STEP 1: Load and process the image ---
image_path = "/Users/valentinacampanelli/Documents/HRVIP/AruCo Tag/aruco_image5.jpg"
assert os.path.exists(image_path), "Image file not found"

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- STEP 2: Detect AprilTag (tag36h11) ---
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
corners, ids, _ = detector.detectMarkers(gray)

# --- STEP 3: Define camera intrinsics (iPhone 12, portrait 3024x4032) ---
camera_matrix = np.array([[2900, 0, 1512],
                          [0, 2900, 2016],
                          [0,    0,    1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))
marker_length = 0.05  # 5 cm

# --- STEP 4: Estimate pose of tag ---
if ids is not None:
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
    rvec = rvecs[0]
    tvec = tvecs[0]
else:
    raise ValueError("No AprilTags detected.")

# --- STEP 5: Convert to transformation matrix ---
rotation_matrix, _ = cv2.Rodrigues(rvec)
transform = np.eye(4)
transform[:3, :3] = rotation_matrix
transform[:3, 3] = tvec.flatten()
print("Transformation matrix (AprilTag → Camera CSYS):\n", transform)

# --- STEP 6: Load STL, scale from cm to m, and apply transform ---
mesh = trimesh.load("/Users/valentinacampanelli/Documents/HRVIP/Hydroponic Garden/CAD_Hydrophonic System/StructureResvrLightBox.stl")
mesh.apply_scale(0.01)  # Convert cm → m
mesh.apply_transform(transform)

# --- STEP 7: Extract vertices/faces for Plotly ---
vertices = mesh.vertices
faces = mesh.faces
x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

mesh_plot = go.Mesh3d(
    x=x, y=y, z=z,
    i=i, j=j, k=k,
    color='lightgray',
    opacity=0.5,
    name='STL Mesh'
)

# --- STEP 8: Create CSYS for AprilTag and Camera ---
origin_tag = tvec.flatten()
origin_cam = np.array([0, 0, 0])
L = 0.05 * np.linalg.norm(mesh.extents)  # Axis length = 5% of mesh size
print(f"STL size (meters): {mesh.extents}, Axis length: {L:.2f} m")

# AprilTag CSYS (solid)
tag_axes = [
    go.Scatter3d(x=[origin_tag[0], origin_tag[0] + rotation_matrix[0, 0]*L],
                 y=[origin_tag[1], origin_tag[1] + rotation_matrix[1, 0]*L],
                 z=[origin_tag[2], origin_tag[2] + rotation_matrix[2, 0]*L],
                 mode='lines', line=dict(color='red', width=10), name='Tag X'),

    go.Scatter3d(x=[origin_tag[0], origin_tag[0] + rotation_matrix[0, 1]*L],
                 y=[origin_tag[1], origin_tag[1] + rotation_matrix[1, 1]*L],
                 z=[origin_tag[2], origin_tag[2] + rotation_matrix[2, 1]*L],
                 mode='lines', line=dict(color='green', width=10), name='Tag Y'),

    go.Scatter3d(x=[origin_tag[0], origin_tag[0] + rotation_matrix[0, 2]*L],
                 y=[origin_tag[1], origin_tag[1] + rotation_matrix[1, 2]*L],
                 z=[origin_tag[2], origin_tag[2] + rotation_matrix[2, 2]*L],
                 mode='lines', line=dict(color='blue', width=10), name='Tag Z')
]

# Camera CSYS at (0,0,0) (dashed)
cam_axes = [
    go.Scatter3d(x=[0, L], y=[0, 0], z=[0, 0],
                 mode='lines', line=dict(color='red', width=6, dash='dash'), name='Camera X'),
    go.Scatter3d(x=[0, 0], y=[0, L], z=[0, 0],
                 mode='lines', line=dict(color='green', width=6, dash='dash'), name='Camera Y'),
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, L],
                 mode='lines', line=dict(color='blue', width=6, dash='dash'), name='Camera Z')
]

# --- STEP 9: Show everything ---
fig = go.Figure(data=[mesh_plot] + tag_axes + cam_axes)
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ),
    width=1000,
    height=800,
    title='STL (rescaled from cm) + AprilTag CSYS + Camera CSYS'
)
fig.show()