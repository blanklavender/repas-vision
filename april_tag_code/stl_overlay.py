import numpy as np
import cv2
import pyrealsense2 as rs
import trimesh
from trimesh.transformations import rotation_matrix
import pyrender
from pupil_apriltags import Detector
import time

# -----------------------------------------------------------------------------
# 1) PRE‑RENDER YOUR STL TO A TRANSPARENT SPRITE
# -----------------------------------------------------------------------------
SPRITE_SIZE    = 200                # px
MESH_PATH      = '..\cad_model\StructureResvrLightBox_AprilTagOrigin.stl'
PADDING_FACTOR = 1.1
YFOV           = np.pi / 3.0       # 60°

# load, center & normalize mesh
mesh = trimesh.load(MESH_PATH)
mesh.apply_translation(-mesh.centroid)
mesh.apply_scale(1 / mesh.extents.max())

# -------------------------------------------------------------------------
# ** APPLY THE TWO ROTATIONS **
# -------------------------------------------------------------------------
#  a) 270° about X (pitch forward)
R_x_270 = rotation_matrix(np.deg2rad(270), [1, 0, 0])
mesh.apply_transform(R_x_270)

#  b)  90° about Y (yaw right)
R_y_90  = rotation_matrix(np.deg2rad(90), [0, 1, 0])
mesh.apply_transform(R_y_90)
# -------------------------------------------------------------------------

# compute camera distance so mesh fits in view
radius       = 0.5
cam_distance = (radius / np.tan(YFOV / 2)) * PADDING_FACTOR

# set up offscreen renderer
renderer = pyrender.OffscreenRenderer(SPRITE_SIZE, SPRITE_SIZE)
scene    = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[0.3]*3)

scene.add(pyrender.Mesh.from_trimesh(mesh), pose=np.eye(4))
cam = pyrender.PerspectiveCamera(yfov=YFOV)
cam_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, cam_distance],
    [0, 0, 0, 1]
])
scene.add(cam, pose=cam_pose)
scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
          pose=np.eye(4))

sprite_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
renderer.delete()

sprite_h, sprite_w, _ = sprite_rgba.shape
sprite_alpha = sprite_rgba[:, :, 3] / 255.0
sprite_rgb   = sprite_rgba[:, :, :3]

# -----------------------------------------------------------------------------
# 2) START REALSENSE & APRILTAG DETECTOR
# -----------------------------------------------------------------------------
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)
# pipeline.start(config)

# 1) define your real‐world tag size (in meters)
TAG_SIZE = 0.05

# 2) grab the RealSense color stream intrinsics
profile       = pipeline.start(config)
color_profile = profile.get_stream(rs.stream.color) \
                       .as_video_stream_profile()
intr          = color_profile.get_intrinsics()

# 3) build your camera matrix
K = np.array([
    [intr.fx,  0,       intr.ppx],
    [0,        intr.fy, intr.ppy],
    [0,        0,       1      ]
])

# 4) assume zero lens distortion
dist_coeffs = np.zeros(4)

detector = Detector(families='tag36h11', nthreads=4,
                    quad_decimate=1.0, refine_edges=1)
align    = rs.align(rs.stream.color)

print("Stabilizing camera...")
time.sleep(5)
print("Starting main loop...")

try:
    while True:
        frames        = pipeline.wait_for_frames()
        aligned       = align.process(frames)
        depth_frame   = aligned.get_depth_frame()
        color_frame   = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray        = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        tags = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[intr.fx, intr.fy, intr.ppx, intr.ppy],
            tag_size=TAG_SIZE
        )


        if tags:
            tag = tags[0]
            cx, cy = tag.center

            # draw tag outline & center
            cv2.polylines(color_image, [tag.corners.astype(int)], True,
                          (0,255,0), 2)
            cv2.circle(color_image, (int(cx),int(cy)), 5, (0,0,255), -1)

            # ——— draw local XYZ axes at the tag center ———

            axis_len = TAG_SIZE * 0.5
            # 3D points: origin, X, Y, Z (Z negative to point “out” of tag)
            pts_3d = np.float32([
                [0,         0,          0        ],
                [axis_len,  0,          0        ],  # X axis
                [0,         axis_len,   0        ],  # Y axis
                [0,         0,         -axis_len ]   # Z axis
            ]).reshape(-1,3)

            # rotation vector + translation vector from the tag detector
            rvec, _ = cv2.Rodrigues(tag.pose_R)
            tvec = tag.pose_t.reshape(3,1)

            # project into pixel coordinates
            imgpts, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist_coeffs)
            imgpts = imgpts.reshape(-1,2).astype(int)

            origin = tuple(imgpts[0])
            cv2.line(color_image, origin, tuple(imgpts[1]), (0,0,255), 2)  # X = red
            cv2.line(color_image, origin, tuple(imgpts[2]), (0,255,0), 2)  # Y = green
            cv2.line(color_image, origin, tuple(imgpts[3]), (255,0,0), 2)  # Z = blue


            # -----------------------------------------------------------------
            # ** BOTTOM‑LEFT OF SPRITE → TAG CENTER **
            # sprite pixel (u=0, v=sprite_h-1) ⇔ image pixel (cx, cy)
            # so top-left corner of sprite goes to:
            #    x0 = cx - 0
            #    y0 = cy - (sprite_h - 1)
            # -----------------------------------------------------------------
            x0 = int(cx)
            y0 = int(cy - (sprite_h - 1))

            # clamp so we stay in the image bounds
            H, W = color_image.shape[:2]
            x0 = max(0, min(W - sprite_w, x0))
            y0 = max(0, min(H - sprite_h, y0))

            # alpha‑blend
            for c in range(3):
                roi = color_image[y0:y0+sprite_h, x0:x0+sprite_w, c]
                color_image[y0:y0+sprite_h, x0:x0+sprite_w, c] = (
                    sprite_alpha * sprite_rgb[:,:,c] +
                    (1 - sprite_alpha) * roi
                )

        cv2.imshow('AR Overlay', color_image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
