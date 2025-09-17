import cv2
import numpy as np
from pupil_apriltags import Detector

# === Camera intrinsics & distortion ===
# Checkerboard calibration values
K = np.array([
    [605.2845686,   0.0,        309.95995203],
    [0.0,           605.44233933, 229.79166863],
    [0.0,             0.0,          1.0     ]
], dtype=np.float32)

dist_coeffs = np.array([
    0.04344582,  0.32076285, -0.00060687, -0.0004814, -1.40593456
], dtype=np.float32)

TAG_SIZE = 0.0303   # meters
MARGIN_THRESH = 10.0  # only accept tags with decision_margin ≥ this

# === Utility: apply CLAHE to even out illumination ===
def enhance_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

# === Attempt detection with varied parameters ===
def detect_with_params(gray_img, quad_decimate, quad_sigma):
    detector = Detector(
        families='tag36h11',
        nthreads=4,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    return detector.detect(gray_img)

# === Load and prepare image ===
image_path = "femto_bolt_code\pyorbbecsdk\examples\captures\color_20250815_133503.png"
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at path: {image_path}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# enhance and stack original + enhanced for detection
variants = {
    "orig": gray,
    "clahe": enhance_contrast(gray),
}

# Try multiple settings until we get high-margin tags
all_detections = []
for name, g in variants.items():
    # first try faster (decimate=1.0, no blur)
    tags = detect_with_params(g, quad_decimate=1.0, quad_sigma=0.0)
    # if too few or too low-margin, try slower but more thorough
    if not tags or max(t.decision_margin for t in tags) < MARGIN_THRESH:
        tags = detect_with_params(g, quad_decimate=0.5, quad_sigma=1.0)
    # collect only those above margin threshold
    good = [t for t in tags if t.decision_margin >= MARGIN_THRESH]
    if good:
        all_detections = good
        break

# annotate each good detection
for tag in all_detections:
    corners = tag.corners.astype(np.float32)
    cv2.polylines(img, [corners.astype(int)], True, (0,255,0), 2)
    cx, cy = map(int, tag.center)
    cv2.circle(img, (cx, cy), 5, (0,0,255), -1)

    # prepare 3D points in tag frame
    h = TAG_SIZE / 2.0
    obj_pts = np.array([
        [-h, -h, 0],
        [ h, -h, 0],
        [ h,  h, 0],
        [-h,  h, 0],
    ], dtype=np.float32)

    # solve for pose
    success, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, dist_coeffs,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        continue

    # draw axes
    cv2.drawFrameAxes(img, K, dist_coeffs, rvec, tvec, TAG_SIZE * 0.5, 2)

    # annotate position
    t = tvec.ravel()
    coord_text = f"X:{t[0]:.3f}m Y:{t[1]:.3f}m Z:{t[2]:.3f}m"
    cv2.putText(img, coord_text, (cx + 10, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(img, coord_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    print(f"Tag {tag.tag_id} margin={tag.decision_margin:.1f} → "
          f"rvec={rvec.ravel()}, tvec={t}")

# show results
cv2.imshow("Enhanced AprilTag Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
