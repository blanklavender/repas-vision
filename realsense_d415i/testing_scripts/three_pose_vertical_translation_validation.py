import cv2
import numpy as np
from pupil_apriltags import Detector

# =========================
# Config
# =========================

# aligned image paths
# IMAGE_PATHS = [
#     r"../02_ply_generation\ply_generation_scripts\aligned_outputs\pose 1\rgb_20250808_142303.png",
#     r"../02_ply_generation\ply_generation_scripts\aligned_outputs\pose 2\rgb_20250808_142806.png",
#     r"../02_ply_generation\ply_generation_scripts\aligned_outputs\pose 3\rgb_20250808_143022.png",
# ]

# not aligned image paths
IMAGE_PATHS = [
    r"../02_ply_generation/ply_generation_scripts/not_aligned_outputs/pose 1/rgb_20250808_142429.png",
    r"../02_ply_generation/ply_generation_scripts/not_aligned_outputs/pose 2/rgb_20250808_142858.png",
    r"../02_ply_generation/ply_generation_scripts/not_aligned_outputs/pose 3/rgb_20250808_143101.png",
]

# If you want to force a specific tag ID to be used in each image, set this.
# Otherwise the highest-decision-margin tag is used.
TARGET_TAG_ID = None  # e.g., 0  or  None

# updated camera intrinsics & distortion for 1280x720 @ 15fps from fetch_factory_intrinsic.py
K = np.array([
    [912.35034180,   0.0,            628.78363037],
    [0.0,            911.77630615,   348.97726440],
    [0.0,              0.0,            1.0        ]
], dtype=np.float32)

dist_coeffs = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0
], dtype=np.float32)

TAG_SIZE = 0.0303     # updated tag size in meters
MARGIN_THRESH = 10.0 # only accept tags with decision_margin ≥ this

# =========================
# Helpers
# =========================
def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE to even out illumination."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def detect_with_params(gray_img: np.ndarray, quad_decimate: float, quad_sigma: float):
    det = Detector(
        families='tag36h11',
        nthreads=4,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    return det.detect(gray_img)

def choose_tag(detections, target_id):
    """Pick a tag by ID if provided, else highest decision margin."""
    if not detections:
        return None
    if target_id is not None:
        candidates = [t for t in detections if t.tag_id == target_id]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t.decision_margin)
    return max(detections, key=lambda t: t.decision_margin)

def solve_pose_from_tag(tag, K, dist_coeffs, tag_size_m):
    """
    tag.corners: np.ndarray shape (4,2), order is TL, TR, BR, BL in pixels.
    We define the 3D tag-frame points to match that ordering:
    (-h,-h,0)=TL, (h,-h,0)=TR, (h,h,0)=BR, (-h,h,0)=BL
    """
    h = tag_size_m / 2.0
    obj_pts = np.array(
        [[-h, -h, 0],
         [ h, -h, 0],
         [ h,  h, 0],
         [-h,  h, 0]], dtype=np.float32
    )
    img_pts = tag.corners.astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return bool(success), rvec, tvec

def detect_pose_for_image(image_path: str, target_id=None):
    """Return (success, tag_id, margin, rvec, tvec) for the selected tag in one image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variants = {
        "orig": gray,
        "clahe": enhance_contrast(gray)
    }

    selected_tag = None
    selected_margin = None

    for _, g in variants.items():
        # First pass: faster
        tags = detect_with_params(g, quad_decimate=1.0, quad_sigma=0.0)
        good = [t for t in tags if t.decision_margin >= MARGIN_THRESH]
        # Second pass if needed: more thorough
        if not good:
            tags = detect_with_params(g, quad_decimate=0.5, quad_sigma=1.0)
            good = [t for t in tags if t.decision_margin >= MARGIN_THRESH]

        tag = choose_tag(good, target_id)
        if tag is not None:
            selected_tag = tag
            selected_margin = tag.decision_margin
            break

    if selected_tag is None:
        return False, None, None, None, None

    ok, rvec, tvec = solve_pose_from_tag(selected_tag, K, dist_coeffs, TAG_SIZE)
    if not ok:
        return False, selected_tag.tag_id, selected_margin, None, None
    return True, selected_tag.tag_id, selected_margin, rvec, tvec

def log_pose(idx, tag_id, margin, rvec, tvec):
    t = tvec.reshape(-1)
    r = rvec.reshape(-1)
    print(f"[Pose {idx}] tag_id={tag_id} margin={margin:.2f}")
    print(f"         rvec = [{r[0]:.6f}, {r[1]:.6f}, {r[2]:.6f}]  (Rodrigues, rad)")
    print(f"         tvec = [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]  (meters)\n")

def diff_translation(name_a, t_a, name_b, t_b):
    """Compute t_b - t_a and return (delta, norm, dy)."""
    da = t_b.reshape(3) - t_a.reshape(3)
    dist = float(np.linalg.norm(da))
    dy = float(da[1])
    print(f"[Δ {name_a} → {name_b}] Δt = [{da[0]:.6f}, {da[1]:.6f}, {da[2]:.6f}] m | ‖Δt‖ = {dist:.6f} m")
    print(f"                 Y-axis translation (m): {dy:.6f}\n")
    return da, dist, dy

# =========================
# Main
# =========================
def main():
    results = []  # list of tuples: (ok, tag_id, margin, rvec, tvec)

    for i, p in enumerate(IMAGE_PATHS, start=1):
        ok, tag_id, margin, rvec, tvec = detect_pose_for_image(p, TARGET_TAG_ID)
        if not ok:
            print(f"[Pose {i}] No acceptable tag pose found in: {p}")
            results.append((False, None, None, None, None))
            continue
        log_pose(i, tag_id, margin, rvec, tvec)
        results.append((True, tag_id, margin, rvec, tvec))

    # Need all three to compute differences
    if not all(r[0] for r in results):
        print("One or more poses missing; cannot compute pose deltas.")
        return

    # Unpack for clarity
    _, _, _, rvec1, tvec1 = results[0]
    _, _, _, rvec2, tvec2 = results[1]
    _, _, _, rvec3, tvec3 = results[2]

    print("=== Translation Differences (t_j - t_i) ===")
    diff_translation("Pose 1", tvec1, "Pose 2", tvec2)
    diff_translation("Pose 2", tvec2, "Pose 3", tvec3)
    diff_translation("Pose 1", tvec1, "Pose 3", tvec3)

if __name__ == "__main__":
    main()
