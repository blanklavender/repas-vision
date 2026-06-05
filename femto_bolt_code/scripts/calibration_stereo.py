import cv2
import numpy as np
import glob

# --- Board geometry ---
PATTERN = (19, 19)            # inner corners (cols, rows)
SQUARE_MM = 12.7            # measure your printed squares with calipers!

# Build the 3D object points for one board pose: (N, 3) in mm
objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2)
objp *= SQUARE_MM

# --- Load synchronized pairs ---
color_files = sorted(glob.glob("captures/color_*.png"))
ir_files    = sorted(glob.glob("captures/ir_*.png"))
assert len(color_files) == len(ir_files)

obj_points = []
color_points = []
ir_points = []

# Sub-pixel refinement criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

color_size = None
ir_size = None

for cf, irf in zip(color_files, ir_files):
    color = cv2.imread(cf, cv2.IMREAD_GRAYSCALE)
    ir    = cv2.imread(irf, cv2.IMREAD_GRAYSCALE)
    # IR from Femto Bolt is 16-bit; normalize to 8-bit for corner detection
    if ir.dtype != np.uint8:
        ir = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    color_size = color.shape[::-1]   # (W, H)
    ir_size    = ir.shape[::-1]

    found_c, corners_c = cv2.findChessboardCorners(color, PATTERN,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    found_i, corners_i = cv2.findChessboardCorners(ir, PATTERN,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if not (found_c and found_i):
        print(f"  skip {cf}: color={found_c} ir={found_i}")
        continue

    cv2.cornerSubPix(color, corners_c, (11, 11), (-1, -1), criteria)
    cv2.cornerSubPix(ir,    corners_i, (5, 5),  (-1, -1), criteria)

    obj_points.append(objp)
    color_points.append(corners_c)
    ir_points.append(corners_i)

print(f"Using {len(obj_points)} pose pairs")

# --- Step 1: calibrate each camera independently ---
rms_c, K_c, dist_c, _, _ = cv2.calibrateCamera(
    obj_points, color_points, color_size, None, None)
rms_i, K_i, dist_i, _, _ = cv2.calibrateCamera(
    obj_points, ir_points, ir_size, None, None)

print(f"Color intrinsic RMS: {rms_c:.4f} px")
print(f"IR    intrinsic RMS: {rms_i:.4f} px")

# --- Step 2: stereo calibrate ---
# Fix the per-camera intrinsics we just solved; only solve for R, t between them.
flags = cv2.CALIB_FIX_INTRINSIC
rms_stereo, K_c, dist_c, K_i, dist_i, R, T, E, F = cv2.stereoCalibrate(
    obj_points, color_points, ir_points,
    K_c, dist_c, K_i, dist_i,
    color_size,                # arbitrary, not used when intrinsics fixed
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=flags)

print(f"Stereo RMS: {rms_stereo:.4f} px")
print("\nColor K:\n", K_c)
print("\nIR K:\n", K_i)
print("\nR (IR -> Color):\n", R)
print("T (IR -> Color, mm):", T.ravel())