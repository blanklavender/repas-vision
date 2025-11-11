"""
Aligns a postoperative (post-op) mesh to its preoperative (pre-op) version using
coarse global registration (RANSAC) followed by fine alignment (ICP). 

Workflow:
1. Runs automatic registration between pre-op and post-op meshes.
2. Displays the result for visual approval.
3. If 'No' → restarts registration from scratch.
4. If 'Yes' → refines alignment using the top portion of the meshes (defined by --top_frac)
   to improve precision, visualizes that region, applies the refined transform to the entire mesh,
   shows the final alignment, and saves the registered mesh with '_registered' suffix.

Arguments:
--preop      Path to pre-op STL file (default: 130_preop.stl)
--postop     Path to post-op STL file (default: 130_postop.stl)
--seed       Optional random seed for reproducibility
--top_frac   Fraction of total Y-extent used for the second-round ICP refinement (default: 0.5)

Example:
python icp_cad_model.py --preop ./exports/cropped_camframe_cleaned.stl --postop ./exports/cad_placed.stl
"""
from __future__ import annotations
import open3d as o3d
import numpy as np
import argparse, os, sys, random, subprocess, shutil, platform


# ---------------------------
# Mesh / registration helpers
# ---------------------------
def read_mesh(path):
    m = o3d.io.read_triangle_mesh(path)
    if m is None or not m.has_triangles():
        raise ValueError(f"Invalid mesh: {path}")
    m.compute_vertex_normals()
    return m

def mesh_to_pcd(mesh, n_points=1000000):
    tri = np.asarray(mesh.triangles)
    n = max(1000000, min(n_points, int(max(1, len(tri)) * 30)))
    pcd = mesh.sample_points_uniformly(number_of_points=n)
    return pcd

def preprocess_pcd(pcd, voxel):
    p = pcd.voxel_down_sample(voxel)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2.0, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        p, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5.0, max_nn=100)
    )
    return p, f

def auto_voxel_from_meshes(ma, mb, frac=0.02):
    # Combine AABBs via min/max bounds (Open3D AABB has no .extend())
    a = ma.get_axis_aligned_bounding_box()
    b = mb.get_axis_aligned_bounding_box()
    minb = np.minimum(a.get_min_bound(), b.get_min_bound())
    maxb = np.maximum(a.get_max_bound(), b.get_max_bound())
    extent = maxb - minb
    diag = float(np.linalg.norm(extent))
    return max(diag * frac, 1e-3)

def align_postop_to_preop(mesh_pre, mesh_post):
    vs = auto_voxel_from_meshes(mesh_pre, mesh_post, frac=0.02)

    # Dense samples for ICP
    tgt = mesh_to_pcd(mesh_pre)
    src = mesh_to_pcd(mesh_post)

    # Ensure normals for ICP (point-to-plane needs target normals)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs*2.0, max_nn=30))
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs*2.0, max_nn=30))

    # Downsample + FPFH for global registration
    tgt_d, tgt_f = preprocess_pcd(tgt, vs)
    src_d, src_f = preprocess_pcd(src, vs)

    checker = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(vs*2.5),
    ]
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_d, tgt_d, src_f, tgt_f, True,
        vs*2.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, checker,
        o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 1000),
    )

    # Refine with point-to-plane ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        src, tgt, vs*1.5, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )

    return result_ransac, result_icp, vs

def compute_y_threshold_top_fraction(pcd, top_frac=0.5):
    """
    Define the Y threshold using full extent:
        y_thresh = y_max - top_frac * (y_max - y_min)
    Keep points with y >= y_thresh → top `top_frac` of the bone along Y.
    """
    P = np.asarray(pcd.points)
    y = P[:, 1]
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    f = float(np.clip(top_frac, 0.0, 1.0))
    return y_max - f * (y_max - y_min)

def crop_pcd_by_y(pcd, y_thresh):
    P = np.asarray(pcd.points)
    idx = np.where(P[:, 1] >= y_thresh)[0]
    return pcd.select_by_index(idx, invert=False)

# ---------------------------
# Approval dialogs (robust)
# ---------------------------

def ask_approval_pyqt5(question: str) -> bool | None:
    """Try a PyQt5 QMessageBox. Return True/False on success, None if PyQt5 unavailable."""
    try:
        from PyQt5 import QtWidgets
        from PyQt5.QtWidgets import QMessageBox
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        msg = QMessageBox()
        msg.setWindowTitle("Approve registration")
        msg.setText(question)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        ret = msg.exec_()
        return ret == QMessageBox.Yes
    except Exception:
        return None

def ask_approval_applescript(question: str) -> bool | None:
    """Use macOS AppleScript dialog. Return True/False on success, None if not macOS or osascript missing/fails."""
    if platform.system() != "Darwin":
        return None
    if shutil.which("osascript") is None:
        return None
    q = question.replace('"', r'\"')
    script = f'display dialog "{q}" buttons {{"No","Yes"}} default button "Yes" with icon note'
    try:
        res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if res.returncode == 0 and "button returned:Yes" in res.stdout:
            return True
        elif res.returncode == 0 and "button returned:No" in res.stdout:
            return False
        else:
            return None
    except Exception:
        return None

def ask_approval_console(question: str) -> bool:
    """Final fallback: console prompt."""
    while True:
        resp = input(f"{question} [y/n]: ").strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")

def ask_approval(question: str) -> bool:
    """Try PyQt5 -> AppleScript -> console."""
    ans = ask_approval_pyqt5(question)
    if isinstance(ans, bool):
        return ans
    ans = ask_approval_applescript(question)
    if isinstance(ans, bool):
        return ans
    return ask_approval_console(question)

# ---------------------------
# Main loop with approval + top-fraction refinement on Yes
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Align postop to preop, visualize, approve, refine on top-fraction, and save.")
    ap.add_argument("--preop", type=str, default="130_preop.stl", help="Path to preop STL.")
    ap.add_argument("--postop", type=str, default="130_postop.stl", help="Path to postop STL.")
    ap.add_argument("--seed", type=int, default=None, help="Base random seed (each attempt adds the attempt index).")
    ap.add_argument(
        "--top_frac",
        type=float,
        default=0.5,
        help="Top fraction of full Y-extent to use for the second-round ICP. "
             "0.7 → top 70%% (y >= y_max - 0.7*(y_max - y_min)); 0 → near y_max; 1 → whole.",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.preop) or not os.path.isfile(args.postop):
        print("Missing input file(s).")
        print(f"preop:  {args.preop}  exists={os.path.isfile(args.preop)}")
        print(f"postop: {args.postop} exists={os.path.isfile(args.postop)}")
        sys.exit(1)

    pre_mesh_orig = read_mesh(args.preop)
    post_mesh_orig = read_mesh(args.postop)

    attempt = 0
    while True:
        # Reseed each attempt to re-run from scratch (changes RANSAC sampling, etc.)
        base_seed = args.seed if args.seed is not None else random.randrange(1, 10**9)
        run_seed = (base_seed + attempt) % (2**32 - 1)
        np.random.seed(run_seed)
        random.seed(run_seed)

        # Work on fresh copies each attempt
        pre = o3d.geometry.TriangleMesh(pre_mesh_orig)
        post = o3d.geometry.TriangleMesh(post_mesh_orig)

        # Run alignment
        ransac, icp, vs = align_postop_to_preop(pre, post)
        T = icp.transformation if icp.fitness > 0 else ransac.transformation

        # Transform postop onto preop
        post_aligned = o3d.geometry.TriangleMesh(post)
        post_aligned.transform(T)

        # Color and visualize (blocking until the window is closed)
        pre.paint_uniform_color([0.10, 0.70, 0.20])          # green = preop
        post_aligned.paint_uniform_color([0.10, 0.35, 0.95]) # blue  = postop

        print(f"[Attempt {attempt}] seed={run_seed}")
        print("Global (RANSAC): fitness=%.4f, rmse=%.4f" % (ransac.fitness, ransac.inlier_rmse))
        print("Refined (ICP):  fitness=%.4f, rmse=%.4f" % (icp.fitness, icp.inlier_rmse))
        print("Transform (postop -> preop) after first round T:\n", T)

        o3d.visualization.draw_geometries(
            [pre, post_aligned],
            window_name=f"Attempt {attempt} — {os.path.basename(args.preop)} (green) vs {os.path.basename(args.postop)} (blue)"
        )

        # Pop-up approval after the viewer closes
        approved = ask_approval(
            "Do you approve the registration?\n\nYes: refine on top-fraction & save\nNo: re-run registration"
        )
        if not approved:
            print("Not approved. Re-running registration from scratch...")
            attempt += 1
            continue

        # ---------------------------
        # YES: refine on top-fraction only, then apply to whole mesh
        # ---------------------------
        print(f"Approved. Refining on top fraction {args.top_frac:.3f} of full Y-extent before saving...")

        # Build dense clouds for refinement
        pcd_pre_full = mesh_to_pcd(pre)
        pcd_post_full = mesh_to_pcd(post_aligned)  # already in pre frame

        # Compute threshold from pre-op (single reference threshold for both)
        y_thresh = compute_y_threshold_top_fraction(pcd_pre_full, args.top_frac)

        # Crop both to top region in Y
        pcd_pre_top = crop_pcd_by_y(pcd_pre_full, y_thresh)
        pcd_post_top = crop_pcd_by_y(pcd_post_full, y_thresh)

        # Safety: if the crop is too small, fall back to full clouds
        if len(pcd_pre_top.points) < 1000 or len(pcd_post_top.points) < 1000:
            print("Top-region crop too small; falling back to full clouds for refinement.")
            pcd_pre_top = pcd_pre_full
            pcd_post_top = pcd_post_full

        # Normals for top-region ICP (point-to-plane)
        pcd_pre_top.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs*1.5, max_nn=50))
        pcd_post_top.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs*1.5, max_nn=50))

        # Tighter ICP on cropped region (post_aligned is already in target frame; init = Identity)
        delta_icp = o3d.pipelines.registration.registration_icp(
            pcd_post_top, pcd_pre_top, vs*0.8, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
        )

        print("Top-region refinement: fitness=%.4f, rmse=%.4f" % (delta_icp.fitness, delta_icp.inlier_rmse))

        # Visualize cropped-region overlay after refinement
        pcd_post_top_refined = o3d.geometry.PointCloud(pcd_post_top)
        pcd_post_top_refined.transform(delta_icp.transformation)
        pcd_pre_top.paint_uniform_color([0.10, 0.70, 0.20])
        pcd_post_top_refined.paint_uniform_color([0.10, 0.35, 0.95])
        o3d.visualization.draw_geometries(
            [pcd_pre_top, pcd_post_top_refined],
            window_name="Refined (top-region only)"
        )

        # Compose final transform for the whole mesh: T2 = delta * T
        T2 = delta_icp.transformation @ T
        print("FINAL Transform (postop -> preop) after second-round refinement T2:\n", T2)

        # Apply to the entire original postop mesh
        post_final = o3d.geometry.TriangleMesh(post_mesh_orig)
        post_final.transform(T2)

        # Final full-scene visualization
        pre_final = o3d.geometry.TriangleMesh(pre_mesh_orig)
        pre_final.paint_uniform_color([0.10, 0.70, 0.20])
        post_final.paint_uniform_color([0.10, 0.35, 0.95])
        o3d.visualization.draw_geometries(
            [pre_final, post_final],
            window_name="Final full alignment after top-region refinement"
        )

        # Save the registered postop mesh (final transform)
        base, ext = os.path.splitext(args.postop)
        output_path = f"{base}_registered{ext}"
        o3d.io.write_triangle_mesh(output_path, post_final)
        print(f"Saved registered postop mesh to: {output_path}")
        print("All good ✅")
        break

if __name__ == "__main__":
    main()