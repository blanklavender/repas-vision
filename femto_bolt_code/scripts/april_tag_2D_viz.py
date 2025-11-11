#!/usr/bin/env python3
"""
Visualizes AprilTag corners in 2D pixel space showing:
1. Detected corners (from image)
2. Reprojected corners (from PnP solution back to image)
This helps understand reprojection quality visually.

This modified version automatically saves figures to ./viz_outputs/.
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2

# ------------ output directory ------------
OUT_DIR = Path("./viz_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def rotation_matrix_to_euler_angles(R):
    """
    Convert rotation matrix to Euler angles (in degrees).
    Returns: (roll, pitch, yaw) in degrees
    Using ZYX convention (yaw-pitch-roll)
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # roll
        y = np.arctan2(-R[2, 0], sy)      # pitch
        z = np.arctan2(R[1, 0], R[0, 0])  # yaw
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.degrees([x, y, z])


def visualize_reprojection(tag_id, detected_corners, obj_pts, rvec, tvec, K, dist, R, t):
    """
    Create a 2D visualization of detected vs reprojected corners and SAVE it.

    Args:
        tag_id: Tag ID number
        detected_corners: (4,2) array of detected pixel coordinates
        obj_pts: (4,3) array of 3D object points used in PnP
        rvec, tvec: PnP solution
        K: Camera intrinsic matrix
        dist: Distortion coefficients
        R: Rotation matrix
        t: Translation vector
    """
    # Compute reprojected corners
    reprojected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    reprojected = reprojected.reshape(-1, 2)
    
    # Compute reprojection errors
    errors = np.linalg.norm(detected_corners - reprojected, axis=1)
    mean_error = np.mean(errors)
    
    # Convert rotation to euler angles
    roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
    
    # Print 6-DOF pose
    print(f"\n{'='*60}")
    print(f"Tag {tag_id} - 6-DOF Pose Estimation")
    print(f"{'='*60}")
    print(f"Translation (meters):")
    print(f"  X: {t[0]:8.4f} m")
    print(f"  Y: {t[1]:8.4f} m")
    print(f"  Z: {t[2]:8.4f} m")
    print(f"\nRotation (Euler angles, degrees):")
    print(f"  Roll  (X): {roll:8.2f}°")
    print(f"  Pitch (Y): {pitch:8.2f}°")
    print(f"  Yaw   (Z): {yaw:8.2f}°")
    print(f"\nRotation Matrix:")
    print(R)
    print(f"\nReprojection Error:")
    print(f"  Per corner: {errors}")
    print(f"  Mean: {mean_error:.2f} pixels")
    print(f"{'='*60}\n")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot detected corners
    det_poly = Polygon(detected_corners, fill=False, edgecolor='blue', 
                       linewidth=2, label='Detected')
    ax.add_patch(det_poly)
    ax.plot(detected_corners[:, 0], detected_corners[:, 1], 
            'bo', markersize=10, label='Detected corners')
    
    # Plot reprojected corners
    repr_poly = Polygon(reprojected, fill=False, edgecolor='red', 
                        linewidth=2, linestyle='--', label='Reprojected')
    ax.add_patch(repr_poly)
    ax.plot(reprojected[:, 0], reprojected[:, 1], 
            'r^', markersize=10, label='Reprojected corners')
    
    # Draw error vectors & labels
    for i in range(4):
        ax.arrow(detected_corners[i, 0], detected_corners[i, 1],
                 reprojected[i, 0] - detected_corners[i, 0],
                 reprojected[i, 1] - detected_corners[i, 1],
                 head_width=5, head_length=5, fc='green', ec='green',
                 alpha=0.5, linewidth=1.5)
        ax.text(detected_corners[i, 0], detected_corners[i, 1] - 15,
                f'C{i}\n{errors[i]:.1f}px',
                ha='center', va='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Centers
    det_center = detected_corners.mean(axis=0)
    repr_center = reprojected.mean(axis=0)
    ax.plot(det_center[0], det_center[1], 'bs', markersize=12, label='Detected center')
    ax.plot(repr_center[0], repr_center[1], 'r*', markersize=15, label='Reprojected center')
    
    # Axis cosmetics
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title(f'Tag {tag_id} - Reprojection Visualization\n'
                 f'Mean Error: {mean_error:.2f} px | '
                 f'Pose: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()

    # ---- SAVE FIG ----
    out_path = OUT_DIR / f"tag_{tag_id}_reprojection.png"
    fig.savefig(out_path, dpi=200)
    print(f"[SAVED] {out_path}")

    return fig


def visualize_depth_validated_positions(tag_poses_with_depth, K, fx, fy, cx, cy):
    """
    Print and visualize depth-validated 3D positions, and SAVE the figure.
    
    Args:
        tag_poses_with_depth: dict {tag_id: (R, t, P_depth, u, v, Zc)}
        K: Camera intrinsic matrix
        fx, fy, cx, cy: Camera parameters
    """
    print(f"\n{'='*80}")
    print(f"Depth-Validated 3D Positions (from median depth)")
    print(f"{'='*80}")
    
    for tag_id, data in tag_poses_with_depth.items():
        R, t_pnp, P_depth, u, v, Zc = data
        
        print(f"\nTag {tag_id}:")
        print(f"  Projected pixel location: u={u:.1f}, v={v:.1f}")
        print(f"  Median depth at location: Zc={Zc:.4f} m")
        print(f"  Depth-validated 3D position (X, Y, Z): [{P_depth[0]:.4f}, {P_depth[1]:.4f}, {P_depth[2]:.4f}] m")
        print(f"  PnP estimated translation  (X, Y, Z): [{t_pnp[0]:.4f}, {t_pnp[1]:.4f}, {t_pnp[2]:.4f}] m")
        
        diff = P_depth - t_pnp
        diff_norm = np.linalg.norm(diff)
        print(f"  Difference (Depth - PnP):  [{diff[0]:.4f}, {diff[1]:.4f}, {diff[2]:.4f}] m")
        print(f"  |Difference|: {diff_norm:.4f} m ({diff_norm*1000:.2f} mm)")
    
    print(f"{'='*80}\n")
    
    # Create 2-panel figure
    fig = plt.figure(figsize=(14, 6))
    
    # XY plane view
    ax1 = fig.add_subplot(121)
    for tag_id, data in tag_poses_with_depth.items():
        R, t_pnp, P_depth, u, v, Zc = data
        ax1.plot(t_pnp[0], t_pnp[1], 'o', markersize=12, label=f'Tag {tag_id} PnP')
        ax1.plot(P_depth[0], P_depth[1], '^', markersize=12, label=f'Tag {tag_id} Depth')
        ax1.arrow(t_pnp[0], t_pnp[1], 
                  P_depth[0] - t_pnp[0], P_depth[1] - t_pnp[1],
                  head_width=0.01, head_length=0.01, fc='gray', ec='gray',
                  alpha=0.5, linewidth=2)
    
    ax1.set_xlabel('X (meters)', fontsize=12)
    ax1.set_ylabel('Y (meters)', fontsize=12)
    ax1.set_title('Top View (XY Plane)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_aspect('equal')
    
    # XZ plane view
    ax2 = fig.add_subplot(122)
    for tag_id, data in tag_poses_with_depth.items():
        R, t_pnp, P_depth, u, v, Zc = data
        ax2.plot(t_pnp[2], t_pnp[0], 'o', markersize=12, label=f'Tag {tag_id} PnP')
        ax2.plot(P_depth[2], P_depth[0], '^', markersize=12, label=f'Tag {tag_id} Depth')
        ax2.arrow(t_pnp[2], t_pnp[0],
                  P_depth[2] - t_pnp[2], P_depth[0] - t_pnp[0],
                  head_width=0.01, head_length=0.01, fc='gray', ec='gray',
                  alpha=0.5, linewidth=2)
    
    ax2.set_xlabel('Z (meters) - Distance from camera', fontsize=12)
    ax2.set_ylabel('X (meters)', fontsize=12)
    ax2.set_title('Side View (XZ Plane)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_aspect('equal')
    
    plt.tight_layout()

    # ---- SAVE FIG ----
    out_path = OUT_DIR / "depth_validated_positions.png"
    fig.savefig(out_path, dpi=200)
    print(f"[SAVED] {out_path}")

    return fig


def create_combined_corner_plot(all_tag_data):
    """
    Create a single plot showing all tags' detected and reprojected corners and SAVE it.
    
    Args:
        all_tag_data: dict {tag_id: (detected_corners, reprojected_corners, mean_error)}
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for idx, (tag_id, data) in enumerate(all_tag_data.items()):
        detected_corners, reprojected_corners, mean_error = data
        color = colors[idx % len(colors)]
        
        # Plot detected corners
        det_poly = Polygon(detected_corners, fill=False, edgecolor=color,
                           linewidth=2, linestyle='-', alpha=0.7)
        ax.add_patch(det_poly)
        ax.plot(detected_corners[:, 0], detected_corners[:, 1],
                'o', color=color, markersize=10,
                label=f'Tag {tag_id} Detected')
        
        # Plot reprojected corners
        repr_poly = Polygon(reprojected_corners, fill=False, edgecolor=color,
                            linewidth=2, linestyle='--', alpha=0.7)
        ax.add_patch(repr_poly)
        ax.plot(reprojected_corners[:, 0], reprojected_corners[:, 1],
                '^', color=color, markersize=10,
                label=f'Tag {tag_id} Reprojected')
        
        # Add tag ID label at center
        center = detected_corners.mean(axis=0)
        ax.text(center[0], center[1], f'{tag_id}',
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title('All Tags - Detected vs Reprojected Corners',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()

    # ---- SAVE FIG ----
    out_path = OUT_DIR / "all_tags_reprojection.png"
    fig.savefig(out_path, dpi=200)
    print(f"[SAVED] {out_path}")

    return fig


# Optional: if you want to quickly test saving behavior, call plt.show() at the end of your pipeline.
# Otherwise, just import these functions and call them from your PnP code—the PNGs will appear in ./viz_outputs/.
