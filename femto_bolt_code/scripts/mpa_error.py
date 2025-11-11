#!/usr/bin/env python3
"""
Point Cloud Alignment Error Analysis
Computes bidirectional distance metrics and overlap percentage between two point clouds.
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# =========================== CONFIG ===========================
# Input point clouds
SCAN_PLY_PATH = Path(r"./new_test_exports/point_cloud_20251031_155222/cropped_camframe.ply")
CAD_PLY_PATH = Path(r"./new_test_exports/transformed_cad_output.ply")

# Output paths
OUTPUT_DIR = Path("./alignment_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = OUTPUT_DIR / "alignment_metrics.json"
REPORT_TXT = OUTPUT_DIR / "alignment_report.txt"
ERROR_MAP_SCAN = OUTPUT_DIR / "error_map_scan.ply"
ERROR_MAP_CAD = OUTPUT_DIR / "error_map_cad.ply"

# Analysis parameters
OVERLAP_THRESHOLD_M = 0.010  # 10mm - points within this distance are "overlapping"
VOXEL_DOWNSAMPLE = 0.002     # 2mm voxel size for downsampling (0 = no downsample)
MAX_CORRESPONDENCE_DIST = 0.100  # 100mm - max distance to consider for nearest neighbor

# Visualization
VISUALIZE = True
SAVE_ERROR_MAPS = True

# =========================== DATA STRUCTURES ===========================
@dataclass
class AlignmentMetrics:
    """Container for alignment error metrics."""
    # Chamfer distance (bidirectional)
    chamfer_distance_mean: float
    chamfer_distance_rms: float
    chamfer_distance_median: float
    chamfer_distance_std: float
    
    # One-way distances
    scan_to_cad_mean: float
    scan_to_cad_median: float
    scan_to_cad_max: float
    cad_to_scan_mean: float
    cad_to_scan_median: float
    cad_to_scan_max: float
    
    # Overlap metrics
    overlap_percentage_scan: float  # % of scan points with nearby CAD point
    overlap_percentage_cad: float   # % of CAD points with nearby scan point
    overlap_percentage_symmetric: float  # Average of both
    
    # Point counts
    num_scan_points: int
    num_cad_points: int
    num_scan_overlapping: int
    num_cad_overlapping: int
    
    # Distribution percentiles
    percentile_50: float  # Median
    percentile_75: float
    percentile_90: float
    percentile_95: float
    percentile_99: float

# =========================== CORE FUNCTIONS ===========================

def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()             # np.float64 -> float, np.int64 -> int
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

def load_point_cloud(path: Path, voxel_size: float = 0.0) -> o3d.geometry.PointCloud:
    """Load and optionally downsample a point cloud."""
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    
    pcd = o3d.io.read_point_cloud(str(path))
    print(f"[LOAD] {path.name}: {len(pcd.points)} points")
    
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"       Downsampled to {len(pcd.points)} points (voxel={voxel_size}m)")
    
    return pcd

def compute_point_to_point_distances(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    max_dist: float = np.inf
) -> np.ndarray:
    """
    Compute nearest neighbor distances from source to target.
    
    Args:
        source_pts: Source point cloud (N, 3)
        target_pts: Target point cloud (M, 3)
        max_dist: Maximum distance to consider
    
    Returns:
        distances: Distance to nearest neighbor for each source point (N,)
    """
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(target_pts)
    
    # Query nearest neighbor for each source point
    distances, indices = tree.query(source_pts, k=1, distance_upper_bound=max_dist)
    
    # Handle points with no neighbor within max_dist
    distances[distances == np.inf] = max_dist
    
    return distances

def compute_chamfer_distance(
    scan_pts: np.ndarray,
    cad_pts: np.ndarray,
    max_dist: float = np.inf
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute bidirectional Chamfer distance.
    
    Chamfer distance = (1/|S|)Σ min||s-c|| + (1/|C|)Σ min||c-s||
    where S = scan points, C = CAD points
    
    Args:
        scan_pts: Scan point cloud (N, 3)
        cad_pts: CAD point cloud (M, 3)
        max_dist: Maximum distance to consider
    
    Returns:
        dist_scan_to_cad: Distance from each scan point to nearest CAD point (N,)
        dist_cad_to_scan: Distance from each CAD point to nearest scan point (M,)
        chamfer_dist: Average of mean distances (scalar)
    """
    print("\n[CHAMFER] Computing bidirectional distances...")
    
    # Scan -> CAD
    dist_scan_to_cad = compute_point_to_point_distances(scan_pts, cad_pts, max_dist)
    mean_scan_to_cad = np.mean(dist_scan_to_cad)
    print(f"  Scan → CAD: mean={mean_scan_to_cad*1000:.2f}mm")
    
    # CAD -> Scan
    dist_cad_to_scan = compute_point_to_point_distances(cad_pts, scan_pts, max_dist)
    mean_cad_to_scan = np.mean(dist_cad_to_scan)
    print(f"  CAD → Scan: mean={mean_cad_to_scan*1000:.2f}mm")
    
    # Chamfer distance (average of both directions)
    chamfer_dist = (mean_scan_to_cad + mean_cad_to_scan) / 2.0
    print(f"  Chamfer distance: {chamfer_dist*1000:.2f}mm")
    
    return dist_scan_to_cad, dist_cad_to_scan, chamfer_dist

def compute_overlap_metrics(
    dist_scan_to_cad: np.ndarray,
    dist_cad_to_scan: np.ndarray,
    threshold: float
) -> Tuple[float, float, float, int, int]:
    """
    Compute overlap percentage metrics.
    
    A point is considered "overlapping" if its nearest neighbor in the other
    cloud is within the threshold distance.
    
    Args:
        dist_scan_to_cad: Distance from each scan point to nearest CAD point
        dist_cad_to_scan: Distance from each CAD point to nearest scan point
        threshold: Distance threshold for overlap (meters)
    
    Returns:
        overlap_pct_scan: % of scan points within threshold of CAD
        overlap_pct_cad: % of CAD points within threshold of scan
        overlap_pct_symmetric: Average of both percentages
        num_scan_overlap: Number of overlapping scan points
        num_cad_overlap: Number of overlapping CAD points
    """
    print(f"\n[OVERLAP] Computing overlap with threshold={threshold*1000:.1f}mm...")
    
    # Count points within threshold
    num_scan_overlap = int(np.sum(dist_scan_to_cad <= threshold))
    num_cad_overlap = int(np.sum(dist_cad_to_scan <= threshold))
    
    # Compute percentages
    overlap_pct_scan = 100.0 * num_scan_overlap / len(dist_scan_to_cad)
    overlap_pct_cad = 100.0 * num_cad_overlap / len(dist_cad_to_scan)
    overlap_pct_symmetric = (overlap_pct_scan + overlap_pct_cad) / 2.0
    
    print(f"  Scan overlap: {overlap_pct_scan:.1f}% ({num_scan_overlap}/{len(dist_scan_to_cad)} points)")
    print(f"  CAD overlap:  {overlap_pct_cad:.1f}% ({num_cad_overlap}/{len(dist_cad_to_scan)} points)")
    print(f"  Symmetric:    {overlap_pct_symmetric:.1f}%")
    
    return overlap_pct_scan, overlap_pct_cad, overlap_pct_symmetric, num_scan_overlap, num_cad_overlap

def compute_comprehensive_metrics(
    scan_pts: np.ndarray,
    cad_pts: np.ndarray,
    overlap_threshold: float,
    max_correspondence_dist: float
) -> AlignmentMetrics:
    """
    Compute all alignment metrics.
    
    Args:
        scan_pts: Scan point cloud (N, 3)
        cad_pts: CAD point cloud (M, 3)
        overlap_threshold: Distance threshold for overlap
        max_correspondence_dist: Maximum distance to consider for correspondences
    
    Returns:
        AlignmentMetrics object with all computed metrics
    """
    # Compute bidirectional distances
    dist_scan_to_cad, dist_cad_to_scan, chamfer_mean = compute_chamfer_distance(
        scan_pts, cad_pts, max_correspondence_dist
    )
    
    # Combined distances for overall statistics
    all_distances = np.concatenate([dist_scan_to_cad, dist_cad_to_scan])
    
    # Compute overlap metrics
    overlap_scan, overlap_cad, overlap_sym, n_scan_overlap, n_cad_overlap = compute_overlap_metrics(
        dist_scan_to_cad, dist_cad_to_scan, overlap_threshold
    )
    
    # Compute statistics
    print("\n[STATS] Computing statistics...")
    
    metrics = AlignmentMetrics(
        # Chamfer distance
        chamfer_distance_mean=chamfer_mean,
        chamfer_distance_rms=np.sqrt(np.mean(all_distances ** 2)),
        chamfer_distance_median=np.median(all_distances),
        chamfer_distance_std=np.std(all_distances),
        
        # One-way distances
        scan_to_cad_mean=np.mean(dist_scan_to_cad),
        scan_to_cad_median=np.median(dist_scan_to_cad),
        scan_to_cad_max=np.max(dist_scan_to_cad),
        cad_to_scan_mean=np.mean(dist_cad_to_scan),
        cad_to_scan_median=np.median(dist_cad_to_scan),
        cad_to_scan_max=np.max(dist_cad_to_scan),
        
        # Overlap metrics
        overlap_percentage_scan=overlap_scan,
        overlap_percentage_cad=overlap_cad,
        overlap_percentage_symmetric=overlap_sym,
        
        # Point counts
        num_scan_points=len(scan_pts),
        num_cad_points=len(cad_pts),
        num_scan_overlapping=n_scan_overlap,
        num_cad_overlapping=n_cad_overlap,
        
        # Percentiles
        percentile_50=np.percentile(all_distances, 50),
        percentile_75=np.percentile(all_distances, 75),
        percentile_90=np.percentile(all_distances, 90),
        percentile_95=np.percentile(all_distances, 95),
        percentile_99=np.percentile(all_distances, 99),
    )
    
    return metrics

# =========================== VISUALIZATION ===========================

def create_error_colormap(distances: np.ndarray, max_dist: float = None) -> np.ndarray:
    """
    Create color map based on distance errors.
    
    Color scheme:
    - Blue: 0mm (perfect match)
    - Green: Low error
    - Yellow: Medium error
    - Red: High error
    
    Args:
        distances: Distance array (N,)
        max_dist: Maximum distance for color scaling (None = use max distance)
    
    Returns:
        colors: RGB colors (N, 3) in range [0, 1]
    """
    if max_dist is None:
        max_dist = np.max(distances)
    
    # Normalize distances to [0, 1]
    normalized = np.clip(distances / max_dist, 0, 1)
    
    # Create colormap: blue -> cyan -> green -> yellow -> red
    colors = np.zeros((len(distances), 3))
    
    for i, val in enumerate(normalized):
        if val < 0.25:  # Blue to Cyan
            t = val / 0.25
            colors[i] = [0, t, 1]
        elif val < 0.5:  # Cyan to Green
            t = (val - 0.25) / 0.25
            colors[i] = [0, 1, 1 - t]
        elif val < 0.75:  # Green to Yellow
            t = (val - 0.5) / 0.25
            colors[i] = [t, 1, 0]
        else:  # Yellow to Red
            t = (val - 0.75) / 0.25
            colors[i] = [1, 1 - t, 0]
    
    return colors

def visualize_alignment_error(
    pcd_scan: o3d.geometry.PointCloud,
    pcd_cad: o3d.geometry.PointCloud,
    dist_scan_to_cad: np.ndarray,
    dist_cad_to_scan: np.ndarray,
    max_error_display: float = 0.050
):
    """
    Visualize alignment error with color-coded point clouds.
    
    Args:
        pcd_scan: Scan point cloud
        pcd_cad: CAD point cloud
        dist_scan_to_cad: Distances from scan to CAD
        dist_cad_to_scan: Distances from CAD to scan
        max_error_display: Maximum error for color scaling (meters)
    """
    print("\n[VIZ] Creating visualization...")
    
    # Color scan points by error
    pcd_scan_colored = o3d.geometry.PointCloud(pcd_scan)
    colors_scan = create_error_colormap(dist_scan_to_cad, max_error_display)
    pcd_scan_colored.colors = o3d.utility.Vector3dVector(colors_scan)
    
    # Color CAD points by error
    pcd_cad_colored = o3d.geometry.PointCloud(pcd_cad)
    colors_cad = create_error_colormap(dist_cad_to_scan, max_error_display)
    pcd_cad_colored.colors = o3d.utility.Vector3dVector(colors_cad)
    
    # Save error maps
    if SAVE_ERROR_MAPS:
        o3d.io.write_point_cloud(str(ERROR_MAP_SCAN), pcd_scan_colored)
        o3d.io.write_point_cloud(str(ERROR_MAP_CAD), pcd_cad_colored)
        print(f"  Saved: {ERROR_MAP_SCAN}")
        print(f"  Saved: {ERROR_MAP_CAD}")
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    print("\n  Color legend:")
    print("    Blue:   0mm (perfect alignment)")
    print("    Green:  Low error")
    print("    Yellow: Medium error")
    print("    Red:    High error (>{}mm)".format(max_error_display * 1000))
    
    if VISUALIZE:
        print("\n  Opening visualization...")
        print("  Left side: Scan cloud (colored by distance to CAD)")
        print("  Right side: CAD cloud (colored by distance to scan)")
        
        # Translate CAD to the side for comparison
        pcd_cad_offset = o3d.geometry.PointCloud(pcd_cad_colored)
        bbox = pcd_scan.get_axis_aligned_bounding_box()
        offset = (bbox.max_bound - bbox.min_bound)[0] * 1.5
        pcd_cad_offset.translate([offset, 0, 0])
        
        o3d.visualization.draw_geometries(
            [pcd_scan_colored, pcd_cad_offset, coord_frame],
            window_name="Alignment Error Visualization",
            width=1920,
            height=1080
        )

# =========================== REPORTING ===========================

def save_metrics_report(metrics: AlignmentMetrics, overlap_threshold: float):
    """Save metrics to JSON and text files."""
    print(f"\n[SAVE] Saving reports...")
    
    # Convert to dictionary for JSON
    metrics_dict = {
        "chamfer_distance_mm": {
            "mean": float(metrics.chamfer_distance_mean * 1000),
            "rms": float(metrics.chamfer_distance_rms * 1000),
            "median": float(metrics.chamfer_distance_median * 1000),
            "std": float(metrics.chamfer_distance_std * 1000),
        },
        "scan_to_cad_mm": {
            "mean": float(metrics.scan_to_cad_mean * 1000),
            "median": float(metrics.scan_to_cad_median * 1000),
            "max": float(metrics.scan_to_cad_max * 1000),
        },
        "cad_to_scan_mm": {
            "mean": float(metrics.cad_to_scan_mean * 1000),
            "median": float(metrics.cad_to_scan_median * 1000),
            "max": float(metrics.cad_to_scan_max * 1000),
        },
        "overlap_percentage": {
            "scan": float(metrics.overlap_percentage_scan),
            "cad": float(metrics.overlap_percentage_cad),
            "symmetric": float(metrics.overlap_percentage_symmetric),
            "threshold_mm": float(overlap_threshold * 1000),
        },
        "point_counts": {
            "scan_total": metrics.num_scan_points,
            "cad_total": metrics.num_cad_points,
            "scan_overlapping": metrics.num_scan_overlapping,
            "cad_overlapping": metrics.num_cad_overlapping,
        },
        "percentiles_mm": {
            "50th": float(metrics.percentile_50 * 1000),
            "75th": float(metrics.percentile_75 * 1000),
            "90th": float(metrics.percentile_90 * 1000),
            "95th": float(metrics.percentile_95 * 1000),
            "99th": float(metrics.percentile_99 * 1000),
        }
    }
    
    # Save JSON
    with open(REPORT_JSON, 'w') as f:
        json.dump(metrics_dict, f, indent=2, default=_json_default)
    print(f"  JSON: {REPORT_JSON}")
    
    # Save text report
    with open(REPORT_TXT, 'w') as f:
        f.write("="*70 + "\n")
        f.write("POINT CLOUD ALIGNMENT ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("CHAMFER DISTANCE (Bidirectional)\n")
        f.write("-"*70 + "\n")
        f.write(f"  Mean:     {metrics.chamfer_distance_mean*1000:8.3f} mm\n")
        f.write(f"  RMS:      {metrics.chamfer_distance_rms*1000:8.3f} mm\n")
        f.write(f"  Median:   {metrics.chamfer_distance_median*1000:8.3f} mm\n")
        f.write(f"  Std Dev:  {metrics.chamfer_distance_std*1000:8.3f} mm\n\n")
        
        f.write("ONE-WAY DISTANCES\n")
        f.write("-"*70 + "\n")
        f.write("Scan -> CAD:\n")
        f.write(f"  Mean:     {metrics.scan_to_cad_mean*1000:8.3f} mm\n")
        f.write(f"  Median:   {metrics.scan_to_cad_median*1000:8.3f} mm\n")
        f.write(f"  Max:      {metrics.scan_to_cad_max*1000:8.3f} mm\n\n")
        f.write("CAD -> Scan:\n")
        f.write(f"  Mean:     {metrics.cad_to_scan_mean*1000:8.3f} mm\n")
        f.write(f"  Median:   {metrics.cad_to_scan_median*1000:8.3f} mm\n")
        f.write(f"  Max:      {metrics.cad_to_scan_max*1000:8.3f} mm\n\n")
        
        f.write(f"OVERLAP METRICS (threshold = {overlap_threshold*1000:.1f}mm)\n")
        f.write("-"*70 + "\n")
        f.write(f"  Scan overlap:  {metrics.overlap_percentage_scan:6.2f}% ")
        f.write(f"({metrics.num_scan_overlapping}/{metrics.num_scan_points} points)\n")
        f.write(f"  CAD overlap:   {metrics.overlap_percentage_cad:6.2f}% ")
        f.write(f"({metrics.num_cad_overlapping}/{metrics.num_cad_points} points)\n")
        f.write(f"  Symmetric:     {metrics.overlap_percentage_symmetric:6.2f}%\n\n")
        
        f.write("ERROR DISTRIBUTION (Percentiles)\n")
        f.write("-"*70 + "\n")
        f.write(f"  50th (median): {metrics.percentile_50*1000:8.3f} mm\n")
        f.write(f"  75th:          {metrics.percentile_75*1000:8.3f} mm\n")
        f.write(f"  90th:          {metrics.percentile_90*1000:8.3f} mm\n")
        f.write(f"  95th:          {metrics.percentile_95*1000:8.3f} mm\n")
        f.write(f"  99th:          {metrics.percentile_99*1000:8.3f} mm\n\n")
        
        f.write("POINT COUNTS\n")
        f.write("-"*70 + "\n")
        f.write(f"  Scan points:   {metrics.num_scan_points:8d}\n")
        f.write(f"  CAD points:    {metrics.num_cad_points:8d}\n")
        
    print(f"  Text: {REPORT_TXT}")

def print_summary(metrics: AlignmentMetrics):
    """Print summary to console."""
    print("\n" + "="*70)
    print("ALIGNMENT ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nChamfer Distance (Mean): {metrics.chamfer_distance_mean*1000:.3f} mm")
    print(f"Chamfer Distance (RMS):  {metrics.chamfer_distance_rms*1000:.3f} mm")
    print(f"\nOverlap (Symmetric):     {metrics.overlap_percentage_symmetric:.1f}%")
    print(f"  Scan overlap:          {metrics.overlap_percentage_scan:.1f}%")
    print(f"  CAD overlap:           {metrics.overlap_percentage_cad:.1f}%")
    print(f"\nMedian Error:            {metrics.chamfer_distance_median*1000:.3f} mm")
    print(f"95th Percentile Error:   {metrics.percentile_95*1000:.3f} mm")
    print("="*70)

# =========================== MAIN ===========================

def main():
    print("="*70)
    print("POINT CLOUD ALIGNMENT ERROR ANALYSIS")
    print("="*70)
    
    # Load point clouds
    print("\n[1/4] Loading point clouds...")
    pcd_scan = load_point_cloud(SCAN_PLY_PATH, VOXEL_DOWNSAMPLE)
    pcd_cad = load_point_cloud(CAD_PLY_PATH, VOXEL_DOWNSAMPLE)
    
    # Extract numpy arrays
    scan_pts = np.asarray(pcd_scan.points)
    cad_pts = np.asarray(pcd_cad.points)
    
    # Compute metrics
    print("\n[2/4] Computing alignment metrics...")
    metrics = compute_comprehensive_metrics(
        scan_pts, cad_pts,
        overlap_threshold=OVERLAP_THRESHOLD_M,
        max_correspondence_dist=MAX_CORRESPONDENCE_DIST
    )
    
    # Get distances for visualization
    dist_scan_to_cad = compute_point_to_point_distances(
        scan_pts, cad_pts, MAX_CORRESPONDENCE_DIST
    )
    dist_cad_to_scan = compute_point_to_point_distances(
        cad_pts, scan_pts, MAX_CORRESPONDENCE_DIST
    )
    
    # Save reports
    print("\n[3/4] Generating reports...")
    save_metrics_report(metrics, OVERLAP_THRESHOLD_M)
    print_summary(metrics)
    
    # Visualize
    print("\n[4/4] Visualizing results...")
    visualize_alignment_error(
        pcd_scan, pcd_cad,
        dist_scan_to_cad, dist_cad_to_scan,
        max_error_display=0.050  # 50mm max for color scale
    )
    
    print("\n✓ Analysis complete!")
    print(f"  Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()