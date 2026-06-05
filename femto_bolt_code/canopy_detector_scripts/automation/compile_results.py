#!/usr/bin/env python3
"""
Compile Results to CSV
=======================
Compiles all JSON results from capture subfolders into a single CSV file.

Usage:
    python compile_results.py captures/
    python compile_results.py /path/to/captures --output summary.csv
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


def find_result_files(root_dir: Path) -> List[Path]:
    """
    Find all results_*.json files in viz_output folders.
    
    Args:
        root_dir: Root directory containing capture_* subfolders
    
    Returns:
        List of paths to results JSON files
    """
    result_files = []
    
    # Look for capture_* folders
    for capture_folder in sorted(root_dir.glob("capture_*")):
        if not capture_folder.is_dir():
            continue
        
        # Look for viz_output_v2_* folders inside
        for viz_folder in capture_folder.glob("viz_output_v2_*"):
            if not viz_folder.is_dir():
                continue
            
            # Look for results_*.json files (not raw)
            for json_file in viz_folder.glob("results_*.json"):
                if not json_file.name.endswith("_raw.json"):
                    result_files.append(json_file)
    
    return result_files


def extract_data_from_json(json_path: Path) -> Dict[str, Any]:
    """
    Extract measurement data from a results JSON file.
    
    Args:
        json_path: Path to results JSON file
    
    Returns:
        Dict with extracted measurements
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract all relevant fields
        extracted = {
            # Metadata
            "timestamp": data.get("metadata", {}).get("timestamp", "N/A"),
            "capture_folder": data.get("metadata", {}).get("capture_folder", "N/A"),
            "processing_date": data.get("metadata", {}).get("processing_date", "N/A"),
            
            # Height measurements
            "height_raw_cm": data.get("measurements", {}).get("canopy_height", {}).get("raw_cm"),
            "height_raw_m": data.get("measurements", {}).get("canopy_height", {}).get("raw_m"),
            "height_corrected_cm": data.get("measurements", {}).get("canopy_height", {}).get("corrected_cm"),
            "height_corrected_m": data.get("measurements", {}).get("canopy_height", {}).get("corrected_m"),
            
            # Baseline measurements
            "measured_baseline_m": data.get("measurements", {}).get("baseline", {}).get("measured_m"),
            "measured_baseline_cm": data.get("measurements", {}).get("baseline", {}).get("measured_cm"),
            "known_baseline_m": data.get("measurements", {}).get("baseline", {}).get("known_m"),
            "known_baseline_cm": data.get("measurements", {}).get("baseline", {}).get("known_cm"),
            "scale_factor": data.get("measurements", {}).get("baseline", {}).get("scale_factor"),
            
            # Segmentation
            "mask_coverage_pct": data.get("measurements", {}).get("segmentation", {}).get("mask_coverage_percent"),
            "bbox_x_min": data.get("measurements", {}).get("segmentation", {}).get("bounding_box", {}).get("x_min"),
            "bbox_y_min": data.get("measurements", {}).get("segmentation", {}).get("bounding_box", {}).get("y_min"),
            "bbox_x_max": data.get("measurements", {}).get("segmentation", {}).get("bounding_box", {}).get("x_max"),
            "bbox_y_max": data.get("measurements", {}).get("segmentation", {}).get("bounding_box", {}).get("y_max"),
            
            # Canopy coordinates
            "canopy_pixel_x": data.get("coordinates", {}).get("canopy_top", {}).get("pixel", {}).get("x"),
            "canopy_pixel_y": data.get("coordinates", {}).get("canopy_top", {}).get("pixel", {}).get("y"),
            "canopy_camera_x": data.get("coordinates", {}).get("canopy_top", {}).get("camera_frame", {}).get("x"),
            "canopy_camera_y": data.get("coordinates", {}).get("canopy_top", {}).get("camera_frame", {}).get("y"),
            "canopy_camera_z": data.get("coordinates", {}).get("canopy_top", {}).get("camera_frame", {}).get("z"),
            "canopy_world_x": data.get("coordinates", {}).get("canopy_top", {}).get("world_frame", {}).get("x"),
            "canopy_world_y": data.get("coordinates", {}).get("canopy_top", {}).get("world_frame", {}).get("y"),
            "canopy_world_z": data.get("coordinates", {}).get("canopy_top", {}).get("world_frame", {}).get("z"),
            
            # Coordinate frame
            "origin_x": data.get("coordinate_frame", {}).get("origin", {}).get("x"),
            "origin_y": data.get("coordinate_frame", {}).get("origin", {}).get("y"),
            "origin_z": data.get("coordinate_frame", {}).get("origin", {}).get("z"),
            
            # Summary
            "final_height_cm": data.get("summary", {}).get("final_height_cm"),
            "calibrated": data.get("summary", {}).get("calibrated"),
            
            # File path for reference
            "json_file": str(json_path)
        }
        
        return extracted
    
    except Exception as e:
        return {
            "timestamp": "ERROR",
            "capture_folder": str(json_path.parent.parent),
            "error": str(e),
            "json_file": str(json_path)
        }


def write_csv(data: List[Dict[str, Any]], output_path: Path):
    """
    Write extracted data to CSV file.
    
    Args:
        data: List of extracted data dictionaries
        output_path: Path to output CSV file
    """
    if not data:
        print("[WARNING] No data to write")
        return
    
    # Get all unique keys from all dictionaries
    all_keys = set()
    for row in data:
        all_keys.update(row.keys())
    
    # Define column order (most important first)
    priority_columns = [
        "timestamp",
        "capture_folder",
        "height_corrected_cm",
        "height_raw_cm",
        "scale_factor",
        "measured_baseline_cm",
        "mask_coverage_pct",
        "canopy_pixel_x",
        "canopy_pixel_y",
        "canopy_world_x",
        "canopy_world_y",
        "canopy_world_z"
    ]
    
    # Add remaining columns
    remaining_columns = sorted(all_keys - set(priority_columns))
    fieldnames = [col for col in priority_columns if col in all_keys] + remaining_columns
    
    # Write CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)
    
    print(f"CSV written: {output_path}")
    print(f"  Rows: {len(data)}")
    print(f"  Columns: {len(fieldnames)}")


def main():
    parser = argparse.ArgumentParser(
        description="Compile all JSON results into a single CSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Create summary.csv in the captures folder
  python compile_results.py captures/
  
  # Specify custom output filename
  python compile_results.py captures/ --output my_results.csv
  
  # Save to different location
  python compile_results.py captures/ --output /path/to/results.csv
  
  # Include error rows in output
  python compile_results.py captures/ --include-errors

The script will:
1. Find all capture_* subfolders
2. Look for viz_output_v2_*/ folders inside each
3. Read results_*.json files
4. Extract all measurement data
5. Compile into a CSV file

CSV includes all measurements:
- Height (raw and corrected)
- Baseline and scale factor
- Mask coverage
- Canopy pixel coordinates
- Canopy world coordinates
- Coordinate frame information
- And more...
        """
    )
    
    parser.add_argument("folder", type=str,
                        help="Path to folder containing capture_* subfolders")
    parser.add_argument("--output", "-o", type=str, default="results_summary.csv",
                        help="Output CSV filename (default: results_summary.csv)")
    parser.add_argument("--include-errors", action="store_true",
                        help="Include rows that had errors during processing")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed information")
    
    args = parser.parse_args()
    
    # Setup paths
    root_dir = Path(args.folder)
    if not root_dir.exists():
        print(f"[ERROR] Directory not found: {root_dir}")
        return 1
    
    if not root_dir.is_dir():
        print(f"[ERROR] Not a directory: {root_dir}")
        return 1
    
    # Determine output path
    if Path(args.output).is_absolute():
        output_path = Path(args.output)
    else:
        # Save in the provided folder
        output_path = root_dir / args.output
    
    print("\n" + "=" * 70)
    print("COMPILE RESULTS TO CSV")
    print("=" * 70)
    print(f"Input folder: {root_dir.absolute()}")
    print(f"Output file: {output_path.absolute()}")
    print()
    
    # Find all result files
    print("Searching for result files...")
    result_files = find_result_files(root_dir)
    
    if not result_files:
        print(f"[WARNING] No result files found in {root_dir}")
        print("\nExpected structure:")
        print("  captures/")
        print("    capture_*/")
        print("      viz_output_v2_*/")
        print("        results_*.json")
        return 1
    
    print(f"Found {len(result_files)} result file(s)")
    
    if args.verbose:
        for f in result_files:
            print(f"  - {f.relative_to(root_dir)}")
    
    print()
    
    # Extract data from each file
    print("Extracting data...")
    all_data = []
    errors = []
    
    for json_file in result_files:
        if args.verbose:
            print(f"  Processing: {json_file.name}")
        
        data = extract_data_from_json(json_file)
        
        if "error" in data:
            errors.append(data)
            if args.include_errors:
                all_data.append(data)
        else:
            all_data.append(data)
    
    print(f"Successfully extracted: {len(all_data) - len(errors)}")
    if errors:
        print(f"Errors encountered: {len(errors)}")
        if not args.include_errors:
            print("  (Use --include-errors to include these in CSV)")
    
    if not all_data:
        print("[ERROR] No data extracted successfully")
        return 1
    
    print()
    
    # Write CSV
    print("Writing CSV...")
    write_csv(all_data, output_path)
    
    # Show preview
    print("\nPreview (first 3 rows):")
    print("-" * 70)
    
    if len(all_data) > 0:
        # Show key columns
        preview_cols = [
            "timestamp",
            "height_corrected_cm",
            "scale_factor",
            "mask_coverage_pct"
        ]
        
        # Print header
        header = " | ".join(f"{col:20s}" for col in preview_cols if col in all_data[0])
        print(header)
        print("-" * len(header))
        
        # Print first 3 rows
        for row in all_data[:3]:
            values = []
            for col in preview_cols:
                if col in row:
                    val = row[col]
                    if val is None:
                        val_str = "N/A"
                    elif isinstance(val, float):
                        val_str = f"{val:.2f}"
                    else:
                        val_str = str(val)
                    values.append(f"{val_str:20s}")
            print(" | ".join(values))
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total captures: {len(all_data)}")
    print(f"CSV file: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Calculate statistics if data is available
    if len(all_data) > 0 and "height_corrected_cm" in all_data[0]:
        heights = [row.get("height_corrected_cm") or row.get("height_raw_cm") 
                   for row in all_data if row.get("height_corrected_cm") or row.get("height_raw_cm")]
        
        if heights:
            print(f"\nHeight statistics:")
            print(f"  Mean: {sum(heights)/len(heights):.2f} cm")
            print(f"  Min: {min(heights):.2f} cm")
            print(f"  Max: {max(heights):.2f} cm")
            print(f"  Range: {max(heights) - min(heights):.2f} cm")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())