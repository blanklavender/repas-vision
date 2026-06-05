#!/usr/bin/env python3
"""
Batch Prompts File Generator
=============================
Automatically creates template prompts JSON files for all capture folders.
Each prompts file is pre-formatted for 3 track points that you can edit later.

Usage:
    # Create template files with placeholder coordinates
    python create_prompts_batch.py /path/to/captures_folder
    
    # Create template files with default coordinates
    python create_prompts_batch.py /path/to/captures_folder --default-points 400,300 600,300 800,300
    
    # Specify mode (points or box)
    python create_prompts_batch.py /path/to/captures_folder --mode box --box 300,250,1000,550
    
    # Skip folders that already have prompts files
    python create_prompts_batch.py /path/to/captures_folder --skip-existing


    Example: python create_prompts_batch.py ../../../../captures/canopy_test_captures_33.4/ --default-points 400,300 600,300 800,300
"""

import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any


def extract_timestamp_from_folder(folder_path: Path) -> Optional[str]:
    """
    Extract timestamp from capture folder name.
    
    Args:
        folder_path: Path to capture folder
    
    Returns:
        Timestamp string (e.g., "20260210_161537") or None if invalid
    """
    folder_name = folder_path.name
    if not folder_name.startswith("capture_"):
        return None
    
    timestamp = folder_name.replace("capture_", "")
    return timestamp


def create_prompts_template(
    mode: str = "points",
    num_points: int = 3,
    default_points: Optional[List[List[int]]] = None,
    box: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Create a template prompts dictionary.
    
    Args:
        mode: "points" or "box"
        num_points: Number of points for point mode
        default_points: Optional list of [x, y] coordinates
        box: Optional [x1, y1, x2, y2] for box mode
    
    Returns:
        Prompts dictionary
    """
    if mode == "points":
        if default_points is not None:
            points = default_points[:num_points]  # Use provided defaults
        else:
            # Create placeholder points - user must edit these
            points = [[0, 0] for _ in range(num_points)]
        
        return {
            "mode": "points",
            "point_prompts": points,
            "point_labels": [1] * len(points),
            "box_prompt": None
        }
    else:  # box mode
        if box is not None:
            box_coords = box
        else:
            # Placeholder box - user must edit
            box_coords = [0, 0, 0, 0]
        
        return {
            "mode": "box",
            "point_prompts": None,
            "point_labels": None,
            "box_prompt": box_coords
        }


def find_capture_folders(root_dir: Path) -> List[Path]:
    """
    Find all capture_* folders in the given directory.
    
    Args:
        root_dir: Root directory to search
    
    Returns:
        List of capture folder paths
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    if not root_dir.is_dir():
        raise ValueError(f"Not a directory: {root_dir}")
    
    # Find all folders starting with "capture_"
    capture_folders = sorted([
        f for f in root_dir.iterdir()
        if f.is_dir() and f.name.startswith("capture_")
    ])
    
    return capture_folders


def process_folder(
    folder_path: Path,
    mode: str,
    num_points: int,
    default_points: Optional[List[List[int]]],
    box: Optional[List[int]],
    skip_existing: bool,
    dry_run: bool
) -> Dict[str, Any]:
    """
    Process a single capture folder.
    
    Returns:
        Dict with status and info
    """
    result = {
        "folder": str(folder_path),
        "success": False,
        "message": "",
        "prompts_path": None
    }
    
    # Extract timestamp
    timestamp = extract_timestamp_from_folder(folder_path)
    if timestamp is None:
        result["message"] = "Invalid folder name (doesn't start with 'capture_')"
        return result
    
    # Check if color file exists (to verify it's a valid capture folder)
    color_path = folder_path / f"color_{timestamp}.png"
    if not color_path.exists():
        result["message"] = f"No color file found (expected: {color_path.name})"
        return result
    
    # Determine prompts file path
    prompts_path = folder_path / f"prompts_{timestamp}.json"
    
    # Check if prompts file already exists
    if prompts_path.exists() and skip_existing:
        result["success"] = True
        result["message"] = "Skipped (prompts file already exists)"
        result["prompts_path"] = str(prompts_path)
        return result
    
    # Create prompts template
    prompts_dict = create_prompts_template(
        mode=mode,
        num_points=num_points,
        default_points=default_points,
        box=box
    )
    
    if dry_run:
        result["success"] = True
        result["message"] = "Dry run (would create prompts file)"
        result["prompts_path"] = str(prompts_path)
        result["prompts_content"] = prompts_dict
        return result
    
    # Write prompts file
    try:
        with open(prompts_path, 'w') as f:
            json.dump(prompts_dict, f, indent=2)
        
        result["success"] = True
        result["message"] = "Created prompts file"
        result["prompts_path"] = str(prompts_path)
        result["prompts_content"] = prompts_dict
    except Exception as e:
        result["message"] = f"Error writing file: {e}"
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch create prompts JSON files for capture folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create template files with placeholder coordinates (all zeros)
  python create_prompts_batch.py captures/
  
  # Create with default coordinates for 3 points
  python create_prompts_batch.py captures/ --default-points 400,300 600,300 800,300
  
  # Create for box mode
  python create_prompts_batch.py captures/ --mode box --box 300,250,1000,550
  
  # Skip folders that already have prompts files
  python create_prompts_batch.py captures/ --skip-existing
  
  # Dry run (see what would be created)
  python create_prompts_batch.py captures/ --dry-run

The script will:
1. Find all folders named "capture_YYYYMMDD_HHMMSS"
2. Create "prompts_YYYYMMDD_HHMMSS.json" in each folder
3. Pre-fill with template data (either placeholders or your defaults)

After running, you can:
- Manually edit the JSON files to set actual coordinates
- Use select_prompts_interactive.py to visually select points
        """
    )
    
    parser.add_argument("folder", type=str,
                        help="Path to folder containing capture_* subfolders")
    parser.add_argument("--mode", type=str, default="points", choices=["points", "box"],
                        help="Prompt mode: 'points' or 'box' (default: points)")
    parser.add_argument("--num-points", type=int, default=3,
                        help="Number of points for point mode (default: 3)")
    parser.add_argument("--default-points", type=str, nargs="+",
                        help="Default point coordinates as 'x,y' (e.g., '400,300 600,300 800,300')")
    parser.add_argument("--box", type=str,
                        help="Box coordinates as 'x1,y1,x2,y2' (e.g., '300,250,1000,550')")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip folders that already have prompts files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be created without actually creating files")
    
    args = parser.parse_args()
    
    # Parse default points if provided
    default_points = None
    if args.default_points:
        try:
            default_points = []
            for point_str in args.default_points:
                x, y = map(int, point_str.split(','))
                default_points.append([x, y])
        except ValueError:
            print("[ERROR] Invalid format for --default-points")
            print("        Use format: '400,300 600,300 800,300'")
            return
    
    # Parse box if provided
    box = None
    if args.box:
        try:
            coords = list(map(int, args.box.split(',')))
            if len(coords) != 4:
                raise ValueError("Box must have exactly 4 coordinates")
            box = coords
        except ValueError as e:
            print(f"[ERROR] Invalid format for --box: {e}")
            print("        Use format: '300,250,1000,550'")
            return
    
    # Find capture folders
    root_dir = Path(args.folder)
    try:
        capture_folders = find_capture_folders(root_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return
    
    if len(capture_folders) == 0:
        print(f"[WARNING] No capture_* folders found in: {root_dir}")
        return
    
    print("\n" + "=" * 70)
    print("BATCH PROMPTS FILE GENERATOR")
    print("=" * 70)
    print(f"Root folder: {root_dir}")
    print(f"Found {len(capture_folders)} capture folders")
    print(f"Mode: {args.mode}")
    if args.mode == "points":
        print(f"Number of points: {args.num_points}")
        if default_points:
            print(f"Default points: {default_points}")
        else:
            print("Points: Placeholder values (all zeros)")
    else:
        if box:
            print(f"Box: {box}")
        else:
            print("Box: Placeholder values (all zeros)")
    
    if args.skip_existing:
        print("Skip existing: Yes")
    if args.dry_run:
        print("[DRY RUN MODE - No files will be created]")
    print()
    
    # Process each folder
    results = []
    for folder in capture_folders:
        result = process_folder(
            folder,
            mode=args.mode,
            num_points=args.num_points,
            default_points=default_points,
            box=box,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run
        )
        results.append(result)
        
        # Print status
        status_symbol = "✓" if result["success"] else "✗"
        folder_name = folder.name
        print(f"  {status_symbol} {folder_name:30s} - {result['message']}")
        
        # Show prompts content if dry run and successful
        if args.dry_run and result["success"] and "prompts_content" in result:
            print(f"      Would create: {result['prompts_path']}")
            if not args.skip_existing or "already exists" not in result["message"]:
                print(f"      Content: {json.dumps(result['prompts_content'], indent=2)}")
            print()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    print(f"Total folders processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if not args.dry_run and successful > 0:
        print(f"\nNext steps:")
        print("  1. Edit the prompts files to set actual point coordinates, OR")
        print("  2. Use select_prompts_interactive.py to visually select points:")
        print(f"     python select_prompts_interactive.py --folder <capture_folder>")
    
    # Show failed folders if any
    if failed > 0:
        print("\nFailed folders:")
        for result in results:
            if not result["success"]:
                print(f"  - {Path(result['folder']).name}: {result['message']}")


if __name__ == "__main__":
    main()