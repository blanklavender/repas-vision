#!/usr/bin/env python3
"""
Batch Process All Captures
===========================
Automatically runs temp_canopy_script_modified.py on all capture subfolders.

Usage:
    python process_all_captures.py captures/ --baseline-cm 45
    python process_all_captures.py /path/to/captures --baseline-cm 45 --no-show

    python process_all_captures.py ../../../../captures/canopy_test_captures_62 --baseline-cm 65
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime


def setup_logging(root_dir: Path):
    """
    Setup logging to file.
    
    Returns:
        Path to log file
    """
    logs_dir = root_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"process_all_{timestamp}.log"
    
    return log_file


def log_message(message: str, log_file: Path = None, print_to_console: bool = True):
    """Log message to file and optionally console."""
    if print_to_console:
        print(message)
    
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + '\n')


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


def check_capture_folder(folder: Path) -> Dict[str, Any]:
    """
    Check if a capture folder has all required files.
    
    Args:
        folder: Path to capture folder
    
    Returns:
        Dict with status and missing files
    """
    folder_name = folder.name
    if not folder_name.startswith("capture_"):
        return {
            "valid": False,
            "message": "Invalid folder name (doesn't start with 'capture_')",
            "missing_files": []
        }
    
    timestamp = folder_name.replace("capture_", "")
    
    # Check for required files
    color_path = folder / f"color_{timestamp}.png"
    depth_path = folder / f"aligned_depth_m_{timestamp}.npy"
    prompts_path = folder / f"prompts_{timestamp}.json"
    
    missing_files = []
    if not color_path.exists():
        missing_files.append(color_path.name)
    if not depth_path.exists():
        missing_files.append(depth_path.name)
    if not prompts_path.exists():
        missing_files.append(prompts_path.name)
    
    if missing_files:
        return {
            "valid": False,
            "message": f"Missing files: {', '.join(missing_files)}",
            "missing_files": missing_files
        }
    
    return {
        "valid": True,
        "message": "All required files present",
        "missing_files": []
    }


def parse_error_from_output(stdout: str, stderr: str) -> str:
    """
    Parse specific error messages from script output.
    
    Args:
        stdout: Standard output
        stderr: Standard error
    
    Returns:
        Human-readable error message
    """
    # Combine both outputs
    full_output = stdout + "\n" + stderr
    
    # Check for specific error patterns
    if "Tag 21 not detected" in full_output or "Tag 16 not detected" in full_output:
        return "AprilTag detection failed - tags not visible in image"
    
    if "error" in full_output.lower() and "apriltag" in full_output.lower():
        return "AprilTag detection error"
    
    if "No depth at origin" in full_output:
        return "No valid depth data at AprilTag location"
    
    if "Failed to establish coordinate frame" in full_output:
        return "Could not establish coordinate frame from AprilTags"
    
    if "Empty segmentation mask" in full_output:
        return "Segmentation failed - no lettuce detected with given prompts"
    
    if "Could not find canopy top point" in full_output:
        return "No valid depth data for canopy measurement"
    
    if "Cannot load color image" in full_output:
        return "Failed to load color image"
    
    if "Cannot load depth image" in full_output:
        return "Failed to load depth image"
    
    if "Prompts file not found" in full_output:
        return "Prompts file missing"
    
    if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
        # Extract module name
        if "No module named" in stderr:
            module = stderr.split("No module named")[1].split()[0].strip("'\"")
            return f"Missing Python module: {module}"
        return "Missing Python dependencies"
    
    if "FileNotFoundError" in stderr:
        return "File not found error"
    
    # Generic error
    if stderr:
        # Return first line of stderr
        first_line = stderr.split('\n')[0].strip()
        if first_line:
            return first_line[:100]  # Limit length
    
    return "Unknown error - check log file for details"


def process_capture(
    folder: Path,
    baseline_cm: float = None,
    efficient_sam_path: str = None,
    no_show: bool = True,
    no_save: bool = False,
    fx: float = None,
    fy: float = None,
    cx: float = None,
    cy: float = None,
    log_file: Path = None
) -> Dict[str, Any]:
    """
    Process a single capture folder.
    
    Returns:
        Dict with processing status and info
    """
    result = {
        "folder": str(folder),
        "success": False,
        "message": "",
        "duration": 0.0,
        "stdout": "",
        "stderr": ""
    }
    
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        "./../temp_canopy_script_better.py",
        "--folder", str(folder)
    ]
    
    if baseline_cm is not None:
        cmd.extend(["--baseline-cm", str(baseline_cm)])
    
    if efficient_sam_path:
        cmd.extend(["--efficient-sam", efficient_sam_path])
    
    if no_show:
        cmd.append("--no-show")
    
    if no_save:
        cmd.append("--no-save")
    
    # Add camera intrinsics if provided
    if fx is not None:
        cmd.extend(["--fx", str(fx)])
    if fy is not None:
        cmd.extend(["--fy", str(fy)])
    if cx is not None:
        cmd.extend(["--cx", str(cx)])
    if cy is not None:
        cmd.extend(["--cy", str(cy)])
    
    # Log command
    cmd_str = " ".join(cmd)
    log_message(f"\nCommand: {cmd_str}", log_file, print_to_console=False)
    
    # Run command
    start_time = time.time()
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        result["duration"] = duration
        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        
        # Log full output to file
        log_message(f"\n--- STDOUT ---\n{process.stdout}", log_file, print_to_console=False)
        log_message(f"\n--- STDERR ---\n{process.stderr}", log_file, print_to_console=False)
        
        if process.returncode == 0:
            result["success"] = True
            result["message"] = "Processing complete"
        else:
            result["success"] = False
            # Parse error message
            error_msg = parse_error_from_output(process.stdout, process.stderr)
            result["message"] = error_msg
    
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        result["duration"] = duration
        result["message"] = "Processing timed out (>5 minutes)"
        log_message(f"\nTimeout after {duration:.1f}s", log_file, print_to_console=False)
    
    except Exception as e:
        duration = time.time() - start_time
        result["duration"] = duration
        result["message"] = f"Error: {str(e)}"
        log_message(f"\nException: {str(e)}", log_file, print_to_console=False)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all capture folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process all captures with baseline calibration
  python process_all_captures.py captures/ --baseline-cm 45
  
  # Process with custom EfficientSAM path
  python process_all_captures.py captures/ --baseline-cm 45 --efficient-sam /path/to/EfficientSAM
  
  # Show visualizations (slower)
  python process_all_captures.py captures/ --baseline-cm 45 --show
  
  # Skip saving (useful for testing)
  python process_all_captures.py captures/ --baseline-cm 45 --no-save

The script will:
1. Find all capture_* subfolders in the provided directory
2. Check each folder has required files (color, depth, prompts)
3. Process each valid folder with temp_canopy_script_modified.py
4. Report success/failure for each capture
5. Save detailed log to logs/process_all_TIMESTAMP.log

Error Handling:
- AprilTag detection failures are caught and reported
- Missing files are detected before processing
- Each capture processes independently (failures don't stop others)
- Full logs saved for debugging
        """
    )
    
    parser.add_argument("folder", type=str,
                        help="Path to folder containing capture_* subfolders")
    parser.add_argument("--baseline-cm", type=float, default=None,
                        help="Known physical baseline distance in centimeters")
    parser.add_argument("--efficient-sam", type=str, default=None,
                        help="Path to EfficientSAM repo (default: ../../../EfficientSAM)")
    parser.add_argument("--show", action="store_true",
                        help="Show visualization windows (default: hidden for batch)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results (useful for testing)")
    
    # Camera intrinsics
    parser.add_argument("--fx", type=float, default=None,
                        help="Focal length X")
    parser.add_argument("--fy", type=float, default=None,
                        help="Focal length Y")
    parser.add_argument("--cx", type=float, default=None,
                        help="Principal point X")
    parser.add_argument("--cy", type=float, default=None,
                        help="Principal point Y")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full output for each capture")
    
    args = parser.parse_args()
    
    # Find capture folders
    root_dir = Path(args.folder)
    try:
        capture_folders = find_capture_folders(root_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return 1
    
    if len(capture_folders) == 0:
        print(f"[WARNING] No capture_* folders found in: {root_dir}")
        return 1
    
    # Setup logging
    log_file = setup_logging(root_dir)
    
    print("\n" + "=" * 70)
    print("BATCH PROCESS ALL CAPTURES")
    print("=" * 70)
    print(f"Root folder: {root_dir.absolute()}")
    print(f"Found {len(capture_folders)} capture folders")
    print(f"Log file: {log_file}")
    if args.baseline_cm:
        print(f"Baseline: {args.baseline_cm} cm")
    else:
        print("Baseline: Not provided (no calibration)")
    print(f"Show visualizations: {'Yes' if args.show else 'No'}")
    print(f"Save results: {'No' if args.no_save else 'Yes'}")
    print()
    
    # Log header
    log_message("=" * 70, log_file, print_to_console=False)
    log_message("BATCH PROCESS ALL CAPTURES", log_file, print_to_console=False)
    log_message(f"Started: {datetime.now().isoformat()}", log_file, print_to_console=False)
    log_message("=" * 70, log_file, print_to_console=False)
    
    # Check each folder for required files
    print("Checking folders...")
    log_message("\nChecking folders...", log_file, print_to_console=False)
    
    valid_folders = []
    invalid_folders = []
    
    for folder in capture_folders:
        status = check_capture_folder(folder)
        if status["valid"]:
            valid_folders.append(folder)
            msg = f"  ✓ {folder.name}"
            print(msg)
            log_message(msg, log_file, print_to_console=False)
        else:
            invalid_folders.append((folder, status))
            msg = f"  ✗ {folder.name} - {status['message']}"
            print(msg)
            log_message(msg, log_file, print_to_console=False)
    
    print(f"\nValid folders: {len(valid_folders)}")
    print(f"Invalid folders: {len(invalid_folders)}")
    
    if len(valid_folders) == 0:
        print("\n[ERROR] No valid capture folders to process")
        print(f"\nLog saved to: {log_file}")
        return 1
    
    # Ask for confirmation
    print(f"\nReady to process {len(valid_folders)} capture(s).")
    response = input("Continue? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("Cancelled.")
        return 0
    
    # Process each folder
    print("\n" + "=" * 70)
    print("PROCESSING")
    print("=" * 70)
    log_message("\n" + "=" * 70, log_file, print_to_console=False)
    log_message("PROCESSING", log_file, print_to_console=False)
    log_message("=" * 70, log_file, print_to_console=False)
    
    results = []
    for i, folder in enumerate(valid_folders, 1):
        folder_name = folder.name
        print(f"\n[{i}/{len(valid_folders)}] Processing: {folder_name}")
        log_message(f"\n[{i}/{len(valid_folders)}] Processing: {folder_name}", log_file, print_to_console=False)
        
        result = process_capture(
            folder,
            baseline_cm=args.baseline_cm,
            efficient_sam_path=args.efficient_sam,
            no_show=not args.show,
            no_save=args.no_save,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            log_file=log_file
        )
        
        results.append(result)
        
        # Print status
        if result["success"]:
            msg = f"  ✓ Success ({result['duration']:.1f}s)"
            print(msg)
            log_message(msg, log_file, print_to_console=False)
        else:
            msg = f"  ✗ Failed: {result['message']}"
            print(msg)
            log_message(msg, log_file, print_to_console=False)
            
            # Show additional details if verbose
            if args.verbose and (result["stdout"] or result["stderr"]):
                print("\n  --- Output ---")
                if result["stdout"]:
                    print(result["stdout"][:500])  # First 500 chars
                if result["stderr"]:
                    print("\n  --- Errors ---")
                    print(result["stderr"][:500])
                print()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    total_time = sum(r["duration"] for r in results)
    
    summary = f"""
Total processed: {len(results)}
Successful: {successful}
Failed: {failed}
Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)
"""
    
    if successful > 0:
        summary += f"Average time per capture: {total_time/successful:.1f}s\n"
    
    print(summary)
    log_message("\n" + "=" * 70, log_file, print_to_console=False)
    log_message("SUMMARY", log_file, print_to_console=False)
    log_message("=" * 70, log_file, print_to_console=False)
    log_message(summary, log_file, print_to_console=False)
    
    # Show failed folders if any
    if failed > 0:
        print("\nFailed folders:")
        log_message("\nFailed folders:", log_file, print_to_console=False)
        
        for result in results:
            if not result["success"]:
                folder_name = Path(result['folder']).name
                msg = f"  - {folder_name}: {result['message']}"
                print(msg)
                log_message(msg, log_file, print_to_console=False)
    
    if not args.no_save and successful > 0:
        msg = f"\nResults saved in each capture's viz_output_v2_*/ folder"
        print(msg)
        log_message(msg, log_file, print_to_console=False)
        print(f"Run compile_results.py to create a summary CSV")
    
    print(f"\nFull log saved to: {log_file}")
    log_message(f"\nCompleted: {datetime.now().isoformat()}", log_file, print_to_console=False)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all capture folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process all captures with baseline calibration
  python process_all_captures.py captures/ --baseline-cm 45
  
  # Process with custom EfficientSAM path
  python process_all_captures.py captures/ --baseline-cm 45 --efficient-sam /path/to/EfficientSAM
  
  # Show visualizations (slower)
  python process_all_captures.py captures/ --baseline-cm 45 --show
  
  # Skip saving (useful for testing)
  python process_all_captures.py captures/ --baseline-cm 45 --no-save

The script will:
1. Find all capture_* subfolders in the provided directory
2. Check each folder has required files (color, depth, prompts)
3. Process each valid folder with temp_canopy_script_modified.py
4. Report success/failure for each capture
        """
    )
    
    parser.add_argument("folder", type=str,
                        help="Path to folder containing capture_* subfolders")
    parser.add_argument("--baseline-cm", type=float, default=None,
                        help="Known physical baseline distance in centimeters")
    parser.add_argument("--efficient-sam", type=str, default=None,
                        help="Path to EfficientSAM repo (default: ../../../EfficientSAM)")
    parser.add_argument("--show", action="store_true",
                        help="Show visualization windows (default: hidden for batch)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results (useful for testing)")
    
    # Camera intrinsics
    parser.add_argument("--fx", type=float, default=None,
                        help="Focal length X")
    parser.add_argument("--fy", type=float, default=None,
                        help="Focal length Y")
    parser.add_argument("--cx", type=float, default=None,
                        help="Principal point X")
    parser.add_argument("--cy", type=float, default=None,
                        help="Principal point Y")
    
    args = parser.parse_args()
    
    # Find capture folders
    root_dir = Path(args.folder)
    try:
        capture_folders = find_capture_folders(root_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return 1
    
    if len(capture_folders) == 0:
        print(f"[WARNING] No capture_* folders found in: {root_dir}")
        return 1
    
    print("\n" + "=" * 70)
    print("BATCH PROCESS ALL CAPTURES")
    print("=" * 70)
    print(f"Root folder: {root_dir.absolute()}")
    print(f"Found {len(capture_folders)} capture folders")
    if args.baseline_cm:
        print(f"Baseline: {args.baseline_cm} cm")
    else:
        print("Baseline: Not provided (no calibration)")
    print(f"Show visualizations: {'Yes' if args.show else 'No'}")
    print(f"Save results: {'No' if args.no_save else 'Yes'}")
    print()
    
    # Check each folder for required files
    print("Checking folders...")
    valid_folders = []
    invalid_folders = []
    
    for folder in capture_folders:
        status = check_capture_folder(folder)
        if status["valid"]:
            valid_folders.append(folder)
            print(f"  ✓ {folder.name}")
        else:
            invalid_folders.append((folder, status))
            print(f"  ✗ {folder.name} - {status['message']}")
    
    print(f"\nValid folders: {len(valid_folders)}")
    print(f"Invalid folders: {len(invalid_folders)}")
    
    if len(valid_folders) == 0:
        print("\n[ERROR] No valid capture folders to process")
        return 1
    
    # Ask for confirmation
    print(f"\nReady to process {len(valid_folders)} capture(s).")
    response = input("Continue? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("Cancelled.")
        return 0
    
    # Process each folder
    print("\n" + "=" * 70)
    print("PROCESSING")
    print("=" * 70)
    
    results = []
    for i, folder in enumerate(valid_folders, 1):
        folder_name = folder.name
        print(f"\n[{i}/{len(valid_folders)}] Processing: {folder_name}")
        
        result = process_capture(
            folder,
            baseline_cm=args.baseline_cm,
            efficient_sam_path=args.efficient_sam,
            no_show=not args.show,
            no_save=args.no_save,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy
        )
        
        results.append(result)
        
        # Print status
        if result["success"]:
            print(f"  ✓ Success ({result['duration']:.1f}s)")
        else:
            print(f"  ✗ Failed: {result['message']}")
            if "stderr" in result and result["stderr"]:
                print(f"  Error output: {result['stderr'][:200]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    total_time = sum(r["duration"] for r in results)
    
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    if successful > 0:
        print(f"Average time per capture: {total_time/successful:.1f}s")
    
    # Show failed folders if any
    if failed > 0:
        print("\nFailed folders:")
        for result in results:
            if not result["success"]:
                folder_name = Path(result['folder']).name
                print(f"  - {folder_name}: {result['message']}")
    
    if not args.no_save and successful > 0:
        print(f"\nResults saved in each capture's viz_output_v2_*/ folder")
        print(f"Run compile_results.py to create a summary CSV")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())