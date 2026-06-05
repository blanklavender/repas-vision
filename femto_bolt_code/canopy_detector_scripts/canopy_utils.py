#!/usr/bin/env python3
"""
Interactive Utility for Canopy Height Pipeline
===============================================

Tools for:
1. Selecting segmentation prompts (points or boxes) interactively
2. Verifying AprilTag detection
3. Checking depth data quality
4. Validating coordinate frame estimation

Usage:
    python canopy_utils.py select-prompts --image /path/to/color.png
    python canopy_utils.py check-tags --image /path/to/color.png
    python canopy_utils.py check-depth --depth /path/to/depth.npy --color /path/to/color.png
    python canopy_utils.py verify-frame --color /path/to/color.png --depth /path/to/depth.npy
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import json


class PromptSelector:
    """Interactive tool for selecting segmentation prompts."""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        
        self.display = self.image.copy()
        self.points = []
        self.labels = []
        self.box_start = None
        self.box_end = None
        self.mode = "point"  # "point" or "box"
        
        self.window_name = "Prompt Selector (Left=FG, Right=BG, 'b'=box mode, 'c'=clear, 'q'=quit)"
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "point":
                self.points.append([x, y])
                self.labels.append(1)  # Foreground
                self._draw_point((x, y), 1)
            elif self.mode == "box":
                self.box_start = (x, y)
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.mode == "point":
                self.points.append([x, y])
                self.labels.append(0)  # Background
                self._draw_point((x, y), 0)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mode == "box" and self.box_start is not None:
                temp = self.display.copy()
                cv2.rectangle(temp, self.box_start, (x, y), (255, 255, 0), 2)
                cv2.imshow(self.window_name, temp)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode == "box" and self.box_start is not None:
                self.box_end = (x, y)
                cv2.rectangle(self.display, self.box_start, self.box_end, (255, 255, 0), 2)
                cv2.imshow(self.window_name, self.display)
    
    def _draw_point(self, point, label):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(self.display, point, 8, color, -1)
        cv2.circle(self.display, point, 10, (255, 255, 255), 2)
        cv2.imshow(self.window_name, self.display)
    
    def run(self):
        """Run interactive prompt selection."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.display)
        
        print("\nPrompt Selector Instructions:")
        print("  Point Mode (default):")
        print("    - Left click: Add foreground point (green)")
        print("    - Right click: Add background point (red)")
        print("  Box Mode:")
        print("    - Press 'b' to switch to box mode")
        print("    - Click and drag to draw box")
        print("  Controls:")
        print("    - 'c': Clear all prompts")
        print("    - 'p': Switch to point mode")
        print("    - 'b': Switch to box mode")
        print("    - 'q' or ESC: Quit and print prompts")
        print()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                self.points = []
                self.labels = []
                self.box_start = None
                self.box_end = None
                self.display = self.image.copy()
                cv2.imshow(self.window_name, self.display)
                print("Cleared all prompts")
            
            elif key == ord('p'):
                self.mode = "point"
                print("Switched to point mode")
            
            elif key == ord('b'):
                self.mode = "box"
                print("Switched to box mode (click and drag)")
            
            elif key in [ord('q'), 27]:  # q or ESC
                break
        
        cv2.destroyAllWindows()
        
        # Print results
        print("\n" + "=" * 50)
        print("SELECTED PROMPTS")
        print("=" * 50)
        
        if self.points:
            print(f"\nPoint prompts ({len(self.points)} points):")
            print(f"  point_prompts = np.array({self.points})")
            print(f"  point_labels = np.array({self.labels})")
        
        if self.box_start and self.box_end:
            x1 = min(self.box_start[0], self.box_end[0])
            y1 = min(self.box_start[1], self.box_end[1])
            x2 = max(self.box_start[0], self.box_end[0])
            y2 = max(self.box_start[1], self.box_end[1])
            print(f"\nBox prompt:")
            print(f"  box_prompt = ({x1}, {y1}, {x2}, {y2})")
        
        return {
            "points": self.points,
            "labels": self.labels,
            "box": (x1, y1, x2, y2) if self.box_start and self.box_end else None
        }


class AprilTagChecker:
    """Tool for checking AprilTag detection."""
    
    def __init__(self, config=None):
        from pupil_apriltags import Detector
        
        self.detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.8,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )
    
    def check(self, image_path: str, show: bool = True):
        """Check AprilTag detection in image."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot load image: {image_path}")
            return
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        tags = self.detector.detect(gray)
        
        print(f"\nDetected {len(tags)} AprilTags:")
        
        output = image.copy()
        colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0)]
        
        for i, tag in enumerate(tags):
            color = colors[i % len(colors)]
            corners = tag.corners.astype(int)
            
            # Draw corners with indices
            corner_names = ["BL(0)", "BR(1)", "TR(2)", "TL(3)"]
            for j, (corner, name) in enumerate(zip(corners, corner_names)):
                cv2.circle(output, tuple(corner), 6, color, -1)
                cv2.putText(output, name, (corner[0] + 5, corner[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw outline
            cv2.polylines(output, [corners], True, color, 2)
            
            # Draw center and ID
            cx, cy = int(tag.center[0]), int(tag.center[1])
            cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(output, f"ID:{tag.tag_id}", (cx + 15, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            print(f"\n  Tag {tag.tag_id}:")
            print(f"    Center: ({cx}, {cy})")
            print(f"    Corners:")
            for j, (corner, name) in enumerate(zip(corners, corner_names)):
                print(f"      {name}: ({corner[0]}, {corner[1]})")
            print(f"    Decision margin: {tag.decision_margin:.2f}")
        
        if show:
            cv2.namedWindow("AprilTag Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("AprilTag Detection", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return tags


class DepthChecker:
    """Tool for checking depth data quality."""
    
    def __init__(self):
        pass
    
    def check(
        self, 
        depth_path: str, 
        color_path: str = None,
        show: bool = True
    ):
        """Check depth data quality and statistics."""
        
        # Load depth
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
            is_meters = True
            print(f"Loaded depth from NPY (assumed meters)")
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            is_meters = False
            print(f"Loaded depth from PNG (raw uint16)")
        
        if depth is None:
            print(f"Cannot load depth: {depth_path}")
            return
        
        print(f"\nDepth image shape: {depth.shape}")
        print(f"Depth dtype: {depth.dtype}")
        
        # Convert to meters for statistics
        if is_meters:
            depth_m = depth.astype(np.float32)
        else:
            depth_m = depth.astype(np.float32) * 0.001
        
        # Statistics
        valid_mask = (depth_m > 0.1) & (depth_m < 10.0)
        valid_depth = depth_m[valid_mask]
        
        print(f"\nDepth Statistics:")
        print(f"  Valid pixels: {valid_mask.sum()} / {depth_m.size} ({100*valid_mask.sum()/depth_m.size:.1f}%)")
        
        if len(valid_depth) > 0:
            print(f"  Min depth: {valid_depth.min():.3f} m")
            print(f"  Max depth: {valid_depth.max():.3f} m")
            print(f"  Mean depth: {valid_depth.mean():.3f} m")
            print(f"  Median depth: {np.median(valid_depth):.3f} m")
            print(f"  Std depth: {valid_depth.std():.3f} m")
        
        if show:
            # Create visualization
            vis = np.zeros((depth_m.shape[0], depth_m.shape[1], 3), dtype=np.uint8)
            
            # Normalize for visualization
            norm = np.zeros_like(depth_m)
            norm[valid_mask] = (depth_m[valid_mask] - 0.3) / (3.0 - 0.3)  # Normalize 0.3-3m
            norm = np.clip(norm * 255, 0, 255).astype(np.uint8)
            
            vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            vis[~valid_mask] = [0, 0, 0]  # Black for invalid
            
            # Side by side with color if available
            if color_path:
                color = cv2.imread(color_path)
                if color is not None:
                    # Resize depth vis to match color
                    if vis.shape[:2] != color.shape[:2]:
                        vis = cv2.resize(vis, (color.shape[1], color.shape[0]))
                    
                    combined = np.hstack([color, vis])
                    cv2.namedWindow("Color | Depth", cv2.WINDOW_NORMAL)
                    cv2.imshow("Color | Depth", combined)
                else:
                    cv2.namedWindow("Depth Visualization", cv2.WINDOW_NORMAL)
                    cv2.imshow("Depth Visualization", vis)
            else:
                cv2.namedWindow("Depth Visualization", cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Visualization", vis)
            
            # Interactive depth probe
            print("\nClick on depth image to probe depth values. Press 'q' to quit.")
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Adjust x if showing combined image
                    if color_path and color is not None:
                        if x >= color.shape[1]:
                            dx = x - color.shape[1]
                            # Scale to original depth dimensions
                            scale_x = depth.shape[1] / color.shape[1]
                            scale_y = depth.shape[0] / color.shape[0]
                            px = int(dx * scale_x)
                            py = int(y * scale_y)
                        else:
                            px = int(x * depth.shape[1] / color.shape[1])
                            py = int(y * depth.shape[0] / color.shape[0])
                    else:
                        px, py = x, y
                    
                    px = max(0, min(px, depth.shape[1] - 1))
                    py = max(0, min(py, depth.shape[0] - 1))
                    
                    d = depth_m[py, px]
                    print(f"  Pixel ({px}, {py}): {d:.4f} m ({d*100:.2f} cm)")
            
            cv2.setMouseCallback("Color | Depth" if color_path else "Depth Visualization", mouse_callback)
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    break
            
            cv2.destroyAllWindows()


class FrameVerifier:
    """Tool for verifying coordinate frame estimation."""
    
    def __init__(self, config=None):
        self.config = config
        
        # Default camera intrinsics (update for your camera)
        self.fx = config.fx if config else 604.0
        self.fy = config.fy if config else 604.0
        self.cx = config.cx if config else 640.0
        self.cy = config.cy if config else 360.0
        self.depth_scale = 0.001
    
    def pixel_to_3d(self, pixel, depth_image):
        x, y = int(pixel[0]), int(pixel[1])
        h, w = depth_image.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        
        # Median over window
        for half in [2, 5, 10]:
            patch = depth_image[max(0,y-half):min(h,y+half+1), max(0,x-half):min(w,x+half+1)]
            if depth_image.dtype == np.float32 or depth_image.dtype == np.float64:
                valid = patch[(patch > 0.1) & (patch < 10.0)]
                if len(valid) > 0:
                    depth_m = float(np.median(valid))
                    break
            else:
                valid = patch[patch > 100]
                if len(valid) > 0:
                    depth_m = float(np.median(valid)) * self.depth_scale
                    break
        else:
            return None
        
        X = (x - self.cx) * depth_m / self.fx
        Y = (y - self.cy) * depth_m / self.fy
        Z = depth_m
        
        return np.array([X, Y, Z])
    
    def verify(self, color_path: str, depth_path: str, show: bool = True):
        """Verify coordinate frame estimation from tags."""
        from pupil_apriltags import Detector
        
        # Load images
        image = cv2.imread(color_path)
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        
        if image is None or depth is None:
            print("Cannot load images")
            return
        
        # Detect tags
        detector = Detector(families="tag36h11", nthreads=4)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        tags = detector.detect(gray)
        tag_dict = {t.tag_id: t for t in tags}
        
        if 21 not in tag_dict or 16 not in tag_dict:
            print(f"Need tags 21 and 16, found: {list(tag_dict.keys())}")
            return
        
        tag_21 = tag_dict[21]
        tag_16 = tag_dict[16]
        
        # Get corner 3D positions
        # Corner indices: [0]=BL, [1]=BR, [2]=TR, [3]=TL
        TOP_LEFT = 3
        TOP_RIGHT = 2
        
        origin_px = tag_21.corners[TOP_LEFT]
        x_end_px = tag_16.corners[TOP_RIGHT]
        
        origin_3d = self.pixel_to_3d(origin_px, depth)
        x_end_3d = self.pixel_to_3d(x_end_px, depth)
        
        if origin_3d is None or x_end_3d is None:
            print("Cannot get 3D positions from depth")
            return
        
        print("\n" + "=" * 50)
        print("COORDINATE FRAME VERIFICATION")
        print("=" * 50)
        
        print(f"\nOrigin (Tag 21 top-left):")
        print(f"  Pixel: ({origin_px[0]:.1f}, {origin_px[1]:.1f})")
        print(f"  3D: ({origin_3d[0]:.4f}, {origin_3d[1]:.4f}, {origin_3d[2]:.4f}) m")
        
        print(f"\nX-axis endpoint (Tag 16 top-right):")
        print(f"  Pixel: ({x_end_px[0]:.1f}, {x_end_px[1]:.1f})")
        print(f"  3D: ({x_end_3d[0]:.4f}, {x_end_3d[1]:.4f}, {x_end_3d[2]:.4f}) m")
        
        # Calculate X-axis
        x_axis_raw = x_end_3d - origin_3d
        baseline = np.linalg.norm(x_axis_raw)
        x_axis = x_axis_raw / baseline
        
        print(f"\nBaseline distance: {baseline*100:.2f} cm")
        print(f"X-axis (normalized): ({x_axis[0]:.4f}, {x_axis[1]:.4f}, {x_axis[2]:.4f})")
        
        # Estimate Z from tag corners
        tl = self.pixel_to_3d(tag_21.corners[TOP_LEFT], depth)
        tr = self.pixel_to_3d(tag_21.corners[TOP_RIGHT], depth)
        bl = self.pixel_to_3d(tag_21.corners[0], depth)
        
        if all(p is not None for p in [tl, tr, bl]):
            tag_horiz = tr - tl
            tag_vert = bl - tl
            z_raw = np.cross(tag_horiz, tag_vert)
            if z_raw[2] < 0:
                z_raw = -z_raw
            z_axis = z_raw / np.linalg.norm(z_raw)
            
            # Y-axis
            y_raw = np.cross(z_axis, x_axis)
            y_axis = y_raw / np.linalg.norm(y_raw)
            if y_axis[1] > 0:  # Should point up (negative Y in camera)
                y_axis = -y_axis
                z_axis = -z_axis
            
            # Re-orthogonalize
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            
            print(f"\nY-axis (up): ({y_axis[0]:.4f}, {y_axis[1]:.4f}, {y_axis[2]:.4f})")
            print(f"Z-axis (into): ({z_axis[0]:.4f}, {z_axis[1]:.4f}, {z_axis[2]:.4f})")
            
            # Verify orthonormality
            print(f"\nOrthonormality check:")
            print(f"  |X| = {np.linalg.norm(x_axis):.6f}")
            print(f"  |Y| = {np.linalg.norm(y_axis):.6f}")
            print(f"  |Z| = {np.linalg.norm(z_axis):.6f}")
            print(f"  X·Y = {np.dot(x_axis, y_axis):.6f}")
            print(f"  X·Z = {np.dot(x_axis, z_axis):.6f}")
            print(f"  Y·Z = {np.dot(y_axis, z_axis):.6f}")
        
        if show:
            output = image.copy()
            
            # Draw tags
            for tag in [tag_21, tag_16]:
                corners = tag.corners.astype(int)
                color = (0, 255, 255) if tag.tag_id == 21 else (255, 0, 255)
                cv2.polylines(output, [corners], True, color, 2)
                cv2.putText(output, f"ID:{tag.tag_id}", 
                           (int(tag.center[0]) + 10, int(tag.center[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw origin
            cv2.circle(output, tuple(origin_px.astype(int)), 8, (0, 255, 0), -1)
            
            # Draw baseline
            cv2.line(output, tuple(origin_px.astype(int)), tuple(x_end_px.astype(int)), (0, 0, 255), 2)
            
            # Draw axes at origin
            def project(p):
                if p[2] <= 0:
                    return None
                u = int(self.fx * p[0] / p[2] + self.cx)
                v = int(self.fy * p[1] / p[2] + self.cy)
                return (u, v)
            
            origin_px_int = project(origin_3d)
            if origin_px_int and 'x_axis' in dir():
                for axis, color, label in [(x_axis, (0,0,255), "X"), 
                                           (y_axis, (0,255,0), "Y"),
                                           (z_axis, (255,0,0), "Z")]:
                    end = origin_3d + 0.05 * axis
                    end_px = project(end)
                    if end_px:
                        cv2.arrowedLine(output, origin_px_int, end_px, color, 2, tipLength=0.2)
                        cv2.putText(output, label, end_px, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            cv2.namedWindow("Coordinate Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Coordinate Frame", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Canopy Height Pipeline Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # select-prompts
    sp_prompts = subparsers.add_parser("select-prompts", help="Interactive prompt selection")
    sp_prompts.add_argument("--image", "-i", required=True, help="Path to color image")
    
    # check-tags
    sp_tags = subparsers.add_parser("check-tags", help="Check AprilTag detection")
    sp_tags.add_argument("--image", "-i", required=True, help="Path to color image")
    
    # check-depth
    sp_depth = subparsers.add_parser("check-depth", help="Check depth data quality")
    sp_depth.add_argument("--depth", "-d", required=True, help="Path to depth (PNG or NPY)")
    sp_depth.add_argument("--color", "-c", help="Optional color image for side-by-side view")
    
    # verify-frame
    sp_frame = subparsers.add_parser("verify-frame", help="Verify coordinate frame estimation")
    sp_frame.add_argument("--color", "-c", required=True, help="Path to color image")
    sp_frame.add_argument("--depth", "-d", required=True, help="Path to depth (PNG or NPY)")
    sp_frame.add_argument("--fx", type=float, default=604.0, help="Focal length X")
    sp_frame.add_argument("--fy", type=float, default=604.0, help="Focal length Y")
    sp_frame.add_argument("--cx", type=float, default=640.0, help="Principal point X")
    sp_frame.add_argument("--cy", type=float, default=360.0, help="Principal point Y")
    
    args = parser.parse_args()
    
    if args.command == "select-prompts":
        selector = PromptSelector(args.image)
        selector.run()
    
    elif args.command == "check-tags":
        checker = AprilTagChecker()
        checker.check(args.image)
    
    elif args.command == "check-depth":
        checker = DepthChecker()
        checker.check(args.depth, args.color)
    
    elif args.command == "verify-frame":
        # Create a simple config-like object
        class SimpleConfig:
            pass
        config = SimpleConfig()
        config.fx = args.fx
        config.fy = args.fy
        config.cx = args.cx
        config.cy = args.cy
        
        verifier = FrameVerifier(config)
        verifier.verify(args.color, args.depth)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()