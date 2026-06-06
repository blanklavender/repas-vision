import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import math

def get_timestamp():
    """Return the current timestamp in YYYY-MM-DDTHHMMSS format."""
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")

def detect_rotate_aluminum_bar_edges(image):
    """
    Detect aluminum bar edges and rotate image to align bar horizontally.
    
    Returns:
        rotation_info: Dictionary with rotation details and bar position
        bar_edges: List of detected bar edge lines
        processed_image: Image with all detected edges drawn
        selected_edge_image: Image with selected edge highlighted
        rotated_image: The rotated image
    """
    processed_image = image.copy()
    selected_edge_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 
        rho=1,           
        theta=np.pi/180, 
        threshold=50,    
        minLineLength=50,  
        maxLineGap=10    
    )
    
    # Filter and visualize lines
    bar_edges = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle_signed = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
            angle = abs(angle_signed)
            line_coverage = length / image.shape[1]
            
            # Filtering criteria: lines at least 10% of the image width and nearly horizontal
            if (length > image.shape[1] * 0.1 and 
                line_coverage >= 0.1 and  
                (angle < 20 or angle > 160)):
                
                cv2.line(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                bar_edges.append({
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'length': length,
                    'angle': angle_signed,  
                    'coverage': line_coverage
                })
    
    # First detected edge is used for rotation
    if len(bar_edges) > 0:
        best_line = bar_edges[0]
        
        # Rotate the image by the angle of the detected edge
        rotation_angle = best_line['angle']
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated_image = cv2.warpAffine(
            image, 
            rotation_matrix, 
            (w, h), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)  
        )
        
        rotation_info = {
            'original_angle': best_line['angle'],
            'rotation_angle': rotation_angle,
            'line_points': (best_line['start'], best_line['end']),
            'line_coverage': best_line['coverage'],
            'rotation_matrix': rotation_matrix
        }

        cv2.line(selected_edge_image, (best_line['start'][0], best_line['start'][1]), 
                (best_line['end'][0], best_line['end'][1]), (0, 0, 255), 2)
        
        return rotation_info, bar_edges, processed_image, selected_edge_image, rotated_image
    else:
        print("No suitable edge detected for rotation.")
        return None, [], processed_image, selected_edge_image, image

def remove_background_grabcut(image):
    """Remove background using GrabCut initialized with green mask."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Thresholds for Green Color
    lower_green = np.array([35, 40, 40]) 
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    grabcut_mask = np.where(green_mask==255, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')
    height, width = image.shape[:2]
    rect = (1, 1, width-2, height-2)
    
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    cv2.grabCut(image, grabcut_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((grabcut_mask==cv2.GC_FGD) | (grabcut_mask==cv2.GC_PR_FGD), 1, 0).astype('uint8')
    result = image * final_mask[:, :, np.newaxis]
    
    return result, final_mask

def apply_green_mask(image):
    """Apply green color mask to extract plant features."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 80, 30])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up the mask with morphological operations
    kernel = np.ones((3,3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    return image, green_mask

def canopy_level_mark(image):
    """Find the highest point of the canopy and return y-coordinate."""
    canopy_image = image.copy()

    mask = np.any(canopy_image != 0, axis=2)

    if np.any(mask):
        ys, xs = np.where(mask)
        print(f"Found {len(ys)} plant pixels")

        # Find the minimum y coordinate (the highest plant pixel along the vertical axis)
        canopy_y = int(np.min(ys))
        indices = np.where(ys == canopy_y)[0]
        canopy_x = int(np.median(xs[indices]))

        return canopy_y, canopy_x
    else:
        print("No plant pixels detected!")
        return None, None

def draw_canopy_visualization(original_image, rotated_image, canopy_x, canopy_y, 
                               bar_pixel, plant_height_m, canopy_3d, bar_3d):
    """
    Draw canopy line, bar line, and height measurement on the image.
    
    Args:
        original_image: Original unrotated image
        rotated_image: Rotated image for visualization
        canopy_x, canopy_y: Canopy position in rotated image
        bar_pixel: (x, y) pixel position of bar in original image
        plant_height_m: Measured plant height in meters
        canopy_3d: (X, Y, Z) of canopy point
        bar_3d: (X, Y, Z) of bar point
    """
    rotated_vis = rotated_image.copy()
    height, width = rotated_vis.shape[:2]
    
    # Draw horizontal red line at canopy height
    cv2.line(rotated_vis, (0, canopy_y), (width - 1, canopy_y), (0, 0, 255), 2)
    
    # Draw blue circle at the canopy point
    cv2.circle(rotated_vis, (canopy_x, canopy_y), 8, (255, 0, 0), -1)
    
    # Draw bar reference line (green) if we have bar position
    if bar_pixel is not None:
        bar_x, bar_y = bar_pixel
        cv2.line(rotated_vis, (0, bar_y), (width - 1, bar_y), (0, 255, 0), 2)
        cv2.circle(rotated_vis, (bar_x, bar_y), 8, (0, 255, 0), -1)
        
        # Draw vertical measurement line connecting canopy to bar
        mid_x = (canopy_x + bar_x) // 2
        cv2.line(rotated_vis, (mid_x, canopy_y), (mid_x, bar_y), (255, 255, 0), 2)
        
        # Draw arrow heads
        arrow_size = 10
        # Top arrow (at canopy)
        cv2.line(rotated_vis, (mid_x, canopy_y), (mid_x - arrow_size, canopy_y + arrow_size), (255, 255, 0), 2)
        cv2.line(rotated_vis, (mid_x, canopy_y), (mid_x + arrow_size, canopy_y + arrow_size), (255, 255, 0), 2)
        # Bottom arrow (at bar)
        cv2.line(rotated_vis, (mid_x, bar_y), (mid_x - arrow_size, bar_y - arrow_size), (255, 255, 0), 2)
        cv2.line(rotated_vis, (mid_x, bar_y), (mid_x + arrow_size, bar_y - arrow_size), (255, 255, 0), 2)
    
    # Add text labels
    plant_height_cm = plant_height_m * 100
    
    # Canopy label
    canopy_label = f"Canopy Y: {canopy_3d[1]:.3f}m"
    cv2.putText(rotated_vis, canopy_label, (canopy_x + 15, canopy_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(rotated_vis, canopy_label, (canopy_x + 15, canopy_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Bar label
    if bar_3d is not None:
        bar_label = f"Bar Y: {bar_3d[1]:.3f}m"
        cv2.putText(rotated_vis, bar_label, (bar_pixel[0] + 15, bar_pixel[1] + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(rotated_vis, bar_label, (bar_pixel[0] + 15, bar_pixel[1] + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Plant height label (large, prominent)
    height_label = f"PLANT HEIGHT: {plant_height_cm:.1f} cm"
    label_y = (canopy_y + bar_pixel[1]) // 2 if bar_pixel else canopy_y + 50
    cv2.putText(rotated_vis, height_label, (20, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(rotated_vis, height_label, (20, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # Add depth info
    depth_label = f"Canopy Z: {canopy_3d[2]:.3f}m | Bar Z: {bar_3d[2]:.3f}m"
    cv2.putText(rotated_vis, depth_label, (20, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(rotated_vis, depth_label, (20, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return rotated_vis

def inverse_rotate_point(x, y, rotation_matrix, image_shape):
    """Apply inverse rotation to get original image coordinates."""
    h, w = image_shape[:2]
    center = (w // 2, h // 2)
    
    # Get the inverse rotation matrix
    inv_rotation_matrix = cv2.invertAffineTransform(rotation_matrix)
    
    # Convert point to homogeneous coordinates
    point = np.array([[[x, y]]], dtype=np.float32)
    
    # Apply inverse transform
    original_point = cv2.transform(point, inv_rotation_matrix)
    
    orig_x = int(original_point[0][0][0])
    orig_y = int(original_point[0][0][1])
    
    return orig_x, orig_y

def rotate_point(x, y, rotation_matrix):
    """Apply rotation to transform original coordinates to rotated coordinates."""
    point = np.array([[[x, y]]], dtype=np.float32)
    rotated_point = cv2.transform(point, rotation_matrix)
    
    rot_x = int(rotated_point[0][0][0])
    rot_y = int(rotated_point[0][0][1])
    
    return rot_x, rot_y

def deproject_pixel_to_point(intrinsics, pixel, depth_value):
    """
    Convert 2D pixel coordinates + depth to 3D point using camera intrinsics.
    
    Args:
        intrinsics: Camera intrinsics object from RealSense
        pixel: Tuple (x, y) pixel coordinates
        depth_value: Depth value at that pixel in meters
    
    Returns:
        Tuple (X, Y, Z) representing 3D coordinates in meters relative to camera
    """
    x, y = pixel
    
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx 
    cy = intrinsics.ppy 
    
    X = (x - cx) * depth_value / fx
    Y = (y - cy) * depth_value / fy
    Z = depth_value
    
    return X, Y, Z

def project_point_to_pixel(intrinsics, point_3d):
    """
    Project 3D point back to 2D pixel coordinates using camera intrinsics.
    This is the inverse of deprojection.
    
    Args:
        intrinsics: Camera intrinsics object from RealSense
        point_3d: Tuple (X, Y, Z) representing 3D coordinates in meters
    
    Returns:
        Tuple (x, y) representing pixel coordinates
    """
    X, Y, Z = point_3d
    
    # Extract intrinsic parameters
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    
    # Project 3D point to 2D pixel
    x = (X * fx / Z) + cx
    y = (Y * fy / Z) + cy
    
    return x, y

def get_depth_at_pixel(depth_frame, x, y, window_size=5):
    """
    Get depth value at pixel (x, y) with averaging over a small window.
    This helps deal with noise in depth measurements.
    
    Args:
        depth_frame: RealSense depth frame
        x, y: Pixel coordinates
        window_size: Size of averaging window (default 5x5)
    
    Returns:
        Depth value in meters, or None if invalid
    """
    depth_image = np.asanyarray(depth_frame.get_data())
    h, w = depth_image.shape
    
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    
    # Extract a small window around the pixel
    half_window = window_size // 2
    y_min = max(0, y - half_window)
    y_max = min(h, y + half_window + 1)
    x_min = max(0, x - half_window)
    x_max = min(w, x + half_window + 1)
    
    depth_window = depth_image[y_min:y_max, x_min:x_max]
    
    valid_depths = depth_window[depth_window > 0]
    
    if len(valid_depths) == 0:
        return None
    
    depth_value = np.median(valid_depths)
    
    # Convert from millimeters to meters (RealSense depth is in mm)
    depth_meters = depth_value / 1000.0
    
    return depth_meters

def detect_bar_3d_position(color_image, depth_frame, intrinsics, rotation_info):
    """
    Get 3D position of the aluminum bar (reference height).
    
    Args:
        color_image: Original color image
        depth_frame: RealSense depth frame
        intrinsics: Camera intrinsics
        rotation_info: Dictionary containing bar line detection info
    
    Returns:
        bar_3d: (X, Y, Z) tuple of bar position in camera coordinates
        bar_pixel: (x, y) pixel position in original image
        bar_pixel_rotated: (x, y) pixel position in rotated image
    """
    if rotation_info is None:
        print("No rotation info available for bar detection")
        return None, None, None
    
    # Use the detected bar line's midpoint
    start = rotation_info['line_points'][0]
    end = rotation_info['line_points'][1]
    
    # Calculate midpoint of the bar line (in original image coordinates)
    bar_x = int((start[0] + end[0]) / 2)
    bar_y = int((start[1] + end[1]) / 2)
    
    print(f"Bar pixel (original): ({bar_x}, {bar_y})")
    
    # Get depth at bar position
    depth_value = get_depth_at_pixel(depth_frame, bar_x, bar_y, window_size=5)
    
    if depth_value is None or depth_value <= 0:
        print("Invalid depth at bar position, trying larger window...")
        depth_value = get_depth_at_pixel(depth_frame, bar_x, bar_y, window_size=11)
    
    if depth_value is None or depth_value <= 0:
        print("Could not get valid depth at bar position")
        return None, None, None
    
    print(f"Bar depth: {depth_value:.4f} m")
    
    # Deproject to 3D
    X, Y, Z = deproject_pixel_to_point(intrinsics, (bar_x, bar_y), depth_value)
    
    # Also compute the rotated pixel position for visualization
    rotation_matrix = rotation_info['rotation_matrix']
    bar_x_rot, bar_y_rot = rotate_point(bar_x, bar_y, rotation_matrix)
    
    return (X, Y, Z), (bar_x, bar_y), (bar_x_rot, bar_y_rot)

def compute_plant_height(canopy_3d, bar_3d):
    """
    Compute actual plant height as vertical distance from bar to canopy.
    
    In camera coordinates:
    - Y axis points downward in the image
    - Bar is typically below the canopy (larger Y value)
    - Canopy is above the bar (smaller Y value)
    
    Args:
        canopy_3d: (X, Y, Z) of canopy top
        bar_3d: (X, Y, Z) of reference bar
    
    Returns:
        height: Plant height in meters (always positive)
    """
    canopy_Y = canopy_3d[1]
    bar_Y = bar_3d[1]
    
    # Height = vertical difference
    # Bar Y should be larger (lower in image) than canopy Y (higher in image)
    height = bar_Y - canopy_Y
    
    print(f"Canopy Y: {canopy_Y:.4f} m")
    print(f"Bar Y: {bar_Y:.4f} m")
    print(f"Height difference (bar_Y - canopy_Y): {height:.4f} m")
    
    return abs(height)

def process_canopy_detection(color_image, depth_frame, color_intrinsics, timestamp):
    """
    Main processing pipeline for canopy detection and height measurement.
    
    Returns:
        plant_height: Measured plant height in meters
        canopy_3d: (X, Y, Z) coordinates of canopy
        bar_3d: (X, Y, Z) coordinates of reference bar
        visualization: Annotated image showing measurements
    """
    try:
        # Step 1: Detect aluminum bar edges and rotate the image
        print("\n" + "="*60)
        print("CANOPY DETECTION PIPELINE")
        print("="*60)
        print("\nStep 1: Detecting aluminum bar edges and rotating image...")
        rotation_info, detected_bars, processed_image, selected_edge_image, rotated_image = detect_rotate_aluminum_bar_edges(color_image)
        
        if rotation_info is not None:
            print(f"  Rotation Applied: {rotation_info['rotation_angle']:.2f} degrees")
            rotation_matrix = rotation_info['rotation_matrix']
        else:
            print("  No rotation applied, using original image")
            rotated_image = color_image
            rotation_matrix = None
        
        # Step 2: Get bar 3D position (reference point)
        print("\nStep 2: Getting bar reference position...")
        bar_result = detect_bar_3d_position(color_image, depth_frame, color_intrinsics, rotation_info)
        
        if bar_result[0] is None:
            print("  ERROR: Could not detect bar reference position")
            return None, None, None, None
        
        bar_3d, bar_pixel_orig, bar_pixel_rot = bar_result
        print(f"  Bar 3D position: X={bar_3d[0]:.4f}, Y={bar_3d[1]:.4f}, Z={bar_3d[2]:.4f}")
        
        # Step 3: Remove background to extract plants using GrabCut
        print("\nStep 3: Removing background with GrabCut...")
        plant_img, plant_mask = remove_background_grabcut(rotated_image)

        # Step 4: Apply a green mask over the extracted foreground
        print("\nStep 4: Applying green mask...")
        green_masked_image, green_mask = apply_green_mask(plant_img)
        colored_mask = cv2.bitwise_and(green_masked_image, green_masked_image, mask=green_mask)

        # Step 5: Find canopy level at the highest detected plant pixel
        print("\nStep 5: Detecting canopy height...")
        canopy_y_rotated, canopy_x_rotated = canopy_level_mark(colored_mask)

        if canopy_y_rotated is None:
            print("  ERROR: Failed to detect canopy position")
            return None, None, None, None
        
        print(f"  Canopy pixel (rotated): ({canopy_x_rotated}, {canopy_y_rotated})")
        
        # Step 6: Convert rotated coordinates back to original image coordinates
        print("\nStep 6: Converting to original image coordinates...")
        if rotation_matrix is not None:
            orig_x, orig_y = inverse_rotate_point(
                canopy_x_rotated, 
                canopy_y_rotated, 
                rotation_matrix, 
                color_image.shape
            )
            print(f"  Canopy pixel (original): ({orig_x}, {orig_y})")
        else:
            orig_x, orig_y = canopy_x_rotated, canopy_y_rotated
        
        # Step 7: Get depth value at the canopy pixel
        print("\nStep 7: Getting depth value at canopy point...")
        depth_value = get_depth_at_pixel(depth_frame, orig_x, orig_y, window_size=5)
        
        if depth_value is None or depth_value <= 0:
            print("  Trying larger window for depth...")
            depth_value = get_depth_at_pixel(depth_frame, orig_x, orig_y, window_size=11)
        
        if depth_value is None or depth_value <= 0:
            print("  ERROR: Could not get valid depth at canopy point")
            return None, None, None, None
        
        print(f"  Canopy depth: {depth_value:.4f} m")
        
        # Step 8: Deproject canopy to 3D coordinates
        print("\nStep 8: Computing canopy 3D coordinates...")
        canopy_3d = deproject_pixel_to_point(
            color_intrinsics, 
            (orig_x, orig_y), 
            depth_value
        )
        print(f"  Canopy 3D: X={canopy_3d[0]:.4f}, Y={canopy_3d[1]:.4f}, Z={canopy_3d[2]:.4f}")
        
        # Step 9: Compute plant height
        print("\nStep 9: Computing plant height...")
        plant_height = compute_plant_height(canopy_3d, bar_3d)
        
        print(f"\n" + "="*60)
        print(f"RESULT: Plant Height = {plant_height:.4f} m ({plant_height*100:.1f} cm)")
        print("="*60)
        
        # Step 10: Create visualization
        print("\nStep 10: Creating visualization...")
        canopy_visualization = draw_canopy_visualization(
            color_image,
            rotated_image,
            canopy_x_rotated,
            canopy_y_rotated,
            bar_pixel_rot,
            plant_height,
            canopy_3d,
            bar_3d
        )
        
        # Save plant height to file
        txt_filename = f"src/camera_sensor/camera_z_data/camera_z.txt"
        try:
            with open(txt_filename, 'w') as f:
                f.write(f"{plant_height:.4f}")
            print(f"\nSaved plant height to '{txt_filename}'")
        except Exception as e:
            print(f"Warning: Could not save to file: {e}")

        return plant_height, canopy_3d, bar_3d, canopy_visualization
    
    except Exception as e:
        print(f"\nERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    WIDTH = 1280
    HEIGHT = 720
    FPS = 6

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    # Start streaming
    print("\n" + "="*60)
    print("PLANT HEIGHT MEASUREMENT SYSTEM")
    print("="*60)
    print(f"\nStarting RealSense camera at {WIDTH}x{HEIGHT} @ {FPS}fps...")
    print("\nControls:")
    print("  'e' - Capture and measure plant height")
    print("  's' - Save current frame without processing")
    print("  'q' - Quit")
    print("="*60)
    
    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    print(f"\nCamera Intrinsics:")
    print(f"  fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
    print(f"  cx={color_intrinsics.ppx:.2f}, cy={color_intrinsics.ppy:.2f}")
    
    align = rs.align(rs.stream.color)

    # Variables for storing last measurement
    last_visualization = None
    last_plant_height = None

    try:
        frame_count = 0
        print("\nCamera warming up...")
        
        while True:
            frames = pipeline.wait_for_frames()
            frame_count += 1
            
            # Align depth frame to color frame
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Create display image with overlay
            display_image = color_image.copy()
            cv2.putText(display_image, f"Resolution: {WIDTH}x{HEIGHT} @ {FPS}fps", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_image, "Press 'E' to capture and measure", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show last measurement if available
            if last_plant_height is not None:
                cv2.putText(display_image, f"Last measurement: {last_plant_height*100:.1f} cm", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show the live feed
            cv2.imshow('RealSense Live Feed', display_image)
            
            # Show last visualization if available
            if last_visualization is not None:
                cv2.imshow('Last Measurement', last_visualization)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('e'):
                print("\n'E' pressed - Capturing and processing...")
                
                # Generate timestamp
                timestamp = get_timestamp()
                
                # Save raw captures
                color_filename = f"new-captures/canopy_capture_{timestamp}_HD.png"
                depth_filename = f"new-captures/depth_snapshot_{timestamp}_HD.png"
                
                try:
                    cv2.imwrite(color_filename, color_image)
                    cv2.imwrite(depth_filename, depth_image)
                    print(f"Saved color image: '{color_filename}'")
                    print(f"Saved depth image: '{depth_filename}'")
                except Exception as e:
                    print(f"Warning: Could not save images: {e}")
                
                # Process and measure
                plant_height, canopy_3d, bar_3d, visualization = process_canopy_detection(
                    color_image, 
                    aligned_depth_frame, 
                    color_intrinsics, 
                    timestamp
                )
                
                if plant_height is not None:
                    last_plant_height = plant_height
                    last_visualization = visualization
                    
                    # Save visualization
                    vis_filename = f"new-captures/measurement_{timestamp}.png"
                    try:
                        cv2.imwrite(vis_filename, visualization)
                        print(f"Saved visualization: '{vis_filename}'")
                    except Exception as e:
                        print(f"Warning: Could not save visualization: {e}")
                    
                    print(f"\n✓ Plant Height: {plant_height*100:.1f} cm")
                else:
                    print("\n✗ Measurement failed")
            
            elif key == ord('s'):
                timestamp = get_timestamp()
                filename = f"new-captures/snapshot_{timestamp}.png"
                cv2.imwrite(filename, color_image)
                print(f"Saved snapshot: '{filename}'")
            
            elif key == ord('q'):
                print("\nQuitting...")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")

if __name__ == "__main__":
    main()