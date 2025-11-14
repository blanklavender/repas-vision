import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import math

def get_timestamp():
    """Return the current timestamp in YYYY-MM-DDTHHMMSS format."""
    return datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")

def detect_rotate_aluminum_bar_edges(image):
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

def draw_canopy_visualization(original_image, rotated_image, canopy_x, canopy_y):
    """Draw canopy line and point on both original and rotated images."""
    # Draw on rotated image
    rotated_vis = rotated_image.copy()
    height, width = rotated_vis.shape[:2]
    
    # Draw horizontal red line at canopy height
    cv2.line(rotated_vis, (0, canopy_y), (width - 1, canopy_y), (0, 0, 255), 2)
    
    # Draw blue circle at the canopy point
    cv2.circle(rotated_vis, (canopy_x, canopy_y), 5, (255, 0, 0), -1)
    
    # Add text label
    label = f"Canopy: ({canopy_x}, {canopy_y})"
    cv2.putText(rotated_vis, label, (canopy_x + 10, canopy_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(rotated_vis, label, (canopy_x + 10, canopy_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
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

def calculate_reprojection_error(intrinsics, original_pixel, point_3d):
    """
    Calculate reprojection error by projecting 3D point back to 2D and comparing
    with the original pixel coordinates.
    
    Args:
        intrinsics: Camera intrinsics object from RealSense
        original_pixel: Tuple (x, y) of original pixel coordinates
        point_3d: Tuple (X, Y, Z) of 3D coordinates
    
    Returns:
        Dictionary containing reprojection error details
    """
    orig_x, orig_y = original_pixel
    
    # Project 3D point back to 2D
    reproj_x, reproj_y = project_point_to_pixel(intrinsics, point_3d)
    
    # Calculate pixel-wise differences
    dx = reproj_x - orig_x
    dy = reproj_y - orig_y
    
    # Calculate Euclidean distance (reprojection error)
    error = np.sqrt(dx**2 + dy**2)
    
    return {
        'original_pixel': (orig_x, orig_y),
        'reprojected_pixel': (reproj_x, reproj_y),
        'error_x': dx,
        'error_y': dy,
        'error_euclidean': error
    }

def print_reprojection_error(error_info):
    """
    Print reprojection error information in a formatted way.
    
    Args:
        error_info: Dictionary returned by calculate_reprojection_error
    """
    print("\n" + "="*60)
    print("REPROJECTION ERROR ANALYSIS")
    print("="*60)
    print(f"Original Pixel:     ({error_info['original_pixel'][0]:.2f}, {error_info['original_pixel'][1]:.2f})")
    print(f"Reprojected Pixel:  ({error_info['reprojected_pixel'][0]:.2f}, {error_info['reprojected_pixel'][1]:.2f})")
    print(f"Error in X:         {error_info['error_x']:+.4f} pixels")
    print(f"Error in Y:         {error_info['error_y']:+.4f} pixels")
    print(f"Euclidean Error:    {error_info['error_euclidean']:.4f} pixels")
    
    # Interpretation
    if error_info['error_euclidean'] < 1.0:
        status = "✓ EXCELLENT (< 1 pixel)"
    elif error_info['error_euclidean'] < 2.0:
        status = "✓ GOOD (< 2 pixels)"
    elif error_info['error_euclidean'] < 5.0:
        status = "⚠ ACCEPTABLE (< 5 pixels)"
    else:
        status = "✗ POOR (≥ 5 pixels) - Check calibration!"
    
    print(f"Status:             {status}")
    print("="*60 + "\n")

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

def process_canopy_detection(color_image, depth_frame, color_intrinsics, timestamp):
    try:
        # Step 1: Detect aluminum bar edges and rotate the image
        print("Step 1: Detecting aluminum bar edges and rotating image...")
        rotation_info, detected_bars, processed_image, selected_edge_image, rotated_image = detect_rotate_aluminum_bar_edges(color_image)
        
        if rotation_info is not None:
            print(f"Rotation Applied: {rotation_info['rotation_angle']:.2f} degrees")
            rotation_matrix = rotation_info['rotation_matrix']
        else:
            print("No rotation applied, using original image")
            rotated_image = color_image
            rotation_matrix = None
        
        # Step 2: Remove background to extract plants using GrabCut
        print("Step 2: Removing background with GrabCut...")
        plant_img, plant_mask = remove_background_grabcut(rotated_image)

        # Step 3: Apply a green mask over the extracted foreground
        print("Step 3: Applying green mask...")
        green_masked_image, green_mask = apply_green_mask(plant_img)
        colored_mask = cv2.bitwise_and(green_masked_image, green_masked_image, mask=green_mask)

        # Step 4: Find canopy level at the highest detected plant pixel
        print("Step 4: Detecting canopy height...")
        canopy_y_rotated, canopy_x_rotated = canopy_level_mark(colored_mask)

        if canopy_y_rotated is not None:
            print("Step 5: Creating visualization...")
            # canopy_visualization = draw_canopy_visualization(
            #     color_image, 
            #     rotated_image, 
            #     canopy_x_rotated, 
            #     canopy_y_rotated
            # )
            
            # # Save the visualization image
            # vis_filename = f"new-captures/canopy_visualization_{timestamp}.png"
            # cv2.imwrite(vis_filename, canopy_visualization)
            # print(f"Saved canopy visualization as '{vis_filename}'")
            
            # Step 6: Convert rotated coordinates back to original image coordinates
            print("Step 6: Converting to original image coordinates...")
            if rotation_matrix is not None:
                orig_x, orig_y = inverse_rotate_point(
                    canopy_x_rotated, 
                    canopy_y_rotated, 
                    rotation_matrix, 
                    color_image.shape
                )
                # print(f"Original image coordinates: x={orig_x}, y={orig_y}")
            else:
                orig_x, orig_y = canopy_x_rotated, canopy_y_rotated
            
            # Step 7: Get depth value at the detected pixel
            print("Step 7: Getting depth value at canopy point...")
            depth_value = get_depth_at_pixel(depth_frame, orig_x, orig_y, window_size=5)
            
            if depth_value is None or depth_value <= 0:
                print("Trying to find nearest valid depth...")
                depth_value = get_depth_at_pixel(depth_frame, orig_x, orig_y, window_size=11)
            
            if depth_value is not None and depth_value > 0:
                print(f"Depth at canopy point: {depth_value:.4f} meters ({depth_value*1000:.1f} mm)")
                
                # Step 8: Deproject to 3D coordinates
                print("Step 8: Computing 3D coordinates...")
                X, Y, Z = deproject_pixel_to_point(
                    color_intrinsics, 
                    (orig_x, orig_y), 
                    depth_value
                )
                
                # # Step 9: Calculate and print reprojection error
                # print("Step 9: Validating 3D reconstruction...")
                # error_info = calculate_reprojection_error(
                #     color_intrinsics,
                #     (orig_x, orig_y),
                #     (X, Y, Z)
                # )
                # print_reprojection_error(error_info)
                
                # Save simple y-coordinate only txt file
                simple_txt_filename = f"new-captures/canopy_y_{timestamp}.txt"
                with open(simple_txt_filename, 'w') as f:
                    f.write(f"{Y:.4f}\n")

                return canopy_y_rotated, (X, Y, Z)
            else:
                print("Failed to get valid depth value at canopy point")
                return canopy_y_rotated, None
        else:
            print("Failed to detect canopy height")
            return None, None
    
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    pipeline = rs.pipeline()
    config = rs.config()

    WIDTH = 1280
    HEIGHT = 720
    FPS = 6

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    # Start streaming
    print("\nStarting RealSense camera...")
    print("Press 'e' to capture and process canopy height")
    print("Press 'q' to quit")
    
    profile = pipeline.start(config)
    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)

    try:
        frame_count = 0
        print("Camera warming up...")
        
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

            display_image = color_image.copy()
            cv2.putText(display_image, f"Resolution: {WIDTH}x{HEIGHT} @ {FPS}fps", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_image, "Press 'E' to capture", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the live feed
            cv2.imshow('RealSense HD Live Feed (1280x720)', display_image)
            
            # Wait for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('e'):
                print("\n'e' key pressed - Capturing and processing...")
                
                # Generate timestamp
                timestamp = get_timestamp()
                color_filename = f"new-captures/canopy_capture_{timestamp}_HD.png"
                depth_filename = f"new-captures/depth_snapshot_{timestamp}_HD.png"
                
                cv2.imwrite(color_filename, color_image)
                cv2.imwrite(depth_filename, depth_image)
                print(f"Saved HD color image as '{color_filename}' ({WIDTH}x{HEIGHT})")
                print(f"Saved HD depth image as '{depth_filename}' ({WIDTH}x{HEIGHT})")
                
                canopy_y, coords_3d = process_canopy_detection(
                    color_image, 
                    aligned_depth_frame, 
                    color_intrinsics, 
                    timestamp
                )
                
                if canopy_y is not None and coords_3d is not None:
                    X, Y, Z = coords_3d
                    print(f"\n✓ Successfully detected canopy at HD resolution")
                    print(f"  Y={Y:.4f} m")
                else:
                    print("\n✗ Failed to detect canopy height or compute 3D coordinates")
            
            elif key == ord('q'):
                print("Quitting...")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")

if __name__ == "__main__":
    main()