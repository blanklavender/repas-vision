import cv2
import numpy as np
import math

def detect_rotate_aluminum_bar_edges(image_path):

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image. Please check the file path.")
    
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
            'line_coverage': best_line['coverage']
        }

        cv2.line(selected_edge_image, (best_line['start'][0], best_line['start'][1]), (best_line['end'][0], best_line['end'][1]), (0, 0, 255), 2)
        
        return rotation_info, bar_edges, processed_image, selected_edge_image, rotated_image
    
    else:
        print("No suitable edge detected for rotation.")
        return image, None, processed_image, []
    
def remove_background_grabcut(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Thresholds for Green Color
    lower_green = np.array([35, 40, 40]) 
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Initialize the GrabCut mask
    grabcut_mask = np.where(green_mask==255, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')
    height, width = image.shape[:2]
    rect = (1, 1, width-2, height-2)
    
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    
    # Run GrabCut
    cv2.grabCut(image, grabcut_mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((grabcut_mask==cv2.GC_FGD) | (grabcut_mask==cv2.GC_PR_FGD), 1, 0).astype('uint8')
    result = image * final_mask[:, :, np.newaxis]
    
    return result, final_mask

def apply_green_mask(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 80, 30])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up the mask with morphological operations
    kernel = np.ones((3,3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    return image, green_mask

def canopy_level_mark(image, rotated_image):

    canopy_image = image.copy()
    height, width = image.shape[:2]

    # Create a boolean mask where True indicates a non-black pixel (i.e. plant pixel)
    mask = np.any(canopy_image != 0, axis=2)

    if np.any(mask):
        ys, xs = np.where(mask)
        print(ys)
        print(xs)

        # Find the minimum y coordinate (the highest plant pixel along the vertical axis)
        canopy_y = int(np.min(ys))
        indices = np.where(ys == canopy_y)[0]
        canopy_x = int(np.median(xs[indices]))

        print("---------------------------------------------------")
        print("The interest y coordinate height is: ", canopy_y)
        print("The corresponding x coordinate is:", canopy_x)

        cv2.line(canopy_image, (0, canopy_y), (width - 1, canopy_y), (0, 0, 255), 2)
        cv2.line(rotated_image, (0, canopy_y), (width - 1, canopy_y), (0, 0, 255), 2)

        # Draw a circle at the (canopy_x, canopy_y) point on both images
        cv2.circle(canopy_image, (canopy_x, canopy_y), 5, (255, 0, 0), -1)
        cv2.circle(rotated_image, (canopy_x, canopy_y), 5, (255, 0, 0), -1)

    return canopy_image, rotated_image

def main():
    image_path = 'test_images/side-view-1_Color.png'

    try:
    
        # Step 1: Detect aluminum bar edges and rotate the image
        rotation_info, detected_bars, processed_image, selected_edge_image, rotated_image = detect_rotate_aluminum_bar_edges(image_path)
        
        if rotation_info is not None:
            print("Rotation Details:")
            print(f"Original Line Angle: {rotation_info['original_angle']:.2f} degrees")
            print(f"Rotation Angle: {rotation_info['rotation_angle']:.2f} degrees")
            print(f"Line Points: {rotation_info['line_points']}")
            print(f"Line Coverage: {rotation_info['line_coverage']:.2f} (% of image width)")
        else:
            print("No rotation applied.")
        
        print(f"\nNumber of aluminum bars detected: {len(detected_bars)}")
        for i, bar in enumerate(detected_bars, 1):
            print(f"\nBar {i}:")
            print(f"  Start Point: {bar['start']}")
            print(f"  End Point: {bar['end']}")
            print(f"  Length: {bar['length']:.2f} pixels")
            print(f"  Angle: {bar['angle']:.2f} degrees")
            print(f"  Coverage: {bar['coverage']:.2f} (fraction of image width)")

        cv2.namedWindow('Aluminum Bar Edges', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Aluminum Bar Edges', 800, 600)
        cv2.imshow('Aluminum Bar Edges', processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.namedWindow('Selected Edge', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Selected Edge', 800, 600)
        cv2.imshow('Selected Edge', selected_edge_image)
        cv2.namedWindow('Rotated Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Rotated Image', 800, 600)
        cv2.imshow('Rotated Image', rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Step 2: Remove background to extract plants using GrabCut (initialized with a green mask)
        plant_img, plant_mask = remove_background_grabcut(rotated_image)

        cv2.namedWindow('Foreground (Plants) Extracted', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Foreground (Plants) Extracted', 800, 600)
        cv2.imshow('Foreground (Plants) Extracted', plant_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Step 3: Apply a green mask over the extracted foreground to highlight plant features
        green_masked_image, green_mask = apply_green_mask(plant_img)

        cv2.namedWindow('Green Mask (Binary)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Green Mask (Binary)', 800, 600)
        cv2.imshow('Green Mask (Binary)', green_mask)
        colored_mask = cv2.bitwise_and(green_masked_image, green_masked_image, mask=green_mask)
        cv2.namedWindow('Green Mask (Colored)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Green Mask (Colored)', 800, 600)
        cv2.imshow('Green Mask (Colored)', colored_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Step 4: Draw canopy level at the highest detected plant pixel
        canopy_image, rotated_image = canopy_level_mark(colored_mask, rotated_image)

        cv2.namedWindow('Canopy Height Orig Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Canopy Height Orig Image', 800, 600)
        cv2.imshow('Canopy Height Orig Image', rotated_image)
        cv2.namedWindow('Canopy Height on Foreground', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Canopy Height on Foreground', 800, 600)
        cv2.imshow('Canopy Height on Foreground', canopy_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
