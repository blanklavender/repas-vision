# RealSense D415i Camera Vision System

## Overview

This codebase contains a comprehensive computer vision system built around the Intel RealSense D415i stereo camera. The system is designed for 3D pose estimation, point cloud generation, and CAD model visualization in hydroponic/aquaponic environments. The primary workflow involves AprilTag detection, pose estimation, and transformation of CAD models for visualization in Open3D.

## Hardware Specifications

### Intel RealSense D415i Stereo Camera
- **Sensor Type**: Stereo depth with IMU
- **Chosen Color Resolution**: Up to 640x480 @ 30fps
- **Chosen Depth Resolution**: Up to 640x480 @ 30fps
- **Depth Range**: 0.16m - 10.0m
- **Field of View**: 
  - Color: 69.4° x 42.5° (H x V)
  - Depth: 65° x 40° (H x V)
- **IMU**: 6-axis (accelerometer + gyroscope)
- **Interface**: USB 3.0
- **SDK**: pyrealsense2 (Intel RealSense SDK)
- **Operating System**: Windows/Linux

## Transformation Pipeline

The complete transformation pipeline from AprilTag detection to CAD visualization follows these steps:

### 1. AprilTag Detection & Pose Estimation
```
Camera Image → AprilTag Detection → Corner Extraction → solvePnP → 6DOF Pose
```

### 2. Coordinate System Transformations
```
OpenCV Coordinates (x→right, y→down, z→forward) 
    ↓
Open3D Coordinates (x→right, y→up, z→into-screen)
    ↓
CAD Model Placement & Visualization
```

### 3. CAD Model Integration
```
AprilTag Pose → Transformation Matrix → CAD Model Loading → 
Coordinate System Alignment → Open3D Visualization
```

### 4. Complete Pipeline Flow
1. **Image Capture**: RGB/Depth streams from RealSense D415i
2. **AprilTag Detection**: Using pupil-apriltags library
3. **Pose Estimation**: cv2.solvePnP with calibrated intrinsics
4. **Coordinate Transformation**: OpenCV to Open3D conversion
5. **CAD Loading**: STL/PLY model loading and scaling
6. **Visualization**: Real-time 3D rendering in Open3D

## Scripts and Use Cases

### AprilTag Detection & Calibration
- **`april_tag_detection_caliberation/april_tag_detection_image.py`**: Static image AprilTag detection
- **`april_tag_detection_caliberation/realtime_pose_estimation_april_tag.py`**: Real-time AprilTag pose estimation
- **`april_tag_detection_caliberation/checkerboard_caliberation.py`**: Camera calibration using checkerboard
- **`april_tag_detection_caliberation/fetch_factory_intrinsic.py`**: Extract factory intrinsics
- **`april_tag_detection_caliberation/fetch_factory_extrinsic.py`**: Extract factory extrinsics

### Data Capture & Processing
- **`capture_scripts/capture_aligned_all.py`**: Capture aligned RGB-D data and PLY files
- **`capture_scripts/capture_aligned_pointcloud.py`**: Generate aligned point clouds and PLY files
- **`capture_scripts/distance_masking_on_ply.py`**: Apply distance masking to point clouds
- **`capture_scripts/visualize_ply.py`**: Point cloud visualization

### Testing & Validation
- **`testing_scripts/color_640x480_live_streaming.py`**: Live color stream testing
- **`testing_scripts/test_camera_status.py`**: Camera connectivity and status testing
- **`testing_scripts/three_pose_vertical_translation_validation.py`**: Multi-pose validation
- **`testing_scripts/visualize_point_cloud.py`**: Point cloud visualization testing
- **`testing_scripts/supported_stream_list.py`**: List available camera streams

### Visualization Tools
- **`vis_tool/vis_tool_solvepnp.py`**: solvePnP pose estimation visualization
- **`vis_tool/vis_tool_april_tag_pose_validaiton.py`**: AprilTag pose validation tool
- **`vis_tool/vis_tool_not_working_ref.py`**: Reference implementation (non-functional)

### Canopy Detection (Specialized)
- **`canopy_detection/bag_to_img.py`**: Convert ROS bag files to images
- **`canopy_detection/combined-logic.py`**: Combined canopy detection logic
- **`canopy_detection/image_capture.py`**: Image capture for canopy analysis

## Key Features

### AprilTag Integration
- Support for tag36h11 family
- Real-time pose estimation
- Multiple tag detection
- Robust corner refinement with CLAHE preprocessing

### CAD Model Support
- PLY format support
- Automatic coordinate system alignment
- Configurable scaling and rotation
- Origin offset handling

### Calibration System
- Factory intrinsics extraction
- Checkerboard calibration
- Distortion correction

### Advanced Processing
- Aligned RGB-D capture
- Point cloud generation
- Distance-based masking

### Visualization Tools
- Real-time 3D rendering
- Point cloud visualization
- Multi-coordinate system support
- Interactive Open3D interface

## Setup Instructions

### Virtual Environment Setup

It is **strongly recommended** to use separate virtual environments for each camera system to avoid dependency conflicts. Open3D requires Python 3.11 for optimal compatibility.

#### Create Virtual Environment (Python 3.11)
```bash
# Create virtual environment with Python 3.11
python3.11 -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Verify Python version
python --version  # Should show Python 3.11.x
```

#### Install Dependencies
```bash
# Install from requirements.txt (if available)
pip install -r requirements.txt

# Or install manually
pip install pyrealsense2 opencv-python numpy open3d pupil-apriltags
```

#### Deactivate Virtual Environment
```bash
deactivate
```

### Dependencies
- **pyrealsense2**: Intel RealSense SDK for D415i camera
- **opencv-python**: Computer vision library
- **numpy**: Numerical computing
- **open3d**: 3D data processing and visualization (requires Python 3.11)
- **pupil-apriltags**: AprilTag detection library

## Usage Examples

### Real-time AprilTag Detection
```bash
python april_tag_detection_caliberation/realtime_pose_estimation_april_tag.py
```

### Camera Calibration
```bash
python april_tag_detection_caliberation/checkerboard_caliberation.py
```

### Point Cloud Capture
```bash
python capture_scripts/capture_aligned_pointcloud.py
```

### Pose Validation
```bash
python vis_tool/vis_tool_solvepnp.py
```

## File Structure

```
realsense_d415i/
├── april_tag_detection_caliberation/  # AprilTag detection and calibration
├── capture_scripts/                   # Data capture and processing
├── testing_scripts/                   # Testing and validation
├── vis_tool/                          # Visualization tools
├── canopy_detection/                  # Specialized canopy detection
└── README.md                         # This file
```

## Configuration

Key configuration parameters are typically set at the top of each script:
- **TAG_SIZE**: AprilTag physical size in meters (0.0303m)
- **Camera intrinsics**: fx, fy, cx, cy values for 640x480 resolution
- **Distortion coefficients**: k1, k2, p1, p2, k3 values
- **Stream configuration**: Resolution and frame rate settings

## Calibration Data

The system includes factory calibration parameters:
- **Color intrinsics**: 640x480 and 1280x720 resolutions
- **Depth-to-color extrinsics**: Spatial alignment parameters
- **Distortion coefficients**: Lens distortion correction

## Data Storage and Existing Captures

### RGB, Depth, and PLY Capture
- **`capture_scripts/capture_aligned_all.py`**: Script for capturing aligned RGB-D data and PLY point cloud files
- **`capture_scripts/capture_aligned_pointcloud.py`**: Specialized script for generating aligned point clouds ONLY
- **Capture format**: Timestamped files containing:
  - RGB and depth images
  - PLY point cloud files
  - Metadata and calibration information

### Factory Camera Intrinsics
- **Location**: `april_tag_detection_caliberation/`
- **Files**:
  - `factory_color_intrinsics_*.json` - Color camera intrinsics for different resolutions
  - `factory_d2c_extrinsics.json` - Depth-to-color extrinsics
  - Checkerboard calibration data

### Existing Capture Data
- **`testing_scripts/aligned_outputs/`**: Contains aligned RGB-D captures and PLY files
  - Ready-to-use test data for validation
  - Includes both color and depth images with corresponding PLY files
- **`testing_scripts/not_aligned_outputs/`**: Contains non-aligned captures for comparison
- **`canopy_detection/new-captures/`**: Contains canopy detection test images
- **`canopy_detection/test_images/`**: Additional test images for canopy analysis

## Notes

- The system is optimized for hydroponic/aquaponic environment monitoring
- AprilTag size is calibrated to 30.3mm (0.0303m)
- Coordinate transformations handle OpenCV to Open3D conversion
- CAD models are scaled from millimeters to meters (0.001 factor)
- Real-time performance optimized for 30fps operation
- IMU data available for advanced applications requiring orientation tracking
- **Use separate virtual environments** for Femto Bolt and RealSense systems to avoid conflicts
