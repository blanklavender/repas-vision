# Femto Bolt Camera Vision System

## Overview

This codebase contains a comprehensive computer vision system built around the Orbbec Femto Bolt ToF camera. The system is designed for 3D pose estimation, point cloud generation, and CAD model visualization in hydroponic/aquaponic environments. The primary workflow involves AprilTag detection, pose estimation, and transformation of CAD models for visualization in Open3D.

## Hardware Specifications

### Orbbec Femto Bolt Camera
- **Sensor Type**: Time-of-Flight (ToF) depth sensor
- **Chosen Color Resolution**: 1280x720 @ 30fps
- **Chosen Depth Resolution**: 640x480 @ 30fps
- **Depth Range**: 0.3m - 3.0m
- **Field of View**: 
  - Color: 69° x 55° (H x V)
  - Depth: 69° x 55° (H x V)
- **Interface**: USB 3.0
- **SDK**: pyorbbecsdk (Orbbec SDK)
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
1. **Image Capture**: RGB/Depth streams from Femto Bolt
2. **AprilTag Detection**: Using pupil-apriltags library
3. **Pose Estimation**: cv2.solvePnP with calibrated intrinsics
4. **Coordinate Transformation**: OpenCV to Open3D conversion
5. **CAD Loading**: STL/PLY model loading and scaling
6. **Visualization**: Real-time 3D rendering in Open3D

## Scripts and Use Cases

### Core Detection & Pose Estimation
- **`april_tag_detector_solvepnp.py`**: Real-time AprilTag detection and pose estimation using solvePnP
- **`april_tag_detector_ToF.py`**: AprilTag detection with ToF depth integration
- **`better_three_capture.py`**: Multi-pose capture system for validation

### Calibration & Intrinsics
- **`checkerboard_callibration.py`**: Camera calibration using checkerboard patterns
- **`fetch_intrinsics.py`**: Extract factory calibration parameters from camera
- **`calibration_parameters/`**: Directory containing calibration data (JSON/NPZ files)

### Data Capture & Storage
- **`rgbd_viewer.py`**: Real-time RGB-D stream viewer
- **`view_point_cloud.py`**: Point cloud visualization tool
- **`supported_stream_list.py`**: List available camera streams and formats
- **`better_three_capture.py`**: Captures RGB, depth, and PLY files with timestamps

### Visualization & CAD Integration
- **`final_view.py`**: Basic point cloud and AprilTag visualization
- **`final_view_with_cad.py`**: Complete CAD model integration with pose estimation
- **`four_pose_captures/`**: Multi-pose capture data for validation
- **`hydroponic_system_captures/`**: Real-world system capture data

### Data Directories
- **`four_pose_captures/`**: Validation captures with multiple poses (contains RGB, depth, PLY files)
- **`hydroponic_system_captures/`**: Production system captures (contains RGB, depth, PLY files)
- **`calibration_parameters/`**: Camera calibration data and intrinsics
  - **Factory intrinsics**: `factory_color_intrinsics_*.json`, `factory_depth_intrinsics_*.json`
  - **Checkerboard calibration**: `checkerboard_color_intrinsics_*.json`
  - **Extrinsics**: `factory_extrinsics_d2c_*.json`

## Key Features

### AprilTag Integration
- Support for tag36h11 family
- Real-time pose estimation
- Multiple tag detection
- Robust corner refinement

### CAD Model Support
- STL and PLY format support
- Automatic coordinate system alignment
- Configurable scaling and rotation
- Origin offset handling

### Calibration System
- Factory intrinsics extraction
- Checkerboard calibration
- Distortion correction
- Multi-resolution support

### Visualization Tools
- Real-time 3D rendering
- Point cloud visualization
- Multi-coordinate system support
- Interactive Open3D interface

## Setup Instructions

### Virtual Environment Setup

It is **strongly recommended** to use separate virtual environments for each camera system to avoid dependency conflicts. Open3D requires Python 3.11 or more for optimal compatibility.

Follow instructions on pyorbbecsdk setup here: https://github.com/orbbec/pyorbbecsdk

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
pip install pyorbbecsdk opencv-python numpy open3d pupil-apriltags
```

#### Deactivate Virtual Environment
```bash
deactivate
```

### Dependencies
- **pyorbbecsdk**: Orbbec SDK for Femto Bolt camera
- **opencv-python**: Computer vision library
- **numpy**: Numerical computing
- **open3d**: 3D data processing and visualization (requires Python 3.11)
- **pupil-apriltags**: AprilTag detection library

## Usage Examples

### Real-time AprilTag Detection
```bash
python scripts/april_tag_detector_solvepnp.py
```

### CAD Model Visualization
```bash
python scripts/final_view_with_cad.py
```

### Camera Calibration
```bash
python scripts/checkerboard_callibration.py
```

## File Structure

```
femto_bolt_code/
├── pyorbbecsdk/           # Orbbec SDK and examples
├── scripts/               # Main application scripts
│   ├── calibration_parameters/  # Camera calibration data
│   ├── four_pose_captures/      # Validation data
│   └── hydroponic_system_captures/  # Production data
└── README.md             # This file
```

## Configuration

Key configuration parameters are typically set at the top of each script:
- **TAG_SIZE_M**: AprilTag physical size in meters
- **CALIB_JSON_PATH**: Path to calibration parameters
- **CAD_PLY**: Path to CAD model file
- **Camera intrinsics**: fx, fy, cx, cy values

## Data Storage and Existing Captures

### RGB, Depth, and PLY Capture
- **`better_three_capture.py`**: Main script for capturing RGB images, depth data, and PLY point cloud files
- **Capture format**: Timestamped folders containing:
  - `color_YYYYMMDD_HHMMSS.png` - RGB image
  - `depth_YYYYMMDD_HHMMSS.png` - Depth image
  - `point_cloud_YYYYMMDD_HHMMSS.ply` - Point cloud file
  - `metadata_YYYYMMDD_HHMMSS.json` - Capture metadata

### Factory Camera Intrinsics
- **Location**: `scripts/calibration_parameters/`
- **Files**:
  - `factory_color_intrinsics_*.json` - Color camera intrinsics
  - `factory_depth_intrinsics_*.json` - Depth camera intrinsics
  - `factory_extrinsics_d2c_*.json` - Depth-to-color extrinsics
  - `checkerboard_color_intrinsics_*.json` - Manual calibration data

### Existing Capture Data
- **`four_pose_captures/`**: Contains captures of a potted plant in 4 rotations
  - Each capture includes RGB, depth, PLY, and metadata files
  - Useful for testing combinging point clouds
- **`hydroponic_system_captures/`**: Contains 6 hydroponic system captures
  - Real-world hydroponic system data
  - Ready-to-use for detecting april tags and visualization

## Notes

- The system is optimized for indoor hydroponic environment monitoring
- AprilTag size is calibrated to 30.3mm (0.0303m)
- Coordinate transformations handle OpenCV to Open3D conversion
- CAD models are scaled from millimeters to meters (0.001 factor)
- Real-time performance optimized for 30fps operation
- **Use separate virtual environments** for Femto Bolt and RealSense systems to avoid conflicts
