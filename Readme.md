# REPAS Vision

Computer vision system for hydroponic plant monitoring using AprilTag-based pose estimation, canopy height detection, 3D reconstruction, and plant segmentation. Built around two camera systems: Orbbec Femto Bolt (ToF) and Intel RealSense D415i (stereo).

## Folder Structure

### `femto_bolt_code/scripts/` — Femto Bolt scripts

**AprilTag detection and pose estimation:**
- `april_tag_detector_solvepnp.py` — Real-time AprilTag detection with solvePnP pose estimation
- `april_tag_detector_ToF.py` — AprilTag detection using ToF depth data
- `april_tag_id_detector.py` — Detects and prints AprilTag IDs from camera feed
- `april_tag_bundle_view.py` — Visualizes multi-tag bundle pose in Open3D
- `april_tag_3d_point.py` — Extracts 3D coordinates of AprilTag corners using depth
- `april_tag_point_picker.py` — Interactive tool to pick points on AprilTag images

**CAD overlay and visualization:**
- `final_view.py` — Visualizes point cloud with AprilTag-derived coordinate frame
- `final_view_with_cad.py` — Overlays CAD model onto point cloud using AprilTag pose
- `view_cad_two_tags.py` — CAD overlay using two-tag bundle for pose
- `mpa_final_view_tag_bundle_with_cad.py` — Multi-pose-average tag bundle with CAD overlay
- `mpe_final_view_tag_bundle_with_cad.py` — Multi-pose-estimation tag bundle with CAD overlay
- `mpa_final_view_with_export.py` — Multi-pose-average view with PLY export
- `origin_viz.py` — Visualizes coordinate frame origin in 3D
- `origin_viz_2.py` — Alternate coordinate frame origin visualization
- `view_point_cloud.py` — Simple Open3D point cloud viewer

**ICP and point cloud registration:**
- `icp_cad_model.py` — ICP alignment of captured point cloud to CAD model
- `mpa_icp.py` — Multi-pose-average followed by ICP refinement
- `mpa_icp_export.py` — Multi-pose-average ICP with PLY export
- `export_6dof.py` — Exports 6DOF transformation as PLY
- `manual_pose_verify.py` — Manual verification of estimated poses

**Error analysis:**
- `point_correspondence_error.py` — Computes point correspondence error between captures
- `visualize_error.py` — Visualizes point-to-mesh distance error as heatmap

**Capture and data processing:**
- `rgbd_viewer.py` — Real-time RGB-D stream viewer
- `create_masked_ply.py` — Creates distance-masked point clouds from RGB-D
- `generate_point_cloud.py` — Generates PLY point cloud from color + depth image pair
- `custom_reader.py` — Custom reader for Femto Bolt data files
- `run_custom.py` — Runs custom pipeline configurations
- `ply_to_stl.py` — Converts PLY files to STL format
- `supported_stream_list.py` — Lists all supported stream modes on the Femto Bolt
- `fetch_intrinsics.py` — Fetches camera intrinsics from the Femto Bolt SDK
- `alignment_test.py` — Verifies depth-to-color alignment quality with edge overlays

**Calibration:**
- `checkerboard_callibration.py` — Checkerboard-based camera calibration
- `calibration_stereo.py` — Stereo calibration between IR and RGB sensors
- `callibration_capture.py` — Captures synchronized COLOR + IR pairs for calibration

### `femto_bolt_code/scripts/capture/` — Session-based capture tools

- `capture_session.py` — Burst capture of color + depth at each turntable rotation angle
- `capture_no_bloat.py` — Minimal live preview with single-frame save
- `better_three_capture.py` — Simultaneous RGB, depth, and PLY capture
- `generate_ply.py` — Generates PLY point clouds from a capture session's frame pairs
- `post-filters.py` — Offline depth-gradient mask testing on captured data
- `filter-test-pointcloud.py` — Visualizes gradient mask effect as point clouds
- `hw_filter_test.py` — Tests hardware noise removal filter settings
- `edge_overlay_check.py` — Depth-to-color alignment diagnostic via edge overlay
- `version_checker.py` — Prints pyorbbecsdk version

### `femto_bolt_code/scripts/calibration_test/` — D2C alignment experiments

- `calibration_param_test.py` — Builds point cloud using manual D2C calibration (bypasses SDK AlignFilter)
- `capture_pairs.py` — Captures paired IR + RGB for stereo calibration
- `capture_test_raw.py` — Raw capture for calibration testing
- `align_parameters.py` — Tests stereo calibration alignment parameters

### `femto_bolt_code/scripts/canopy_detector/` — Canopy height detection

- `canopy_script_img_based.py` — Canopy height using EfficientSAM segmentation + bounding box top edge
- `canopy_script_depth_based.py` — Canopy height using EfficientSAM + top 20% depth points with outlier rejection
- `canopy_utils.py` — Interactive utility for selecting prompts, verifying tags, and checking depth

### `femto_bolt_code/scripts/canopy_detector/automation/` — Batch canopy processing

- `process_all_captures.py` — Runs canopy detection on all capture subfolders
- `create_prompts_batch.py` — Generates template prompt JSON files for batch processing
- `compile_results.py` — Compiles all canopy height JSON results into a single CSV

### `femto_bolt_code/scripts/leaf_segmentation/` — Leaf segmentation

- `leaf_semgent_using_vitt.py` — Two-stage leaf segmentation: whole plant mask then individual leaf masks using EfficientSAM

### `femto_bolt_code/scripts/TDFS/` — Turntable 3D reconstruction (TSDF fusion)

- `main_script.py` — Point cloud fusion and TSDF mesh reconstruction from multi-angle captures
- `april_tag_pivot_generator.py` — Finds turntable pivot point using AprilTag detections across frames
- `find_pivot.py` — Pivot point finder using depth-derived 3D corners
- `optimize_pivot.py` — Optimizes pivot X/Z to minimize alignment error
- `transform_json.py` — Generates NerfStudio-compatible transforms.json from captures
- `transform_json_by_motion.py` — Generates transforms.json using inter-frame motion estimation
- `background_masking_test.py` — Tests depth-based background masking for object isolation
- `batch_process_masking.py` — Batch applies background masking across a capture session
- `debug_2_view_test.py` — Debug: fuses two views to verify rotation alignment
- `debug_angle.py` — Debug: tests rotation angles for alignment correctness
- `debug_ply.py` — Debug: visualizes intermediate PLY outputs
- `transform_test_debug.py` — Debug: step-by-step transform verification with PLY output

### `femto_bolt_code/scripts/foundation_pose_localization/` — Foundation pose

- `6dof_icp_export.py` — ICP-based 6DOF pose export
- `hs_isolation/april_tag_bg_removal_pl.py` — AprilTag-guided background removal for hydroponic system isolation

### `femto_bolt_code/flora/` — SAM3 integration

- `sam3_depth_to_pointcloud.py` — Segments objects using SAM3 text/box prompts, projects masks to 3D, exports colored PLY point clouds
- `sam3-repo/` — Cloned SAM3 repository (Facebook Research)

### `realsense_d415i/april_tag_detection_caliberation/` — RealSense AprilTag and calibration

- `realtime_pose_estimation_april_tag.py` — Real-time AprilTag pose estimation with RealSense
- `april_tag_detection_image.py` — AprilTag detection from saved images
- `checkerboard_caliberation.py` — Checkerboard camera calibration for RealSense
- `fetch_factory_intrinsic.py` — Fetches factory intrinsics from RealSense device
- `fetch_factory_extrinsic.py` — Fetches factory extrinsics from RealSense device

### `realsense_d415i/capture_scripts/` — RealSense capture

- `capture_aligned_all.py` — Captures aligned RGB, depth, and PLY simultaneously
- `capture_aligned_pointcloud.py` — Captures aligned point clouds
- `distance_masking_on_ply.py` — Applies distance-based masking to PLY files
- `visualize_ply.py` — Visualizes PLY point clouds in Open3D

### `realsense_d415i/canopy_detection/` — RealSense canopy detection

- `canopy_return.py` — Canopy height detection using depth data
- `canopy_return_upgraded.py` — Improved canopy detection with better filtering
- `combined-logic.py` — Combined canopy detection pipeline
- `image_capture.py` — Captures images for canopy detection
- `bag_to_img.py` — Extracts images from RealSense .bag recording files

### `realsense_d415i/testing_scripts/` — RealSense testing

- `test_camera_status.py` — Tests RealSense camera connection status
- `supported_stream_list.py` — Lists supported stream configurations
- `color_live_streaming.py` — Live color stream preview at 640x480 or 1280x720 (`--res`)
- `visualize_point_cloud.py` — Point cloud visualization
- `three_pose_vertical_translation_validation.py` — Validates vertical translation across three poses

### `realsense_d415i/vis_tool/` — RealSense visualization

- `vis_tool_solvepnp.py` — SolvePnP pose visualization tool
- `vis_tool_april_tag_pose_validaiton.py` — AprilTag pose validation visualization
- `vis_tool_not_working_ref.py` — Reference file (non-functional, kept for notes)

### `captures/` — Captured data

- `realsense/` — RealSense captures (canopy images, depth snapshots, .bag files, canopy detection pipeline outputs)
- `femto_bolt/` — Femto Bolt captures (hydroponic system data, four-pose validation)
- `03-04_FLORA_captures/` through `2026-04-24_*` — Turntable capture sessions at various dates and lighting conditions
- `canopy_test_captures_*/` — Canopy height test captures at different plant heights (33.4cm to 62cm)

### `captures/realsense/canopy-detection/` — Canopy detection pipeline

- `canopy_height_pipeline.py` — End-to-end canopy height pipeline using AprilTags + EfficientSAM
- `canopy_utils.py` — Interactive utility for prompt selection and tag verification

### `cad_model/` — CAD models (mm scale)

STL, PLY, and OBJ files of the hydroponic structure for point cloud overlay and error analysis.

### `cad_model_in_m/` — CAD models (meter scale)

Same structure model in meters for pipelines that work in metric units.

### `EfficientSAM/` — EfficientSAM repository

Cloned EfficientSAM repo used for plant segmentation in canopy detection and leaf segmentation pipelines.

### `git_images/` — Documentation images

Example output images used in this README.

## Environment Setup

Both camera systems require **Python 3.11** and separate virtual environments.

### Femto Bolt virtual environment

```bash
cd femto_bolt_code
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Key packages: `pyorbbecsdk`, `open3d`, `opencv-python`, `pupil_apriltags`, `numpy`, `matplotlib`

You also need the Orbbec SDK installed on your system. Download from the Orbbec developer portal and follow their installation instructions. `pyorbbecsdk` wraps the native SDK, so the Orbbec drivers must be installed for the camera to be detected.

### RealSense virtual environment

```bash
cd realsense_d415i
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
git clone https://github.com/yformer/EfficientSAM.git
git clone https://github.com/facebookresearch/sam3.git
```

Try running canopy_detection_old/canopy_img_based.py to make sure efficientSAM is working.
Next try running sam3
For that enter sam3_canopy_detection folder and then run: python sam3_test.py
You may encounter torch and torchvision error.

Use this command: 
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install huggingface_hub matplotlib
hf auth login
```

Run 
```bash
python .\sam3_test.py ..\canopy_detection_old\test_images\color_frame.png 
```
Sam3 should now start donwloading weights and run with the test

Key packages: `pyrealsense2`, `open3d`, `opencv-python`, `pupil_apriltags`, `numpy`

Intel RealSense SDK 2.0 must be installed. The `pyrealsense2` pip package wraps the native SDK.

### EfficientSAM setup

EfficientSAM is used by the canopy detection and leaf segmentation scripts. It is already cloned in `EfficientSAM/`.

```bash
# From within the femto_bolt or realsense venv:
pip install torch torchvision   # CUDA version recommended for GPU acceleration
```

The scripts load EfficientSAM by adding `EfficientSAM/` to `sys.path` at runtime. No separate install is needed — just make sure `torch` and `torchvision` are in your active venv. Model weights are downloaded automatically on first run.

### SAM3 setup

SAM3 is used by `femto_bolt_code/flora/sam3_depth_to_pointcloud.py`. The repo is cloned at `femto_bolt_code/flora/sam3-repo/`.

```bash
# From within the realsense venv (which already has torch):
pip install -e femto_bolt_code/flora/sam3-repo
```

SAM3 requires `torch`, `torchvision`, `timm`, `huggingface_hub`, and other ML dependencies. The RealSense `requirements.txt` already includes these. Model weights are downloaded from HuggingFace which reuires authentication on first use.
