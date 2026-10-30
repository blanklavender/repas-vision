# RealSense D415i testing scripts

Small standalone scripts for checking that the D415i is connected, seeing what it can stream, previewing the color feed, and a couple of offline checks (AprilTag pose deltas, point cloud viewer). Each one is run on its own; none of them import each other.

| Script | Needs camera | What it does |
|---|---|---|
| `test_camera_status.py` | yes | Lists every connected RealSense device with its name and serial number. Prints `No RealSense devices were found` if nothing is detected. Run this first. |
| `supported_stream_list.py` | yes | Prints every video stream profile of the first connected camera (`WIDTHxHEIGHT @ FPS, fmt=...`) for the stereo (depth/IR) and RGB sensors. Use it to confirm a resolution/fps combination before using it elsewhere. |
| `color_live_streaming.py` | yes | Opens a live `bgr8` color preview window at **640x480** (default) or **1280x720**. Press `q` in the window to quit. |
| `three_pose_vertical_translation_validation.py` | no | Loads three saved RGB images, detects a tag36h11 AprilTag in each (plain + CLAHE pass, retried with finer detector settings), solves the tag pose with `cv2.solvePnP`, and prints the translation between poses 1→2, 2→3 and 1→3, including the Y-axis (vertical) component. Uses the factory 1280x720 color intrinsics and a 30.3 mm tag. |
| `visualize_point_cloud.py` | no | Opens a `.ply` point cloud in an Open3D window with a coordinate frame at the origin and prints the point count and first 5 points. |

## Setup

The scripts use the virtual environment in `realsense_d415i/` (one level up). Python 3.11 is recommended because of Open3D.

Packages used by these scripts: `pyrealsense2`, `opencv-python`, `numpy`, `open3d`, `pupil-apriltags`.

### Windows (PowerShell)

```powershell
cd realsense_d415i
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install pyrealsense2 opencv-python numpy open3d pupil-apriltags
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.venv\Scripts\activate.bat` from `cmd`.

No driver install is needed on Windows 10/11; the camera shows up as soon as it is plugged in. Close the Intel RealSense Viewer before running a script, since only one program can open the camera at a time.

### Ubuntu

```bash
cd realsense_d415i
python3.11 -m venv .venv          # on Ubuntu 22.04: sudo apt install python3.11 python3.11-venv (deadsnakes PPA)
source .venv/bin/activate
pip install pyrealsense2 opencv-python numpy open3d pupil-apriltags
```

Install the RealSense udev rules once so the camera can be opened without `sudo`, then unplug and replug the camera:

```bash
sudo curl -L -o /etc/udev/rules.d/99-realsense-libusb.rules https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

The `pyrealsense2` wheel on PyPI is for x86_64. On ARM boards (Jetson, Raspberry Pi) build librealsense from source with `-DBUILD_PYTHON_BINDINGS=ON` instead.

## Running

Run from inside `testing_scripts/` with the venv active. The commands are the same on both systems except for the path separator.

```bash
cd testing_scripts
```

### Camera status

```bash
python test_camera_status.py
```

### Supported stream profiles

```bash
python supported_stream_list.py
```

### Live color stream

```bash
# 640x480 @ 30fps (default)
python color_live_streaming.py

# 1280x720 @ 30fps
python color_live_streaming.py --res 1280x720

# 1280x720 @ 15fps (e.g. on a USB 2 port)
python color_live_streaming.py --res 1280x720 --fps 15
```

| Option | Values | Default |
|---|---|---|
| `--res` | `640x480`, `1280x720` | `640x480` |
| `--fps` | `6`, `15`, `30` | `30` |

On start the script prints the resolution, fps and USB type (`3.2`, `2.1`, ...). The window is resizable, so a 1280x720 stream can be shrunk on a small screen. If the camera can't provide the requested mode, the script exits with a message instead of crashing in `pipeline.start`. This usually means the camera is on a USB 2 port or cable; plug it into a USB 3 port or lower `--fps`, and check `supported_stream_list.py` for what is available.

### AprilTag three-pose validation

Edit `IMAGE_PATHS` at the top of the script so it points at your three captured RGB images (the paths currently in the file are from an older folder layout and no longer exist). Relative paths resolve from the directory you run the script in. Optionally set `TARGET_TAG_ID` to force a specific tag.

```bash
python three_pose_vertical_translation_validation.py
```

The `K` matrix in the script is for **1280x720** color. Images captured at a different resolution need the matching intrinsics from `../april_tag_detection_caliberation/factory_color_intrinsics_*.json`.

### Point cloud viewer

Edit the path in `o3d.io.read_point_cloud(...)` to point at a `.ply` file (the current path no longer exists), for example one of `../capture_scripts/session_captures/pc_*.ply`.

```bash
python visualize_point_cloud.py
```

## Troubleshooting

- **`No RealSense devices were found` / `No device connected`**: check the cable, try a different USB 3 port, close RealSense Viewer or any other script using the camera. On Ubuntu, confirm the udev rules are installed and the camera was replugged afterwards; `lsusb | grep 8086` should list an Intel device.
- **`Couldn't resolve requests` or the "cannot stream" message**: the mode isn't offered on the current connection. Compare with `supported_stream_list.py`.
- **Frames time out (`Frame didn't arrive within 5000`)**: usually a power or bandwidth problem on a hub. Connect directly to the computer.
- **OpenCV window doesn't appear on Ubuntu over SSH**: `cv2.imshow` and Open3D need a display. Run on the machine's desktop session, or use `ssh -X`.
