Make sure to have python = 3.9.13

Make a virtual environment by running "python -m venv .venv"

Activate virtual environment.

Once in virtual environment, run the command: "pip install -r requirements.txt"

Run the files by running the command: "python {filename}"

/canopy_detection
bag_to_img.py => converts bag files to images
combined-logic.py => will implement edge detection, and background extraction
image_capture.py => captures image from camera

/april_tag_code
april_tag_detector.py => detects april tag, draws bounding boxes and then displays coordinates realtime
intrsinsic-1.py (Author: Valentina) => Finds camera intrinsics
instrinsic-2.py (Author: Valentina) => Applies transformations and displays a 3d plot using Plotly
stl_overlay.py => Detects april tag, maps stl object coordinate with the april tag center


