# save_color_intrinsics.py
import json
import pyrealsense2 as rs

# Set this to the SAME resolution you used for your RGB captures
RES_W, RES_H, FPS = 1280, 720, 15
OUT_JSON = "factory_color_intrinsics_{}_{}.json".format(RES_W, RES_H)

pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, RES_W, RES_H, rs.format.bgr8, FPS)
profile = pipe.start(cfg)

# let auto-exposure settle a bit
for _ in range(5):
    pipe.wait_for_frames()

color_vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr = color_vsp.get_intrinsics()
serial = profile.get_device().get_info(rs.camera_info.serial_number)

# intr.model is an enum; store both numeric and a best-effort name
model_name = str(intr.model)  # e.g. "rs.distortion.brown_conrady"

data = {
    "device_serial": serial,
    "width":  intr.width,
    "height": intr.height,
    "fx": intr.fx,
    "fy": intr.fy,
    "ppx": intr.ppx,
    "ppy": intr.ppy,
    # RealSense gives up to 5 Brown-Conrady coeffs for color
    "coeffs": list(intr.coeffs[:5]),
    "distortion_model": model_name
}

with open(OUT_JSON, "w") as f:
    json.dump(data, f, indent=2)

pipe.stop()
print(f"[OK] Saved color intrinsics → {OUT_JSON}")
