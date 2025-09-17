import json, pyrealsense2 as rs, numpy as np

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)

depth_vsp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
color_vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
extr = depth_vsp.get_extrinsics_to(color_vsp)

R_dc = np.array(extr.rotation, dtype=float).reshape(3,3)   # depth -> color rotation
t_dc = np.array(extr.translation, dtype=float).reshape(3,) # depth -> color translation (m)

print("R_dc:\n", R_dc)
print("t_dc:", t_dc)

with open("factory_d2c_extrinsics.json","w") as f:
    json.dump({"R_dc": R_dc.tolist(), "t_dc": t_dc.tolist()}, f, indent=2)

pipe.stop()
