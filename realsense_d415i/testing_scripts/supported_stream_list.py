import pyrealsense2 as rs

ctx    = rs.context()
dev    = ctx.query_devices()[0]
print("Device:", dev.get_info(rs.camera_info.name))

# z16 is depth, bgr8 is the standard color profile used throughout

for sensor in dev.query_sensors():
    print("\nSensor:", sensor.get_info(rs.camera_info.name))
    for profile in sensor.get_stream_profiles():
        if profile.is_video_stream_profile():
            vsp = profile.as_video_stream_profile()
            print(f"  → {vsp.width()}x{vsp.height()} @ {profile.fps()}fps, fmt={vsp.format()}")
