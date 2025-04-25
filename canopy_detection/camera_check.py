import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) > 0:
    print("RealSense camera is connected.")
else:
    print("No RealSense camera connected.")
