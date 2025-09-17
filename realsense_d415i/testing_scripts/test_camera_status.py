import pyrealsense2 as rs

# Create a context object. This object will manage all connected RealSense devices.
ctx = rs.context()

# Query the connected devices.
devices = ctx.query_devices()

if len(devices) == 0:
    print("No RealSense devices were found")
else:
    print("Found RealSense devices:")
    for dev in devices:
        print("Device Name:", dev.get_info(rs.camera_info.name))
        print("Serial Number:", dev.get_info(rs.camera_info.serial_number))
