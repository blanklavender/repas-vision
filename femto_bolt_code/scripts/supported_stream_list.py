from pyorbbecsdk import *

def print_stream_profiles(sensor_type, sensor_name):
    pipeline = Pipeline()
    profile_list = pipeline.get_stream_profile_list(sensor_type)
    if profile_list is None:
        print(f"No {sensor_name} profiles found.")
        return

    print(f"\n=== {sensor_name} supported profiles ===")
    for i in range(len(profile_list)):
        sp = profile_list[i]
        try:
            # Only video stream profiles have width/height; otherwise skip
            vsp = sp.as_video_stream_profile()
            print(f"- {vsp.get_width()} x {vsp.get_height()} @ {vsp.get_fps()} FPS | format={sp.get_format()}")
        except:
            # Non-video profile, ignore
            pass

if __name__ == "__main__":
    print_stream_profiles(OBSensorType.COLOR_SENSOR, "COLOR")   # RGB camera
    print_stream_profiles(OBSensorType.DEPTH_SENSOR, "DEPTH")   # Depth camera
