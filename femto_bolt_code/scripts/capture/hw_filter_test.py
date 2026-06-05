# HW noise removal filter: NOT supported on this device
# Default Software Filters
# Noise removal filter enabled: False
# Current max_diff=10, max_speckle_size=50
# Noise removal filter: enabled, max_diff=512, max_speckle_size=960

from pyorbbecsdk import Pipeline, OBPropertyID, OBPermissionType

pipeline = Pipeline()

# 1.Get device by Pipeline
device = pipeline.get_device()

# 2.Determine whether to enable filter
enable = device.get_bool_property(OBPropertyID.OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL)
print(f"Noise removal filter enabled: {enable}")

# 3.Get filter-related parameters
max_diff = device.get_int_property(OBPropertyID.OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_MAX_DIFF_INT)
min_diff = device.get_int_property(OBPropertyID.OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_MAX_SPECKLE_SIZE_INT)
print(f"Current max_diff={max_diff}, max_speckle_size={min_diff}")

# 4.Set the relevant filter parameters
device.set_bool_property(OBPropertyID.OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL, True)
device.set_int_property(OBPropertyID.OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_MAX_DIFF_INT, 512)
device.set_int_property(OBPropertyID.OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_MAX_SPECKLE_SIZE_INT, 960)
print("Noise removal filter: enabled, max_diff=512, max_speckle_size=960")