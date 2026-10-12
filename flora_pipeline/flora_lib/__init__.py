"""
flora_lib  --  shared library of the Flora turntable pipeline.

    dataset       the capture-folder layout every step reads and writes (file names,
                  manifest.csv, camera_params.json, cleaning.json, picked-points file)
    geometry      rigid transforms, camera rays / projection, voxel fusion
    tags          AprilTag detection + colour-only PnP poses, pairwise tag-pose registration
    registration  chaining + loop closure over a sweep, turntable-axis "plant frame"

The step scripts (sweep_capture.py, sam3_data_cleaning.py, tag_pose_pick.py) import from
here; they live in other folders and put this folder on sys.path themselves.
"""
