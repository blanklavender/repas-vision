#!/usr/bin/env python3
"""
Alignment verification for Femto Bolt SW D2C (AlignFilter).
  (1) color-edge-on-depth overlay                  (qualitative, live)
  (2) colored PLY for MeshLab inspection           (qualitative, on 'e')
  (3) AprilTag corners -> planarity + edge length  (quantitative, on 'e')
"""
import cv2
import numpy as np
from pathlib import Path
from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, OBStreamType,
    AlignFilter, PointCloudFilter, OBFrameAggregateOutputMode,
    save_point_cloud_to_ply,
)
from pupil_apriltags import Detector   # swap for whichever detector you use

OUT = Path("./align_test"); OUT.mkdir(exist_ok=True)
TAG_SIZE_M = 0.050   # <-- set to the black-border edge length of your tag

# color intrinsics from your notes (Femto Bolt 1280x720)
K = dict(fx=748.8987, fy=748.3514, cx=639.87, cy=361.9517)

def decode_color(frame):
    fmt, w, h = frame.get_format(), frame.get_width(), frame.get_height()
    buf = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if fmt == OBFormat.MJPG: return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if fmt == OBFormat.BGRA: return cv2.cvtColor(buf.reshape((h,w,4)), cv2.COLOR_BGRA2BGR)
    if fmt == OBFormat.RGB:  return cv2.cvtColor(buf.reshape((h,w,3)), cv2.COLOR_RGB2BGR)
    raise RuntimeError(f"unhandled fmt {fmt}")

def main():
    pipe, cfg = Pipeline(), Config()
    cprof = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)\
                .get_video_stream_profile(1280, 720, OBFormat.MJPG, 30)
    dprof = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)\
                .get_video_stream_profile(640, 576, OBFormat.Y16, 30)
    cfg.enable_stream(cprof); cfg.enable_stream(dprof)
    cfg.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipe.enable_frame_sync(); pipe.start(cfg)

    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    pcf = PointCloudFilter(); pcf.set_create_point_format(OBFormat.RGB_POINT)
    det = Detector(families="tag36h11")

    print("Place an AprilTag 0.3-0.8 m from camera. 'e' = evaluate, 'q'/ESC = quit.")
    while True:
        frames = pipe.wait_for_frames(100)
        if frames is None: continue
        aligned = align.process(frames)
        if aligned is None: continue
        fs = aligned.as_frame_set()
        cf, df = fs.get_color_frame(), fs.get_depth_frame()
        if cf is None or df is None: continue

        color = decode_color(cf)
        dh, dw = df.get_height(), df.get_width()
        depth_m = np.frombuffer(df.get_data(), dtype=np.uint16).reshape((dh, dw)).astype(np.float32) * 0.001

        # (1) live edge overlay
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        vis = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        vis[edges > 0] = (255, 255, 255)
        cv2.imshow("edges-on-depth", vis); cv2.imshow("color", color)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        if key != ord('e'): continue

        # (2) colored PLY
        pcf.set_position_data_scaled(0.001)
        save_point_cloud_to_ply(str(OUT/"cloud.ply"), pcf.process(fs))
        cv2.imwrite(str(OUT/"color.png"), color)
        cv2.imwrite(str(OUT/"overlay.png"), vis)

        # (3) AprilTag 3D-corner check
        tags = det.detect(gray, estimate_tag_pose=False)
        if not tags:
            print("no tag detected"); continue

        for t in tags:
            pts3d, bad = [], False
            for (u, v) in t.corners:
                ui, vi = int(round(u)), int(round(v))
                z = depth_m[vi, ui]
                if z == 0:
                    patch = depth_m[max(0,vi-1):vi+2, max(0,ui-1):ui+2]
                    nz = patch[patch > 0]
                    if nz.size == 0: bad = True; break
                    z = float(np.median(nz))
                x = (ui - K['cx']) * z / K['fx']
                y = (vi - K['cy']) * z / K['fy']
                pts3d.append([x, y, z])
            if bad: print(f"tag {t.tag_id}: missing corner depth"); continue
            P = np.asarray(pts3d)

            # planarity RMSE
            c = P.mean(0)
            n = np.linalg.svd(P - c)[2][-1]
            plane_rmse_mm = 1000 * np.sqrt((((P - c) @ n) ** 2).mean())

            # edge lengths vs. physical size
            edges_m = np.linalg.norm(np.diff(P, axis=0, append=P[:1]), axis=1)
            edge_err_mm = 1000 * (edges_m - TAG_SIZE_M)

            print(f"\ntag {t.tag_id} @ {P[:,2].mean():.2f} m")
            print(f"  planarity RMSE : {plane_rmse_mm:.2f} mm   (good: <3)")
            print(f"  edge errors    : {np.round(edge_err_mm, 2).tolist()} mm   (good: <5)")

        print(f"saved -> {OUT/'cloud.ply'}  (inspect in MeshLab)")

    pipe.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()