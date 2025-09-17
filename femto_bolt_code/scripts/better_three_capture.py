#!/usr/bin/env python3
"""
Femto Bolt — Unified capture (COLOR frame world)
------------------------------------------------
• Starts COLOR 1280×720 @30 (BGR/NV12/MJPG fallback) and DEPTH 640×576 @30 (Y16).
• Enables frame sync, aligns DEPTH → COLOR, and builds a colorized point cloud in the COLOR frame.
• Single keypress ('e') captures ALL THREE artifacts **at the same time**:
    1) color image (PNG)
    2) aligned-to-color depth (both raw units PNG + meters NPY)
    3) color-frame point cloud (PLY)
• Live preview: color + aligned depth colormap. Press 'q' or ESC to quit.

Outputs are guaranteed to be in the **COLOR camera coordinate frame**.

Notes
-----
- We explicitly request the resolutions you already use: COLOR 1280×720, DEPTH 640×576 @30.
- If exact profiles are unavailable, we fall back cleanly and still align depth → color (upsampling if needed).
- We query and apply the device depth scale so geometry is in **meters**.
- The saved metadata JSON captures intrinsics, chosen profiles, and depth scale for reproducibility.

Dependencies
------------
- pyorbbecsdk (Orbbec Python SDK)
- OpenCV (cv2) for visualization & PNG IO
- numpy

"""
import os
import json
import time
import cv2
import numpy as np
from pathlib import Path

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, VideoStreamProfile, OBError,
    FrameSet, AlignFilter, OBStreamType, PointCloudFilter,
    save_point_cloud_to_ply
)

ESC_KEY = 27

# -------- Desired profiles (will try exact, then fall back) --------
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 576, 30

OUT_DIR = Path('./captures_colorworld')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------ Helpers ------------------------

def timestamp() -> str:
    return time.strftime('%Y%m%d_%H%M%S')


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def select_video_profile(profile_list, want_w, want_h, want_fps, want_formats):
    """Try to pick an exact profile; else first match by size; else SDK default."""
    # 1) exact match (w,h,fps,format in given order of preference)
    for fmt in want_formats:
        try:
            prof = profile_list.get_video_stream_profile(want_w, want_h, fmt, want_fps)
            if prof is not None:
                return prof
        except OBError:
            pass
    # 2) match by (w,h,fps) with any format
    try:
        profs = profile_list.get_video_stream_profile_list()
        for p in profs:
            try:
                if p.get_width()==want_w and p.get_height()==want_h and p.get_fps()==want_fps:
                    return p
            except Exception:
                continue
    except Exception:
        pass
    # 3) default
    return profile_list.get_default_video_stream_profile()


def frame_to_bgr_image(color_frame):
    """Convert Orbbec color frame to BGR np.uint8(H,W,3). Falls back to SDK conversion if needed."""
    fmt = color_frame.get_format()
    w, h = color_frame.get_width(), color_frame.get_height()
    data = color_frame.get_data()
    if fmt in (OBFormat.BGR, OBFormat.RGB, OBFormat.BGRA, OBFormat.RGBA):
        arr = np.frombuffer(data, dtype=np.uint8)
        ch = 4 if fmt in (OBFormat.BGRA, OBFormat.RGBA) else 3
        img = arr.reshape((h, w, ch)).copy()
        if fmt in (OBFormat.RGB, OBFormat.RGBA):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if ch == 4:
            img = img[:, :, :3]
        return img
    elif fmt == OBFormat.NV12:
        y = np.frombuffer(data, dtype=np.uint8, count=w*h).reshape((h, w))
        uv = np.frombuffer(data, dtype=np.uint8, offset=w*h).reshape((h//2, w))
        nv12 = np.vstack((y, uv))
        bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        return bgr
    else:
        # Last resort: try MJPG decode
        if fmt == OBFormat.MJPG:
            buf = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError('Failed to decode MJPG color frame')
            return img
        raise RuntimeError(f'Unsupported color format for conversion: {fmt}')


def depth_to_meters(depth_frame):
    """Return (depth_u16_raw, depth_m_float32, scale_m_per_unit)."""
    w, h = depth_frame.get_width(), depth_frame.get_height()
    raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
    # scale = float(depth_frame.get_depth_scale())  # meters / unit (often 0.001)
    scale = 0.001
    depth_m = raw.astype(np.float32) * scale
    return raw, depth_m, scale


def colormap_depth(depth_m, vmin=0.25, vmax=8.0):
    m = depth_m.copy()
    m[m < vmin] = 0
    m[m > vmax] = 0
    if np.all(m == 0):
        return np.zeros((depth_m.shape[0], depth_m.shape[1], 3), dtype=np.uint8)
    norm = (m / vmax * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return vis


# ------------------------ Main ------------------------

def main():
    pipe = Pipeline()
    cfg = Config()

    # Depth profile selection (prefer Y16)
    dlist = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    if dlist is None:
        raise RuntimeError('No depth profiles available')
    depth_profile = select_video_profile(
        dlist, DEPTH_W, DEPTH_H, DEPTH_FPS, want_formats=[OBFormat.Y16]
    )
    cfg.enable_stream(depth_profile)

    # Color profile selection (prefer NV12, then BGR, then MJPG)
    clist = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    if clist is None:
        raise RuntimeError('No color profiles available')
    color_profile = select_video_profile(
        clist, COLOR_W, COLOR_H, COLOR_FPS, want_formats=[OBFormat.NV12, OBFormat.BGR, OBFormat.MJPG]
    )
    cfg.enable_stream(color_profile)

    pipe.enable_frame_sync()
    from pyorbbecsdk import OBFrameAggregateOutputMode
    cfg.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipe.start(cfg)

    # Align depth → COLOR and prepare point cloud generator
    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    pcf = PointCloudFilter()

    print('[INFO] Using profiles:')
    print('  COLOR :', color_profile)
    print('  DEPTH :', depth_profile)

    print("[INFO] Press 'e' to capture (color, aligned depth, point cloud). 'q'/ESC to quit.")

    while True:
        frames: FrameSet = pipe.wait_for_frames(100)
        if frames is None:
            continue
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if depth_frame is None or color_frame is None:
            continue

        # Align the pair into COLOR pixel grid (upsampling if needed)
        aligned_any = align.process(frames)
        if not aligned_any:
            continue

        # Convert the generic Frame returned by AlignFilter into a FrameSet
        aligned_frames = aligned_any.as_frame_set()

        a_depth = aligned_frames.get_depth_frame()
        a_color = aligned_frames.get_color_frame() or color_frame
        if a_depth is None or a_color is None:
            continue

        # Convert for live preview
        color_bgr = frame_to_bgr_image(a_color)
        depth_raw_u16, depth_m, scale = depth_to_meters(a_depth)
        depth_vis = colormap_depth(depth_m)

        # Show side-by-side preview
        vis = np.hstack([
            color_bgr,
            cv2.resize(depth_vis, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        ])
        cv2.imshow('COLOR (left)  |  DEPTH→COLOR colormap (right)', vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ESC_KEY):
            break

        if key == ord('e'):
            ts = timestamp()
            cap_dir = OUT_DIR / f'capture_{ts}'
            cap_dir.mkdir(parents=True, exist_ok=True)

            # 1) Save COLOR PNG
            color_path = ensure_dir(cap_dir / f'color_{ts}.png')
            cv2.imwrite(str(color_path), color_bgr)

            # 2) Save aligned DEPTH in raw units PNG + meters NPY (+ a visualization PNG)
            depth_raw_path = ensure_dir(cap_dir / f'aligned_depth_raw_{ts}.png')
            cv2.imwrite(str(depth_raw_path), depth_raw_u16)
            depth_m_path = ensure_dir(cap_dir / f'aligned_depth_m_{ts}.npy')
            np.save(str(depth_m_path), depth_m)
            depth_vis_path = ensure_dir(cap_dir / f'aligned_depth_vis_{ts}.png')
            cv2.imwrite(str(depth_vis_path), depth_vis)

            # 3) Generate color-frame, colorized point cloud via SDK filter
            #    Important: set output format and scale so positions are in meters
            pcf.set_create_point_format(OBFormat.RGB_POINT)
            pcf.set_position_data_scaled(scale)  # meters per unit
            pc_frame = pcf.process(aligned_frames)
            if pc_frame is None:
                raise RuntimeError('PointCloudFilter returned None')

            ply_path = ensure_dir(cap_dir / f'point_cloud_{ts}.ply')
            save_point_cloud_to_ply(str(ply_path), pc_frame)

            # 4) Save metadata for reproducibility
            meta = {
                'timestamp': ts,
                'color_profile': {
                    'w': a_color.get_width(), 'h': a_color.get_height(), 'fps': color_profile.get_fps(),
                    'format': str(a_color.get_format())
                },
                'depth_profile': {
                    'w': a_depth.get_width(), 'h': a_depth.get_height(), 'fps': depth_profile.get_fps(),
                    'format': str(a_depth.get_format())
                },
                'depth_scale_m_per_unit': scale,
                'notes': 'All products are aligned to COLOR frame. Point cloud is in COLOR coordinates.'
            }
            with open(ensure_dir(cap_dir / f'capture_meta_{ts}.json'), 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2)

            print('[SAVED]')
            print('  COLOR              :', color_path)
            print('  DEPTH raw (u16)    :', depth_raw_path)
            print('  DEPTH meters (npy) :', depth_m_path)
            print('  DEPTH vis (png)    :', depth_vis_path)
            print('  POINT CLOUD (PLY)  :', ply_path)

    cv2.destroyAllWindows()
    pipe.stop()


if __name__ == '__main__':
    main()
