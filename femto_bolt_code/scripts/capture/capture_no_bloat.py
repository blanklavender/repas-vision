#!/usr/bin/env python3
"""
Femto Bolt — RGB + aligned depth live preview.
'e' = save frame pair, 'q'/ESC = quit.
"""
import cv2
import numpy as np
from pathlib import Path

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, OBError,
    AlignFilter, OBStreamType, OBFrameAggregateOutputMode,
)

ESC = 27
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 576, 30
OUT_DIR = Path('./captures')


def select_profile(profile_list, w, h, fps, formats):
    for fmt in formats:
        try:
            p = profile_list.get_video_stream_profile(w, h, fmt, fps)
            if p is not None:
                return p
        except OBError:
            pass
    return profile_list.get_default_video_stream_profile()

# Being flexible with the color formats to account for USB bandwidth, can set to RGB only if that makes a difference
def color_to_bgr(frame):
    fmt = frame.get_format()
    w, h = frame.get_width(), frame.get_height()
    data = frame.get_data()
    if fmt in (OBFormat.BGR, OBFormat.RGB, OBFormat.BGRA, OBFormat.RGBA):
        ch = 4 if fmt in (OBFormat.BGRA, OBFormat.RGBA) else 3
        img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, ch)).copy()
        if fmt in (OBFormat.RGB, OBFormat.RGBA):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img[:, :, :3]
    if fmt == OBFormat.NV12:
        y = np.frombuffer(data, dtype=np.uint8, count=w * h).reshape((h, w))
        uv = np.frombuffer(data, dtype=np.uint8, offset=w * h).reshape((h // 2, w))
        return cv2.cvtColor(np.vstack((y, uv)), cv2.COLOR_YUV2BGR_NV12)
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    raise RuntimeError(f'Unsupported color format: {fmt}')


def depth_to_u16(frame):
    w, h = frame.get_width(), frame.get_height()
    return np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((h, w))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipe = Pipeline()
    cfg = Config()
    cfg.enable_stream(select_profile(
        pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR),
        DEPTH_W, DEPTH_H, DEPTH_FPS, [OBFormat.Y16]))
    cfg.enable_stream(select_profile(
        pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR),
        COLOR_W, COLOR_H, COLOR_FPS, [OBFormat.NV12, OBFormat.BGR, OBFormat.MJPG]))

    cfg.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipe.enable_frame_sync()
    pipe.start(cfg)
    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM) # SOFTWARE D2C ENABLED HERE

    print("'e' = capture  |  'q'/ESC = quit")
    idx = 0
    try:
        while True:
            frames = pipe.wait_for_frames(100)
            if frames is None:
                continue
            aligned = align.process(frames) # D2C ALIGNMENT USING FACTORY PARAMS HAPPENS HERE
            if not aligned:
                continue
            aligned = aligned.as_frame_set()
            c, d = aligned.get_color_frame(), aligned.get_depth_frame()
            if c is None or d is None:
                continue

            color = color_to_bgr(c)
            depth = depth_to_u16(d)

            cv2.imshow('color', color)
            cv2.imshow('depth', depth)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ESC):
                break
            if key == ord('e'):
                cv2.imwrite(str(OUT_DIR / f'color_{idx:03d}.png'), color)
                cv2.imwrite(str(OUT_DIR / f'depth_{idx:03d}.png'), depth)
                print(f'  saved {idx:03d}')
                idx += 1
    finally:
        cv2.destroyAllWindows()
        pipe.stop()


if __name__ == '__main__':
    main()