#!/usr/bin/env python3
"""
Femto Bolt — Session capture (burst, angle-indexed)
---------------------------------------------------
Captures 5-frame bursts of raw COLOR (PNG) and aligned-to-color DEPTH
(uint16 PNG) at each rotation angle. Turntable must remain STILL during
the burst — press 'e' only after the turntable has stopped moving.

Filters are explicitly DISABLED before capture to produce raw data for
offline filter experimentation. Before/after filter states are logged
in meta.json.

Naming convention
-----------------
    <base_dir>/<session>/
        color_000_f00.png      # 0°, burst frame 0
        color_000_f01.png      # 0°, burst frame 1
        ...
        color_000_f04.png      # 0°, burst frame 4
        depth_000_f00.png
        ...
        color_010_f00.png      # 10°
        ...
        meta.json

Controls
--------
    e   — capture 5-frame burst at current angle, auto-advance
    s   — skip current angle without capturing
    u   — undo last burst (deletes all 5 frame pairs, steps angle back)
    q / ESC — finish session and write meta.json
"""
import os
import json
import time
import cv2
import numpy as np
from pathlib import Path

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, OBError,
    FrameSet, AlignFilter, OBStreamType, PointCloudFilter,
    OBPropertyID, OBPermissionType,
)

ESC_KEY = 27

# -------- Desired stream profiles --------
COLOR_W, COLOR_H, COLOR_FPS = 1280, 720, 30
DEPTH_W, DEPTH_H, DEPTH_FPS = 640, 576, 30

# -------- Session parameters --------
BASE_DIR      = Path('../../../captures')
PLANT_NAME    = 'april_tag'
LIGHTING      = 'no_led'
ANGLE_START   = 0
ANGLE_STEP    = 10                 # CHANGED: 5 → 10
ANGLE_END     = 350                # CHANGED: 36 shots for 0..350 @ 10°
BURST_SIZE    = 5                  # NEW: frames per angle
DEPTH_SCALE   = 0.001

# -------- Filter properties to disable for raw capture --------
# (name, property_id) — add more here as you discover them
FILTER_PROPS = [
    ('OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL',
     OBPropertyID.OB_PROP_DEPTH_NOISE_REMOVAL_FILTER_BOOL),
    ('OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL',
     OBPropertyID.OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL),
]

# ----------------------------------------------------------------

def make_session_dir() -> Path:
    date_str = time.strftime('%Y-%m-%d')
    name = f'{date_str}_{PLANT_NAME}_{LIGHTING}_raw'
    p = BASE_DIR / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def disable_filters_and_log(device) -> dict:
    """Query all filter props, force to False, return before/after states."""
    states = {}
    for name, prop in FILTER_PROPS:
        entry = {'supported_read': False, 'supported_write': False,
                 'before': None, 'after': None}
        try:
            if device.is_property_supported(prop, OBPermissionType.PERMISSION_READ):
                entry['supported_read'] = True
                entry['before'] = bool(device.get_bool_property(prop))
            if device.is_property_supported(prop, OBPermissionType.PERMISSION_WRITE):
                entry['supported_write'] = True
                device.set_bool_property(prop, False)
                if entry['supported_read']:
                    entry['after'] = bool(device.get_bool_property(prop))
        except Exception as e:
            entry['error'] = str(e)
        states[name] = entry
        print(f"  [filter] {name}: before={entry['before']}  after={entry['after']}")
    return states


def select_video_profile(profile_list, want_w, want_h, want_fps, want_formats):
    for fmt in want_formats:
        try:
            prof = profile_list.get_video_stream_profile(want_w, want_h, fmt, want_fps)
            if prof is not None:
                return prof
        except OBError:
            pass
    try:
        profs = profile_list.get_video_stream_profile_list()
        for p in profs:
            try:
                if (p.get_width() == want_w and p.get_height() == want_h
                        and p.get_fps() == want_fps):
                    return p
            except Exception:
                continue
    except Exception:
        pass
    return profile_list.get_default_video_stream_profile()


def frame_to_bgr(color_frame):
    fmt = color_frame.get_format()
    w, h = color_frame.get_width(), color_frame.get_height()
    data = color_frame.get_data()
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
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError('MJPG decode failed')
        return img
    raise RuntimeError(f'Unsupported color format: {fmt}')


def depth_raw_u16(depth_frame):
    w, h = depth_frame.get_width(), depth_frame.get_height()
    return np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))


def colormap_depth(depth_u16, scale=DEPTH_SCALE, vmin=0.25, vmax=8.0):
    m = depth_u16.astype(np.float32) * scale
    m[(m < vmin) | (m > vmax)] = 0
    norm = (m / vmax * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def angle_str(deg: int) -> str:
    return f'{deg:03d}'


def grab_aligned_frameset(pipe, align, timeout_ms=500):
    """Grab one frameset and return (color_bgr, depth_u16) or None."""
    frames: FrameSet = pipe.wait_for_frames(timeout_ms)
    if frames is None:
        return None
    if frames.get_depth_frame() is None or frames.get_color_frame() is None:
        return None
    aligned_any = align.process(frames)
    if not aligned_any:
        return None
    aligned = aligned_any.as_frame_set()
    a_depth = aligned.get_depth_frame()
    a_color = aligned.get_color_frame() or frames.get_color_frame()
    if a_depth is None or a_color is None:
        return None
    return frame_to_bgr(a_color), depth_raw_u16(a_depth)


def capture_burst(pipe, align, session_dir, angle, burst_size,
                  first_color, first_depth):
    """Save first frame + (burst_size-1) more consecutive frames."""
    astr = angle_str(angle)
    saved = []

    # Save the already-grabbed first frame
    c_path = session_dir / f'color_{astr}_f00.png'
    d_path = session_dir / f'depth_{astr}_f00.png'
    cv2.imwrite(str(c_path), first_color)
    cv2.imwrite(str(d_path), first_depth)
    saved.append(0)

    # Grab remaining frames
    for bi in range(1, burst_size):
        result = None
        # retry up to 5 times if a frameset comes back empty
        for _ in range(5):
            result = grab_aligned_frameset(pipe, align, timeout_ms=500)
            if result is not None:
                break
        if result is None:
            print(f'    [warn] burst frame {bi} dropped — skipping')
            continue
        c_bgr, d_u16 = result
        c_path = session_dir / f'color_{astr}_f{bi:02d}.png'
        d_path = session_dir / f'depth_{astr}_f{bi:02d}.png'
        cv2.imwrite(str(c_path), c_bgr)
        cv2.imwrite(str(d_path), d_u16)
        saved.append(bi)
    return saved


def delete_burst(session_dir, angle, burst_size):
    """Remove all burst frames for a given angle."""
    astr = angle_str(angle)
    for bi in range(burst_size):
        for kind in ('color', 'depth'):
            p = session_dir / f'{kind}_{astr}_f{bi:02d}.png'
            if p.exists():
                p.unlink()

# ----------------------------------------------------------------

def main():
    session_dir = make_session_dir()
    angles = list(range(ANGLE_START, ANGLE_END + 1, ANGLE_STEP))
    total = len(angles)
    current_idx = 0
    captured_angles = []

    # ---------- Pipeline setup ----------
    pipe = Pipeline()
    cfg = Config()

    dlist = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = select_video_profile(
        dlist, DEPTH_W, DEPTH_H, DEPTH_FPS, [OBFormat.Y16])
    cfg.enable_stream(depth_profile)

    clist = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = select_video_profile(
        clist, COLOR_W, COLOR_H, COLOR_FPS, [OBFormat.NV12, OBFormat.BGR, OBFormat.MJPG])
    cfg.enable_stream(color_profile)

    pipe.enable_frame_sync()
    from pyorbbecsdk import OBFrameAggregateOutputMode
    cfg.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

    # ---- NEW: disable filters BEFORE starting pipeline ----
    print('[FILTERS] Disabling depth filters for raw capture:')
    device = pipe.get_device()
    filter_states = disable_filters_and_log(device)

    pipe.start(cfg)
    align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    try:
        cam_params = pipe.get_camera_param()
    except Exception:
        cam_params = None

    print(f'[SESSION] {session_dir}')
    print(f'[INFO]    {total} angles: {angles[0]}° – {angles[-1]}° '
          f'(step {ANGLE_STEP}°, burst {BURST_SIZE})')
    print(f'[INFO]    COLOR {color_profile}  |  DEPTH {depth_profile}')
    print()
    print("  'e' capture burst  |  's' skip  |  'u' undo  |  'q'/ESC finish")
    print()

    while current_idx < total:
        result = grab_aligned_frameset(pipe, align, timeout_ms=100)
        if result is None:
            continue
        color_bgr, raw_u16 = result

        # Live preview
        depth_vis = colormap_depth(raw_u16)
        depth_vis_resized = cv2.resize(
            depth_vis, (color_bgr.shape[1], color_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST)
        vis = np.hstack([color_bgr, depth_vis_resized])

        ang = angles[current_idx]
        status = (f'Angle: {ang} deg  |  Shot {current_idx + 1}/{total}  |  '
                  f'Captured: {len(captured_angles)}  |  Burst: {BURST_SIZE}')
        cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "e=capture burst  s=skip  u=undo  q=quit", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Capture Session (raw, burst)', vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), ESC_KEY):
            break

        if key == ord('e'):
            print(f'  [{len(captured_angles) + 1:>3}/{total}]  {ang:>3}°  '
                  f'capturing burst of {BURST_SIZE}...')
            saved = capture_burst(pipe, align, session_dir, ang,
                                  BURST_SIZE, color_bgr, raw_u16)
            if len(saved) == BURST_SIZE:
                captured_angles.append(ang)
                print(f'           ✓ {len(saved)} frames saved')
            else:
                # partial burst — still delete and warn
                delete_burst(session_dir, ang, BURST_SIZE)
                print(f'           ✗ only {len(saved)}/{BURST_SIZE} — '
                      f'discarded, please retry')
                continue
            current_idx += 1

        elif key == ord('s'):
            print(f'  [skip]  {angles[current_idx]}°')
            current_idx += 1

        elif key == ord('u'):
            if captured_angles:
                last = captured_angles.pop()
                delete_burst(session_dir, last, BURST_SIZE)
                current_idx = angles.index(last)
                print(f'  [undo]  {last}° removed ({BURST_SIZE} frames)')
            else:
                print('  [undo]  nothing to undo')

    # ---------- Write metadata ----------
    meta = {
        'session_dir': str(session_dir),
        'date': time.strftime('%Y-%m-%d'),
        'plant': PLANT_NAME,
        'lighting': LIGHTING,
        'angle_start': ANGLE_START,
        'angle_step': ANGLE_STEP,
        'angle_end': ANGLE_END,
        'burst_size': BURST_SIZE,
        'total_planned': total,
        'captured_angles': captured_angles,
        'skipped_angles': sorted(set(angles) - set(captured_angles)),
        'depth_scale_m_per_unit': DEPTH_SCALE,
        'filter_states': filter_states,
        'alignment_mode': 'software_AlignFilter_to_color',
        'color_profile': {
            'w': color_profile.get_width(),
            'h': color_profile.get_height(),
            'fps': color_profile.get_fps(),
            'format': str(color_profile.get_format()),
        },
        'depth_profile': {
            'w': depth_profile.get_width(),
            'h': depth_profile.get_height(),
            'fps': depth_profile.get_fps(),
            'format': str(depth_profile.get_format()),
        },
        'notes': 'RAW capture for offline filter experimentation. Depth is '
                 'aligned to COLOR (software align). Each angle has BURST_SIZE '
                 'frames named color_<deg>_f<NN>.png / depth_<deg>_f<NN>.png. '
                 'Hardware/software depth filters explicitly disabled — see '
                 'filter_states.',
    }

    if cam_params is not None:
        try:
            intr = cam_params.rgb_intrinsic
            meta['color_intrinsics'] = {
                'fx': intr.fx, 'fy': intr.fy,
                'cx': intr.cx, 'cy': intr.cy,
                'width': intr.width, 'height': intr.height,
            }
        except Exception:
            pass
        try:
            intr = cam_params.depth_intrinsic
            meta['depth_intrinsics'] = {
                'fx': intr.fx, 'fy': intr.fy,
                'cx': intr.cx, 'cy': intr.cy,
                'width': intr.width, 'height': intr.height,
            }
        except Exception:
            pass

    meta_path = session_dir / 'meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print()
    print(f'[DONE] {len(captured_angles)}/{total} angles captured '
          f'({len(captured_angles) * BURST_SIZE} frame pairs)')
    print(f'[META] {meta_path}')

    cv2.destroyAllWindows()
    pipe.stop()


if __name__ == '__main__':
    main()