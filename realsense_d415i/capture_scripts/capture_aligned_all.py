import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs

"""
Usage:
------
Mouse: 
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.

Keyboard: 
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export point cloud (.ply) and 2D image (.png)
    [f]     Take a snapshot of RGB and depth images
    [q\ESC] Quit
"""

class AppState:
    def __init__(self):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return (Ry @ Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

state = AppState()

# -----------------------------------------------------------------------
# 1) Configure & start RealSense, set up alignment to color
# -----------------------------------------------------------------------
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Verify color sensor exists
pw = rs.pipeline_wrapper(pipeline)
pp = config.resolve(pw)
dev = pp.get_device()
if not any(s.get_info(rs.camera_info.name) == 'RGB Camera'
           for s in dev.sensors):
    print("Requires a RealSense device with Depth+Color")
    exit(0)

profile = pipeline.start(config)

# This is the key new bit: align depth → color
align = rs.align(rs.stream.color)

# Set up pointcloud + filters + colorizer
pc        = rs.pointcloud()
decimate  = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()

# Get the depth intrinsics once for window size
d_intr = profile.get_stream(rs.stream.depth) \
                .as_video_stream_profile() \
                .get_intrinsics()
w, h = d_intr.width, d_intr.height

# Prepare OpenCV window
cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)

# -----------------------------------------------------------------------
# 2) Define all your view‐projection & draw helpers (unchanged)
# -----------------------------------------------------------------------
def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: state.mouse_btns[0] = True
    if event == cv2.EVENT_LBUTTONUP:   state.mouse_btns[0] = False
    if event == cv2.EVENT_RBUTTONDOWN: state.mouse_btns[1] = True
    if event == cv2.EVENT_RBUTTONUP:   state.mouse_btns[1] = False
    if event == cv2.EVENT_MBUTTONDOWN: state.mouse_btns[2] = True
    if event == cv2.EVENT_MBUTTONUP:   state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:
        h_, w_ = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]
        if state.mouse_btns[0]:
            state.yaw   += float(dx) / w_ * 2
            state.pitch -= float(dy) / h_ * 2
        elif state.mouse_btns[1]:
            dp = np.array((dx / w_, dy / h_, 0), dtype=np.float32)
            state.translation -= state.rotation @ dp
        elif state.mouse_btns[2]:
            dz = math.hypot(dx, dy) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance       -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance       -= dz

    state.prev_mouse = (x, y)

cv2.setMouseCallback(state.WIN_NAME, mouse_cb)

def project(v):
    h_, w_ = out.shape[:2]
    asp = h_/w_
    with np.errstate(divide='ignore', invalid='ignore'):
        p = v[:, :-1] / v[:, -1, None] * (w_*asp, h_) + (w_/2, h_/2)
    p[v[:,2]<0.03] = np.nan
    return p

def view(v):
    return (v - state.pivot) @ state.rotation + state.pivot - state.translation

def line3d(img, a, b, col=(128,128,128), th=1):
    q1 = project(a.reshape(-1,3))[0]
    q2 = project(b.reshape(-1,3))[0]
    if np.isnan(q1).any() or np.isnan(q2).any(): return
    q1, q2 = tuple(q1.astype(int)), tuple(q2.astype(int))
    ok, q1, q2 = cv2.clipLine((0,0,img.shape[1],img.shape[0]), q1, q2)
    if ok: cv2.line(img, q1, q2, col, th, cv2.LINE_AA)

def grid(img, pos, rot=np.eye(3), size=1, n=10, col=(128,128,128)):
    pos = np.array(pos)
    step, half = size/n, size/2
    for i in range(n+1):
        x = -half + i*step
        line3d(img, view(pos + rot.dot((x,0,-half))), view(pos + rot.dot((x,0, half))), col)
        z = -half + i*step
        line3d(img, view(pos + rot.dot((-half,0,z))), view(pos + rot.dot(( half,0,z))), col)

def axes(img, pos, rot=np.eye(3), size=0.075, th=2):
    line3d(img, pos, pos + rot.dot((0,0,size)), (255,0,0), th)
    line3d(img, pos, pos + rot.dot((0,size,0)), (0,255,0), th)
    line3d(img, pos, pos + rot.dot((size,0,0)), (0,0,255), th)

def frustum(img, intr, col=(64,64,64)):
    o = view(np.array([0,0,0],float))
    iw, ih = intr.width, intr.height
    for d in (1,3,5):
        def pt(x,y):
            p = rs.rs2_deproject_pixel_to_point(intr, [x,y], d)
            line3d(img, o, view(np.array(p)), col)
            return p
        tl = pt(0,0); tr = pt(iw,0); br = pt(iw,ih); bl = pt(0,ih)
        line3d(img, view(tl), view(tr), col)
        line3d(img, view(tr), view(br), col)
        line3d(img, view(br), view(bl), col)
        line3d(img, view(bl), view(tl), col)

def pointcloud(img, verts, tex, src, painter=True):
    v = view(verts)
    order = v[:,2].argsort()[::-1] if painter else np.arange(len(v))
    proj = project(v[order]) if painter else project(v)
    if state.scale: proj *= 0.5**state.decimate
    ih, iw = img.shape[:2]
    js, is_ = proj.astype(np.uint32).T
    mask = (is_>=0)&(is_<ih)&(js>=0)&(js<iw)
    cw, ch = src.shape[:2][::-1]
    vu, uu = ((tex[order] if painter else tex)*(cw,ch)+0.5).astype(np.uint32).T
    np.clip(uu, 0, ch-1, out=uu); np.clip(vu, 0, cw-1, out=vu)
    img[is_[mask], js[mask]] = src[uu[mask], vu[mask]]

# -----------------------------------------------------------------------
# 3) Main loop
# -----------------------------------------------------------------------
out = np.zeros((h, w, 3), np.uint8)
last_time = time.time()

while True:
    if not state.paused:
        # a) wait + align
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        df = frames.get_depth_frame()
        cf = frames.get_color_frame()
        df = decimate.process(df)

        # b) images for display & saving
        depth_image    = np.asanyarray(df.get_data())                # 16-bit
        color_image    = np.asanyarray(cf.get_data())                # BGR
        depth_colormap = np.asanyarray(colorizer.colorize(df).get_data())

        # c) build pointcloud (map→then→calculate)
        mapped = cf if state.color else df
        pc.map_to(mapped)
        points = pc.calculate(df)                                   # returns rs.points

        verts     = np.asanyarray(points.get_vertices() ) \
                        .view(np.float32).reshape(-1,3)
        texcoords = np.asanyarray(points.get_texture_coordinates()) \
                        .view(np.float32).reshape(-1,2)
        source = color_image if state.color else depth_colormap

    # d) draw background primitives
    out.fill(0)
    grid(out, (0,0.5,1))
    frustum(out, d_intr)
    axes(out, view(np.array([0,0,0],float)), state.rotation, size=0.1, th=1)

    # e) draw pointcloud
    if not state.scale or out.shape[:2] == (h,w):
        pointcloud(out, verts, texcoords, source)
    else:
        tmp = np.zeros_like(out)
        pointcloud(tmp, verts, texcoords, source)
        tmp = cv2.resize(tmp, out.shape[:2][::-1], cv2.INTER_NEAREST)
        np.putmask(out, tmp>0, tmp)

    if any(state.mouse_btns):
        axes(out, view(state.pivot), state.rotation, th=4)

    # f) show & compute FPS
    now = time.time()
    fps = 1.0 / (now - last_time + 1e-6)
    last_time = now
    cv2.setWindowTitle(state.WIN_NAME, f"RealSense {w}×{h}  FPS: {fps:.1f}")
    cv2.imshow(state.WIN_NAME, out)

    key = cv2.waitKey(1) & 0xFF

    # -------------------------------------------------------------------
    # 4) Keyboard shortcuts
    # -------------------------------------------------------------------
    if key == ord('r'): state.reset()
    if key == ord('p'): state.paused ^= True
    if key == ord('d'):
        state.decimate = (state.decimate + 1) % 3
        decimate.set_option(rs.option.filter_magnitude, 2**state.decimate)
    if key == ord('z'): state.scale ^= True
    if key == ord('c'): state.color ^= True
    if key == ord('s'): cv2.imwrite('out.png', out)

    if key == ord('e'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        ply = f'pc_{ts}.ply'
        img = f'rgb_{ts}.png'
        points.export_to_ply(ply, mapped)
        cv2.imwrite(img, color_image)
        print(f"[INFO] Saved PLY → {ply}")
        print(f"[INFO] Saved RGB → {img}")

    if key == ord('f'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"rgb_{ts}.png",        color_image)
        cv2.imwrite(f"depth_raw_{ts}.png", depth_image)
        cv2.imwrite(f"depth_cm_{ts}.png",  depth_colormap)
        print("[INFO] Snapshots saved.")

    if key in (27, ord('q')) \
       or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
        break

pipeline.stop()
