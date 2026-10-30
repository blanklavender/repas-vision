#!/usr/bin/env python3
"""
overlap_view.py  --  inspect, shot by shot, how the side sweep and the top shot overlap.

Every side shot and the top shot is its own layer in the reference frame (the first side
shot's colour camera), so each one can be switched on and off, scrubbed through the turn,
coloured to tell the shots apart, and checked against its neighbours.

Shot data (--source):
    cleaned       each shot's cleaned cloud as clean_side.py / clean_top.py wrote it
    downsampled   each shot after the downsample_view.py significance steps, kept per shot:
                  incidence filter, multi-view consensus, its own incidence-weighted
                  --voxel-mm grid, and only what survives near the final significant cloud
                  (plane fit + component filter); the fused significant cloud is a layer too

Panel (right):
    Colour        rgb       camera colour
                  shot      one hue per side shot around the turn, top shot white
                  focus     focus shot orange, the other shown shots grey, top magenta
                  overlap   only the focus shot is drawn: green where another SHOWN shot
                            has a point within --overlap-mm, red where only it sees the surface
    Focus         slider / Prev / Next / arrow keys (left, right): the shot in focus;
                  "Scrub" shows only the focus shot and --window shots either side of it
    Shots         a checkbox per shot + All / None / Invert (any combination when not scrubbing)
    Layers        top shot, fused significant cloud (downsampled), cameras, axis, plant frame
    Views         look from the focus shot's camera, from camera pose 1, from the top camera
    Info          focus shot: points, overlap with the previous / next shot (fraction within
                  --overlap-mm and median distance there = how well the two agree)

    python overlap_view.py ../../../captures/femto_bolt/flora_unit/plant_10-16_deg_5_speed_5
    python overlap_view.py <capture> --source downsampled --color shot
    python overlap_view.py <capture> --scrub --focus 18 --window 2 --color overlap
    python overlap_view.py <capture> --shots 0-5,36 --no-top
    python overlap_view.py <capture> --source downsampled --screenshot overlap.png   # render once, no window

Reads the stitch_view.py inputs (registration.json, side/, top/, the pose-transform file);
writes nothing except --screenshot.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

import downsample_view as dv
import stitch_view as sv
import tv_common as tv  # noqa: F401  (puts flora_lib on sys.path)
from flora_lib import geometry as geo
from flora_lib import registration as reg

COLORS = ["rgb", "shot", "focus", "overlap"]
FOCUS_RGB = np.array([1.0, 0.5, 0.05])
OTHER_RGB = np.array([0.55, 0.58, 0.6])
TOP_RGB = sv.TOP_TINT
OK_RGB, ALONE_RGB = np.array([0.15, 0.85, 0.3]), np.array([0.95, 0.15, 0.15])


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def parse_shots(text, n):
    """'0-5,36' -> sorted indices (all when empty)."""
    if not text:
        return list(range(n))
    out = set()
    for part in text.split(","):
        a, _, b = part.strip().partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(k for k in out if 0 <= k < n)


def build_layers(S, args):
    """-> list of layers (dict name, label, P, C, T camera pose in ref), fused (P, C) or None."""
    args.downsample, args.method = args.source == "downsampled", "significant"
    D = dv.load_shots(S, args)
    P, C, shot, n_side = D["P"], D["C"], D["shot"], D["n_side"]
    labels = [f"{f['stem']}  {f['true_deg']:5.1f} deg" for f in S["frames"]] + ["top shot"]
    poses = list(S["poses"]) + [S["T_top"]]
    fused = None
    if args.source == "cleaned":
        sel = [shot == k for k in range(n_side + 1)]
        layers = [dict(P=P[m], C=C[m]) for m in sel]
    else:
        side, top, _ = dv.significant(D, args, print)
        fused = (np.vstack([side[0], top[0]]), np.clip(np.vstack([side[1], top[1]]), 0, 1))
        final = sv.to_pcd(*fused)
        is_top = shot == n_side
        keep = D["cos"] >= np.cos(np.radians(args.max_incidence_deg))
        sup = np.zeros(len(P), np.int64)
        sup[keep] = dv.view_support(P[keep], shot[keep], n_side + 1, args.support_mm / 1000.0)
        keep &= is_top | (sup >= args.min_views)
        layers, n_in, n_out = [], 0, 0
        for k in range(n_side + 1):
            m = keep & (shot == k)
            n_in += int((shot == k).sum())
            if not m.any():
                layers.append(dict(P=np.zeros((0, 3)), C=np.zeros((0, 3))))
                continue
            Pk, (Ck,) = dv.fuse(P[m], [C[m]], args.voxel_mm / 1000.0, D["cos"][m])
            near = np.asarray(sv.to_pcd(Pk, Ck).compute_point_cloud_distance(final)) * 1000 < args.near_mm
            layers.append(dict(P=Pk[near], C=np.clip(Ck[near], 0, 1)))
            n_out += int(near.sum())
        print(f"[SHOT] per-shot downsampled: {n_in:,} -> {n_out:,} points over {n_side + 1} layers "
              f"({n_out / max(n_in, 1):.1%}); the fused significant cloud has {len(fused[0]):,}")
    for k, L in enumerate(layers):
        L.update(name=f"shot {k}", label=labels[k], T=poses[k], top=k == n_side)
    return layers, fused


def hue(k, n):
    hsv = np.uint8([[[int(k * 180 / max(n, 1)) % 180, 220, 255]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0] / 255.0


def overlap_stats(A, B, tol_m):
    """Fraction of A's points with a B point within tol, and the median distance of those."""
    if not len(A) or not len(B):
        return 0.0, float("nan")
    d = np.asarray(sv.to_pcd(A, np.zeros_like(A)).compute_point_cloud_distance(sv.to_pcd(B, np.zeros_like(B))))
    close = d < tol_m
    return float(close.mean()), float(np.median(d[close]) * 1000) if close.any() else float("nan")


# ---------------------------------------------------------------------------
# viewer
# ---------------------------------------------------------------------------
class OverlapViewer:
    PANEL_EM = 22

    def __init__(self, layers, fused, F, cam, args):
        from open3d.visualization import gui, rendering
        self.gui, self.rendering = gui, rendering
        self.layers, self.fused, self.F, self.cam, self.args = layers, fused, F, cam, args
        self.n = len(layers)
        self.n_side = self.n - 1
        self.focus = min(max(args.focus, 0), self.n_side - 1)
        self.window = args.window
        self.scrub = args.scrub
        self.color = args.color
        start = set(parse_shots(args.shots, self.n_side))
        self.visible = np.array([k in start for k in range(self.n_side)] + [not args.no_top])
        self.stats_cache = {}
        self.agree = 0.0
        self.overlap_tol = args.overlap_mm / 1000.0
        self.P_all = np.vstack([L["P"] for L in layers if len(L["P"])])

        app = gui.Application.instance
        app.initialize()
        self.app = app
        self.win = app.create_window(f"Flora shot overlap ({args.source})", 1600, 980)
        em = self.win.theme.font_size

        self.view = gui.SceneWidget()
        self.view.scene = rendering.Open3DScene(self.win.renderer)
        self.view.scene.set_background([0.12, 0.12, 0.12, 1.0])
        self.view.scene.show_axes(False)
        self.view.set_on_key(self.on_key)
        self.pts = rendering.MaterialRecord()
        self.pts.shader, self.pts.point_size = "defaultUnlit", args.point_size * self.win.scaling
        self.line = rendering.MaterialRecord()
        self.line.shader, self.line.line_width = "unlitLine", 1.5 * self.win.scaling
        lit = rendering.MaterialRecord()
        lit.shader = "defaultLit"

        for k, L in enumerate(layers):                 # cameras first: they never change
            fr = o3d.geometry.LineSet.create_camera_visualization(
                cam["size"][0], cam["size"][1], cam["K"], np.linalg.inv(L["T"]), scale=0.035)
            fr.paint_uniform_color(TOP_RGB if L["top"] else hue(k, self.n_side))
            self.view.scene.add_geometry(f"cam {k}", fr, self.line)
        a, O = F["axis"], F["origin"]
        h0, h1 = np.array(F["height_range_mm"]) / 1000.0
        axis = o3d.geometry.LineSet(o3d.utility.Vector3dVector([O + a * (h0 - 0.02), O + a * (h1 + 0.03)]),
                                    o3d.utility.Vector2iVector([[0, 1]]))
        axis.paint_uniform_color([0.2, 0.9, 0.9])
        self.view.scene.add_geometry("axis", axis, self.line)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        frame.transform(geo.make_T(F["R"], O))
        self.view.scene.add_geometry("plant frame", frame, lit)
        if fused is not None:
            self.view.scene.add_geometry("fused", sv.to_pcd(*fused), self.pts)
        self.layer_flags = {"cameras": args.cameras, "axis": True, "plant frame": False, "fused": False}

        # ---- panel
        panel = gui.ScrollableVert(0.4 * em, gui.Margins(em, em, em, em))
        self.panel = panel
        panel.add_child(gui.Label(f"{self.n_side} side shots + top shot | source: {args.source}"))

        panel.add_child(gui.Label("Colour"))
        self.combo = gui.Combobox()
        for c in COLORS:
            self.combo.add_item(c)
        self.combo.selected_index = COLORS.index(self.color)
        self.combo.set_on_selection_changed(self.on_color)
        panel.add_child(self.combo)

        panel.add_child(gui.Label("Focus shot  (arrow keys in the 3D view)"))
        self.slider = gui.Slider(gui.Slider.INT)
        self.slider.set_limits(0, self.n_side - 1)
        self.slider.int_value = self.focus
        self.slider.set_on_value_changed(lambda v: self.set_focus(int(v)))
        panel.add_child(self.slider)
        row = gui.Horiz(0.4 * em)
        for text, cb in (("< Prev", lambda: self.set_focus(self.focus - 1)),
                         ("Next >", lambda: self.set_focus(self.focus + 1))):
            b = gui.Button(text)
            b.set_on_clicked(cb)
            row.add_child(b)
        panel.add_child(row)
        self.scrub_cb = gui.Checkbox("Scrub: focus +- window only")
        self.scrub_cb.checked = self.scrub
        self.scrub_cb.set_on_checked(self.on_scrub)
        panel.add_child(self.scrub_cb)
        panel.add_child(gui.Label("Window (shots either side)"))
        self.win_slider = gui.Slider(gui.Slider.INT)
        self.win_slider.set_limits(0, max(1, self.n_side // 2))
        self.win_slider.int_value = self.window
        self.win_slider.set_on_value_changed(self.on_window)
        panel.add_child(self.win_slider)

        panel.add_child(gui.Label("Point size"))
        ps = gui.Slider(gui.Slider.DOUBLE)
        ps.set_limits(1.0, 8.0)
        ps.double_value = args.point_size
        ps.set_on_value_changed(self.on_point_size)
        panel.add_child(ps)

        layers_box = gui.CollapsableVert("Layers", 0.25 * em, gui.Margins(0.5 * em, 0, 0, 0))
        self.top_cb = gui.Checkbox("Top shot")
        self.top_cb.checked = bool(self.visible[-1])
        self.top_cb.set_on_checked(lambda on: self.set_visible(self.n_side, on))
        layers_box.add_child(self.top_cb)
        for text, key in (("Fused significant cloud", "fused"), ("Shot cameras", "cameras"),
                          ("Turntable axis", "axis"), ("Plant frame", "plant frame")):
            if key == "fused" and fused is None:
                continue
            c = gui.Checkbox(text)
            c.checked = self.layer_flags[key]
            c.set_on_checked(lambda on, k=key: self.on_layer(k, on))
            layers_box.add_child(c)
        panel.add_child(layers_box)

        views = gui.Horiz(0.4 * em)
        for text, cb in (("Focus cam", self.view_focus), ("Pose 1", self.view_side), ("Top cam", self.view_top)):
            b = gui.Button(text)
            b.set_on_clicked(cb)
            views.add_child(b)
        panel.add_child(gui.Label("Look from"))
        panel.add_child(views)

        self.info = gui.Label("")
        panel.add_child(self.info)

        shots_box = gui.CollapsableVert("Shots", 0.25 * em, gui.Margins(0.5 * em, 0, 0, 0))
        row = gui.Horiz(0.4 * em)
        for text, cb in (("All", lambda: self.set_all(True)), ("None", lambda: self.set_all(False)),
                         ("Invert", self.invert)):
            b = gui.Button(text)
            b.set_on_clicked(cb)
            row.add_child(b)
        shots_box.add_child(row)
        grid = gui.VGrid(3, 0.25 * em)
        self.shot_cbs = []
        for k in range(self.n_side):
            c = gui.Checkbox(f"{k:02d} {self.layers[k]['label'].split()[-2]}")
            c.checked = bool(self.visible[k])
            c.set_on_checked(lambda on, k=k: self.set_visible(k, on))
            grid.add_child(c)
            self.shot_cbs.append(c)
        shots_box.add_child(grid)
        panel.add_child(shots_box)

        self.win.add_child(self.view)
        self.win.add_child(panel)
        self.win.set_on_layout(self.on_layout)

        self.recolor_all()
        self.sync()
        self.view_side()

    # ---- colours -----------------------------------------------------------
    def colours(self, k):
        L = self.layers[k]
        n = len(L["P"])
        if self.color == "rgb":
            return L["C"]
        if self.color == "shot":
            return np.tile(dv.TOP_WHITE if L["top"] else hue(k, self.n_side), (n, 1))
        if k != self.focus:
            return np.tile(TOP_RGB if L["top"] else OTHER_RGB, (n, 1))
        if self.color == "focus":
            return np.tile(FOCUS_RGB, (n, 1))
        others = [self.layers[j]["P"] for j in self.shown() if j != k and len(self.layers[j]["P"])]
        if not others or not n:
            self.agree = 0.0
            return np.tile(ALONE_RGB, (n, 1))
        d = np.asarray(sv.to_pcd(L["P"], L["C"]).compute_point_cloud_distance(
            sv.to_pcd(np.vstack(others), np.zeros((sum(len(o) for o in others), 3)))))
        self.agree = float((d < self.overlap_tol).mean())
        return np.where((d < self.overlap_tol)[:, None], OK_RGB, ALONE_RGB)

    def recolor(self, k):
        name = self.layers[k]["name"]
        if self.view.scene.has_geometry(name):
            self.view.scene.remove_geometry(name)
        if len(self.layers[k]["P"]):
            self.view.scene.add_geometry(name, sv.to_pcd(self.layers[k]["P"], np.clip(self.colours(k), 0, 1)),
                                         self.pts)

    def recolor_all(self):
        for k in range(self.n):
            self.recolor(k)

    # ---- visibility --------------------------------------------------------
    def shown(self):
        return [k for k in range(self.n) if self.visible[k]]

    def sync(self):
        """Scene + checkboxes from self.visible / layer flags, then the info text."""
        if self.scrub:
            idx = [(self.focus + d) % self.n_side for d in range(-self.window, self.window + 1)]
            self.visible[:self.n_side] = False
            self.visible[idx] = True
        sc = self.view.scene
        for k in range(self.n):
            on = bool(self.visible[k])
            drawn = on and (self.color != "overlap" or k == self.focus)   # overlap: compare, draw focus only
            if sc.has_geometry(self.layers[k]["name"]):
                sc.show_geometry(self.layers[k]["name"], drawn)
            sc.show_geometry(f"cam {k}", on and self.layer_flags["cameras"])
            if k < self.n_side:
                self.shot_cbs[k].checked = on
        self.top_cb.checked = bool(self.visible[-1])
        for key in ("axis", "plant frame", "fused"):
            if sc.has_geometry(key):
                sc.show_geometry(key, self.layer_flags[key])
        self.update_info()

    def set_visible(self, k, on):
        if self.scrub and k < self.n_side:
            self.scrub = self.scrub_cb.checked = False
        self.visible[k] = on
        self.after_visibility_change()

    def set_all(self, on):
        self.scrub = self.scrub_cb.checked = False
        self.visible[:self.n_side] = on
        self.after_visibility_change()

    def invert(self):
        self.scrub = self.scrub_cb.checked = False
        self.visible[:self.n_side] = ~self.visible[:self.n_side]
        self.after_visibility_change()

    def after_visibility_change(self):
        if self.color == "overlap":
            self.recolor(self.focus)
        self.sync()

    def set_focus(self, k):
        old, self.focus = self.focus, k % self.n_side
        self.slider.int_value = self.focus
        if self.color in ("focus", "overlap"):
            self.recolor(old)
        self.sync()                                   # scrub moves the window before the overlap colours
        if self.color in ("focus", "overlap"):
            self.recolor(self.focus)
            self.sync()

    # ---- callbacks ---------------------------------------------------------
    def on_color(self, text, idx):
        self.color = COLORS[idx]
        self.recolor_all()
        self.sync()

    def on_scrub(self, on):
        self.scrub = on
        if not on:
            self.visible[:self.n_side] = True
        self.after_visibility_change()

    def on_window(self, v):
        self.window = int(v)
        if self.scrub:
            self.after_visibility_change()

    def on_layer(self, key, on):
        self.layer_flags[key] = on
        self.sync()

    def on_point_size(self, v):
        self.pts.point_size = float(v) * self.win.scaling
        for name in [L["name"] for L in self.layers] + ["fused"]:
            if self.view.scene.has_geometry(name):
                self.view.scene.modify_geometry_material(name, self.pts)

    def on_key(self, ev):
        gui = self.gui
        if ev.type == gui.KeyEvent.Type.DOWN and ev.key in (gui.KeyName.LEFT, gui.KeyName.RIGHT):
            self.set_focus(self.focus + (1 if ev.key == gui.KeyName.RIGHT else -1))
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def on_layout(self, ctx):
        r = self.win.content_rect
        pw = self.PANEL_EM * ctx.theme.font_size
        self.view.frame = self.gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = self.gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    # ---- info + views ------------------------------------------------------
    def neighbour_stats(self, k):
        if k not in self.stats_cache:
            P = self.layers[k]["P"]
            prev = overlap_stats(P, self.layers[(k - 1) % self.n_side]["P"], self.overlap_tol)
            nxt = overlap_stats(P, self.layers[(k + 1) % self.n_side]["P"], self.overlap_tol)
            top = overlap_stats(P, self.layers[-1]["P"], self.overlap_tol)
            self.stats_cache[k] = (prev, nxt, top)
        return self.stats_cache[k]

    def update_info(self):
        k = self.focus
        L = self.layers[k]
        (fp, dp), (fn, dn), (ft, dt) = self.neighbour_stats(k)
        shown = self.shown()
        lines = [f"Focus  {L['label']}", f"  {len(L['P']):,} points",
                 f"  within {self.args.overlap_mm:g} mm of",
                 f"    prev shot  {fp:5.1%}  (median {dp:.1f} mm)",
                 f"    next shot  {fn:5.1%}  (median {dn:.1f} mm)",
                 f"    top shot   {ft:5.1%}  (median {dt:.1f} mm)",
                 f"Shown  {len(shown)} layer(s), {sum(len(self.layers[j]['P']) for j in shown):,} points"]
        if self.color == "overlap":
            lines.append(f"Overlap mode: only the focus is drawn\n  green {self.agree:.0%} = a shown shot agrees\n"
                         f"  red = only this shot sees it")
        self.info.text = "\n".join(lines)
        self.win.set_needs_layout()

    def look(self, eye, up):
        bb = self.view.scene.bounding_box
        self.view.setup_camera(35.0, bb, bb.get_center())
        self.view.look_at(self.P_all.mean(0).astype(np.float32), np.asarray(eye, np.float32),
                          np.asarray(up, np.float32))

    def view_pose(self, T):
        self.look(T[:3, 3], -T[:3, 1])

    def view_focus(self):
        self.view_pose(self.layers[self.focus]["T"])

    def view_side(self):
        self.look(np.zeros(3), [0.0, -1.0, 0.0])

    def view_top(self):
        self.view_pose(self.layers[-1]["T"])

    # ---- run ---------------------------------------------------------------
    def run(self, screenshot=None):
        if screenshot:
            def grab():
                img = self.app.render_to_image(self.view.scene, 1200, 900)
                o3d.io.write_image(str(screenshot), img)
                print(f"[SHOT] {screenshot}")
                self.win.close()
            self.app.post_to_main_thread(self.win, grab)
        self.app.run()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Toggle shots on / off and see how they overlap",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sv.add_scene_args(ap)
    ap.add_argument("--source", choices=["cleaned", "downsampled"], default="cleaned",
                    help="per-shot clouds as cleaned, or after the significance downsampling")
    ap.add_argument("--color", choices=COLORS, default="shot", help="initial colour mode")
    ap.add_argument("--shots", default="", help="side shots shown at start, e.g. 0-5,36 (default all)")
    ap.add_argument("--no-top", action="store_true", help="top shot hidden at start")
    ap.add_argument("--focus", type=int, default=0, help="initial focus shot")
    ap.add_argument("--window", type=int, default=1, help="scrub: shots shown either side of the focus")
    ap.add_argument("--scrub", action="store_true", help="start in scrub mode (focus +- window only)")
    ap.add_argument("--cameras", action="store_true", help="show the shot cameras at start")
    ap.add_argument("--overlap-mm", type=float, default=3.0, help="two shots agree where points are this close")
    ap.add_argument("--near-mm", type=float, default=5.0,
                    help="downsampled: keep a shot's points this close to the final significant cloud")
    ap.add_argument("--view-from", choices=["side", "top", "focus"], default="side", help="initial viewpoint")
    ap.add_argument("--screenshot", default=None, help="render once to this image and exit (no window)")
    dv.add_significance_args(ap)
    args = ap.parse_args()

    S = sv.load_scene(args)
    layers, fused = build_layers(S, args)
    P_side = np.vstack([L["P"] for L in layers[:-1] if len(L["P"])])
    F = reg.plant_frame(S["A"], P_side, args.base_percentile)
    print(f"[VIEW] {len(layers) - 1} side shots + top shot, {sum(len(L['P']) for L in layers):,} points "
          f"({args.source})" + ("" if args.screenshot else " -- close the window when done"))
    v = OverlapViewer(layers, fused, F, S["cam"], args)
    {"side": v.view_side, "top": v.view_top, "focus": v.view_focus}[args.view_from]()
    v.run(Path(args.screenshot) if args.screenshot else None)


if __name__ == "__main__":
    sys.exit(main())
