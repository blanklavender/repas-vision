#!/usr/bin/env python3
"""
surface_pick_viewer.py  --  pick points on the RGB-only 2DGS surface; get the 2DGS point + normal,
the ToF point, the gap between them, the pixel in the best shot and the best turntable command.
All coordinates in the colour-camera frame. Optionally (--depth-run) also the same point on a second
2DGS mesh, e.g. one trained with ToF depth.

    conda activate repas
    python surface_pick_viewer.py                      # defaults: plant_10-16 clean55 runs
    python surface_pick_viewer.py --run <2dgs run> --raw <capture> --tof-cache <dir> --clean-dir <dir>
    python surface_pick_viewer.py --depth-run <2dgs+ToF-depth run>    # add the comparison mesh

Controls: Ctrl + left click picks (snaps to the visible mesh vertex under the cursor);
drag rotates, wheel zooms, right-drag pans.
Markers: white = RGB-only 2DGS point, blue = ToF-depth 2DGS point, red = ToF point,
yellow line = 2DGS normal (15 mm), grey line = the best shot's camera ray through the point.

The numbers are explained in surface_pick.py. Every pick is printed in full in the terminal and
saved to <run>/surface_picks/: picks.json (full records), picks.csv (one row per pick) and
pick_<k>_<shot>.png (the best shot with a crosshair at the pixel).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import surface_pick as sp
from tag_pose_pick import screen_pick

WHITE, BLUE, RED, YELLOW, GREY = [1, 1, 1], [0.2, 0.45, 1.0], [1.0, 0.15, 0.15], [1.0, 0.85, 0.1], [0.7, 0.7, 0.7]


def sphere(X, r_mm, rgb):
    s = o3d.geometry.TriangleMesh.create_sphere(r_mm / 1000.0, resolution=12)
    s.translate(X)
    s.paint_uniform_color(rgb)
    s.compute_vertex_normals()
    return s


def segment(a, b, rgb):
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector([a, b]), o3d.utility.Vector2iVector([[0, 1]]))
    ls.paint_uniform_color(rgb)
    return ls


def panel_text(k, r):
    """Compact summary for the side panel (the terminal gets the full report)."""
    if "error" in r:
        return f"P{k}: {r['error']}"
    rc, b, t, g = r["ref_camera"], r["best_shot"], r["turntable"], r["gaps"]
    f = sp.fmt
    lines = [f"P{k}   vertex {r['vertex']}   leaf {('L%d' % r['leaf']) if r['leaf'] > 0 else 'none'}",
             "Camera frame, turntable 0 deg (mm):",
             f" 2DGS      {f(rc['point_mm'])}",
             f" normal    {f(rc['normal'], 3)}",
             f" ToF       {f(r['tof']['point_mm']) if r['tof'] else 'none'}",
             "Gaps along the camera ray (mm, + = farther):"]
    if r.get("has_depth_run"):
        lines.insert(-1, f" 2DGS+ToF  {f(r['depth2dgs']['point_mm']) if r['depth2dgs'] else 'none'}")
    for key, name in (("tof_minus_2dgs_mm", "ToF - 2DGS"), ("depth2dgs_minus_2dgs_mm", "2DGS+ToF - 2DGS"),
                      ("tof_minus_depth2dgs_mm", "ToF - 2DGS+ToF")):
        if isinstance(g.get(key), float):
            lines.append(f" {name:16s} {g[key]:+7.1f}")
    if g.get("suspect"):
        lines.append(" !! a gap > 35 mm: likely another surface")
    lines += [f"Best shot {b['stem']}  ({b['incidence_deg']:.0f} deg off head-on)",
              f" pixel {f(b['pixel_undistorted'])} undist.",
              f"       {f(b['pixel_raw'])} raw"]
    if "error" in t:
        lines.append(f"Turntable: {t['error']}")
    else:
        lines += [f"Turntable: command {t['command_deg']} deg  (true {t['true_deg']:.1f})",
                  f" {t['incidence_deg']:.0f} deg off head-on; <=30 deg at {t['commands_within_30deg'] or '-'}",
                  f" point  {f(t['point_cam_mm'])}",
                  f" normal {f(t['normal_cam'], 3)}"]
    return "\n".join(lines)


class Viewer:
    PANEL_EM = 27

    def __init__(self, q, args):
        self.q, self.args = q, args
        self.out = Path(args.run) / "surface_picks"
        self.out.mkdir(exist_ok=True)
        self.picks = []

        app = gui.Application.instance
        app.initialize()
        self.win = app.create_window("2DGS surface picker", 1600, 980)
        em = self.win.theme.font_size
        self.view = gui.SceneWidget()
        self.view.scene = rendering.Open3DScene(self.win.renderer)
        self.view.scene.set_background([0.12, 0.12, 0.12, 1.0])
        self.view.scene.show_axes(False)

        self.unlit = rendering.MaterialRecord()
        self.unlit.shader = "defaultUnlit"
        self.lit = rendering.MaterialRecord()
        self.lit.shader = "defaultLit"
        self.line = rendering.MaterialRecord()
        self.line.shader, self.line.line_width = "unlitLine", 3.0 * self.win.scaling

        self.view.scene.add_geometry("rgb", q.mesh, self.unlit)
        leaves = Path(args.run) / "leaf_track/leaves_mesh.ply"
        self.has_leaves = leaves.is_file()
        if self.has_leaves:
            self.view.scene.add_geometry("leaves", o3d.io.read_triangle_mesh(str(leaves)), self.unlit)
            self.view.scene.show_geometry("leaves", False)
        self.has_depth = q.depth_run is not None
        if self.has_depth:
            dm = o3d.io.read_triangle_mesh(str(q.depth_run / "mesh/tsdf_mesh_metres.ply"))
            self.view.scene.add_geometry("depth mesh", dm, self.unlit)
            self.view.scene.show_geometry("depth mesh", False)

        F = q.F
        h = q.V @ F["axis"] - F["origin"] @ F["axis"]
        self.view.scene.add_geometry("axis", segment(F["origin"] + F["axis"] * (h.min() - 0.02),
                                                     F["origin"] + F["axis"] * (h.max() + 0.03), [0.2, 0.9, 0.9]),
                                     self.line)
        self.view.set_on_mouse(self.on_mouse)

        self.panel = gui.Vert(0.4 * em, gui.Margins(em, em, em, em))
        self.panel.add_child(gui.Label("Ctrl + left click: pick    Drag: rotate\nWheel: zoom    Right-drag: pan\n"
                                       + ("white 2DGS  blue 2DGS+ToF-depth  red ToF\n" if self.has_depth
                                          else "white 2DGS  red ToF\n") +
                                       "yellow normal  grey best-shot ray"))
        row = gui.Horiz(0.4 * em)
        for text, cb in (("Undo", self.undo), ("Clear", self.clear), ("Reset view", self.reset_view)):
            b = gui.Button(text)
            b.set_on_clicked(cb)
            row.add_child(b)
        self.panel.add_child(row)
        if self.has_leaves:
            c = gui.Checkbox("Leaf colours (leaf_track)")
            c.set_on_checked(self.toggle_leaves)
            self.panel.add_child(c)
        if self.has_depth:
            c = gui.Checkbox("Show 2DGS+ToF-depth mesh instead")
            c.set_on_checked(self.toggle_depth_mesh)
            self.panel.add_child(c)
        c = gui.Checkbox("Turntable axis")
        c.checked = True
        c.set_on_checked(lambda on: self.view.scene.show_geometry("axis", on))
        self.panel.add_child(c)
        self.info = gui.Label("No point picked yet")
        self.panel.add_child(self.info)
        self.image = gui.ImageWidget(o3d.geometry.Image(np.zeros((320, 320, 3), np.uint8)))
        self.panel.add_child(self.image)

        self.win.add_child(self.view)
        self.win.add_child(self.panel)
        self.win.set_on_layout(self.on_layout)
        self.reset_view()

    # --- layout / view ----------------------------------------------------------
    def on_layout(self, ctx):
        r = self.win.content_rect
        pw = self.PANEL_EM * ctx.theme.font_size
        self.view.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    def reset_view(self):
        """Look at the plant from the reference camera (turntable 0 deg)."""
        bb = self.view.scene.bounding_box
        self.view.setup_camera(35.0, bb, bb.get_center())
        self.view.look_at(self.q.V.mean(0).astype(np.float32), np.zeros(3, np.float32),
                          np.array([0.0, -1.0, 0.0], np.float32))

    def toggle_leaves(self, on):
        self.view.scene.show_geometry("leaves", on)
        if not (self.has_depth and self.view.scene.geometry_is_visible("depth mesh")):
            self.view.scene.show_geometry("rgb", not on)

    def toggle_depth_mesh(self, on):
        self.view.scene.show_geometry("depth mesh", on)
        self.view.scene.show_geometry("rgb", not on and not (self.has_leaves and
                                                              self.view.scene.geometry_is_visible("leaves")))
        if on and self.has_leaves:
            self.view.scene.show_geometry("leaves", False)

    # --- picking ---------------------------------------------------------------------
    def on_mouse(self, ev):
        if not (ev.type == gui.MouseEvent.Type.BUTTON_DOWN and ev.is_modifier_down(gui.KeyModifier.CTRL)
                and ev.is_button_down(gui.MouseButton.LEFT)):
            return gui.Widget.EventCallbackResult.IGNORED
        fr, cam = self.view.frame, self.view.scene.camera
        hit = screen_pick(self.q.V, np.asarray(cam.get_view_matrix()), np.asarray(cam.get_projection_matrix()),
                          (fr.width, fr.height), (ev.x - fr.x, ev.y - fr.y), self.args.pick_radius_px,
                          self.args.pick_depth_mm / 1000.0)
        if hit is None:
            self.set_info("No surface under the cursor -- click on the plant")
        else:
            self.add_pick(hit[0])
        return gui.Widget.EventCallbackResult.HANDLED

    def add_pick(self, i):
        k = len(self.picks) + 1
        r = self.q.query(i)
        print(f"\n[PICK {k}]")
        print("\n".join(sp.report_lines(r)), flush=True)
        names = []
        if "error" not in r:
            X = r["ref_camera"]["point_mm"] / 1000.0
            n = r["ref_camera"]["normal"]
            C = self.q.views[[v["stem"] for v in self.q.views].index(r["best_shot"]["stem"])]["c2w"][:3, 3]
            d = (C - X) / np.linalg.norm(C - X)
            geoms = [("2dgs", sphere(X, 1.4, WHITE), self.lit),
                     ("normal", segment(X, X + n * 0.015, YELLOW), self.line),
                     ("ray", segment(X - d * 0.03, X + d * 0.06, GREY), self.line)]
            if r["tof"]:
                geoms.append(("tof", sphere(r["tof"]["point_mm"] / 1000.0, 1.4, RED), self.lit))
            if r["depth2dgs"]:
                geoms.append(("depth", sphere(r["depth2dgs"]["point_mm"] / 1000.0, 1.2, BLUE), self.lit))
            for g, geom, mat in geoms:
                name = f"pick {k} {g}"
                self.view.scene.add_geometry(name, geom, mat)
                names.append(name)
            label = self.view.add_3d_label(X + n * 0.018, f"P{k}")
            crop = self.q.shot_image(r["best_shot"]["stem"], r["best_shot"]["pixel_undistorted"])
            self.image.update_image(o3d.geometry.Image(np.ascontiguousarray(crop)))
            import cv2
            cv2.imwrite(str(self.out / f"pick_{k}_{r['best_shot']['stem']}.png"), crop[..., ::-1])
        else:
            label = None
        self.picks.append(dict(rec=r, names=names, label=label))
        self.set_info(panel_text(k, r))
        self.write_outputs()

    def undo(self):
        if not self.picks:
            return
        p = self.picks.pop()
        for n in p["names"]:
            self.view.scene.remove_geometry(n)
        if p["label"] is not None:
            self.view.remove_3d_label(p["label"])
        self.set_info(f"Last pick removed, {len(self.picks)} left")
        self.write_outputs()

    def clear(self):
        while self.picks:
            self.undo()
        self.set_info("No point picked yet")

    def set_info(self, text):
        self.info.text = text
        self.win.set_needs_layout()

    def write_outputs(self):
        recs = [sp.to_jsonable(p["rec"]) for p in self.picks]
        (self.out / "picks.json").write_text(json.dumps(dict(
            frame="colour camera with the turntable at the first shot (000_ang000, 0 deg); x right, y down, "
                  "z forward; mm. turntable.* is the camera frame at that turntable command.",
            run=str(self.args.run), depth_run=self.args.depth_run, picks=recs), indent=2))
        cols = ["pick", "vertex", "leaf", "x_2dgs", "y_2dgs", "z_2dgs", "nx", "ny", "nz",
                "x_tof", "y_tof", "z_tof", "x_2dgs_tofdepth", "y_2dgs_tofdepth", "z_2dgs_tofdepth",
                "gap_tof_minus_2dgs", "gap_2dgstofdepth_minus_2dgs", "gap_tof_minus_2dgstofdepth",
                "best_shot", "u_undist", "v_undist", "u_raw", "v_raw",
                "turntable_command_deg", "turntable_true_deg", "incidence_deg",
                "x_at_turn", "y_at_turn", "z_at_turn", "nx_at_turn", "ny_at_turn", "nz_at_turn"]
        with open(self.out / "picks.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for k, r in enumerate(recs, 1):
                if "error" in r:
                    w.writerow([k, r["vertex"]] + [""] * (len(cols) - 2))
                    continue
                rc, b, t, g = r["ref_camera"], r["best_shot"], r["turntable"], r["gaps"]
                tof = r["tof"]["point_mm"] if r["tof"] else ["", "", ""]
                dep = r["depth2dgs"]["point_mm"] if r["depth2dgs"] else ["", "", ""]
                turn = ([t["command_deg"], t["true_deg"], t["incidence_deg"], *t["point_cam_mm"], *t["normal_cam"]]
                        if "error" not in t else [""] * 9)
                w.writerow([k, r["vertex"], r["leaf"], *rc["point_mm"], *rc["normal"], *tof, *dep,
                            g.get("tof_minus_2dgs_mm", ""), g.get("depth2dgs_minus_2dgs_mm", ""),
                            g.get("tof_minus_depth2dgs_mm", ""), b["stem"], *b["pixel_undistorted"],
                            *b["pixel_raw"], *turn])

    def run(self):
        gui.Application.instance.run()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp.add_args(ap)
    ap.add_argument("--pick-radius-px", type=float, default=12.0, help="snap radius around the click")
    ap.add_argument("--pick-depth-mm", type=float, default=4.0, help="front-surface depth band for the snap")
    args = ap.parse_args()
    print("[LOAD] meshes, cameras, ToF reference ...", flush=True)
    q = sp.from_args(args)
    st = q.selftest()
    print(f"[TURN] turntable model vs the 70 shot cameras: centre {st['centre_mm_median']:.1f} mm median "
          f"({st['centre_mm_max']:.1f} max), rotation {st['rot_deg_median']:.2f} deg median")
    print("[TURN] commands assume: home(), then a single goto(command) (whole degrees, 0.044 deg motor steps)")
    print(f"[OUT ] picks are saved to {Path(args.run) / 'surface_picks'}")
    Viewer(q, args).run()


if __name__ == "__main__":
    main()
