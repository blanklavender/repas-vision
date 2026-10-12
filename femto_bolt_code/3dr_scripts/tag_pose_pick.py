#!/usr/bin/env python3
"""
tag_pose_pick.py  --  stitch the cleaned shots of a turntable sweep from AprilTag poses,
then pick points on the fused plant in a 3D viewer and get their camera coordinates.

Input is the clean dataset of sam3_data_cleaning.py: one cleaned cloud per shot (colour-
camera frame). The tags are detected in the ORIGINAL colour images.

Registration (flora_lib.tags / flora_lib.registration, = tag_pose_register.py "pose"):
    per adjacent pair, per shared tag   T = P_(i+1) @ P_i^-1      (colour-only PnP tag poses)
    pair transform                      chordal L2 mean of the per-tag rotations + mean
                                        tag-centre translation
    the pairs are chained into the first shot's frame; on a full turn the last->first edge
    closes the loop and its error is spread evenly over the shots. All shots are then
    voxel-fused (--voxel-mm).

Plant frame (from the registered poses):
    z       turntable axis, pointing up
    origin  on that axis, at the pot base (--base-percentile of the cloud's height along z)
    x       horizontal, from the axis toward the reference camera;  y = z cross x

Picking: Ctrl + left click on the cloud (drag rotates, wheel zooms). The click snaps to the
nearest fused point, gets a marker and is printed:
    camera  the reference (first) shot's COLOUR camera: x right, y down, z forward, mm,
            plus the pixel in that shot's colour image
    plant   plant-frame x y z, radius / azimuth / height above the pot base
    turn    the turntable angle at which the point faces the camera (+ coordinates there)
The picked camera coordinates go to --points-file, which is rewritten (emptied) when the
viewer opens and after every pick / undo, for the turntable at --report-deg (default: the
first shot's angle, i.e. where sweep_capture.py --home-after leaves the plant).

    python tag_pose_pick.py ../../captures/femto_bolt/flora_unit/yellow_pot_10-06_deg_10_speed_5
    python tag_pose_pick.py <capture> --points-file ../../picked_points.txt --at-deg 90,180
    python tag_pose_pick.py <capture> --restitch --max-obliquity 45

The stitch is cached in <capture>/<--out> and reused while its settings and the cleaning
run match (--restitch forces a new one).

I/O contract (layout: flora_pipeline/flora_lib/dataset.py)
------------
in   <capture>/manifest.csv, camera_params.json, color_<stem>.png      sweep_capture.py
     <capture>/cleaning.json, pointcloud/cloud_<stem>.ply                sam3_data_cleaning.py
out  <capture>/<--out, default tag_pick>/
        poses.json         T_shot_to_ref per shot + edges, loop error, settings
        fused.ply          all cleaned shots in the reference shot's colour-camera frame
        plant_frame.json   turntable axis, plant origin and axes (reference camera frame)
        picks.csv          every pick of the last session, camera + plant coordinates
     --points-file         default <capture>/<--out>/picked_points.txt: 'P<k> x y z' in mm
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "flora_pipeline"))
from flora_lib import dataset as ds  # noqa: E402
from flora_lib import geometry as geo  # noqa: E402
from flora_lib import registration as reg  # noqa: E402
from flora_lib import tags as tg  # noqa: E402

SETTING_KEYS = ("max_obliquity", "tag_size_mm", "loop", "voxel_mm", "tag_family")


# ---------------------------------------------------------------------------
# stitching
# ---------------------------------------------------------------------------
def stitch(capture, frames, cam, args):
    """-> poses (T_shot_to_ref), fused points + colours, registration info."""
    det = tg.make_detector(args.tag_family)
    tags, P, C = [], [], []
    for f in frames:
        gray = cv2.imread(str(capture / ds.color_name(f["stem"])), cv2.IMREAD_GRAYSCALE)
        t = tg.detect_tags(gray, cam["K"], cam["dist"], det, args.tag_size_mm, args.max_obliquity)
        pcd = o3d.io.read_point_cloud(str(ds.clean_cloud_path(capture, f["stem"])))
        tags.append(t)
        P.append(np.asarray(pcd.points))
        C.append(np.asarray(pcd.colors))
        print(f"[SHOT] {f['stem']}  {len(P[-1]):6d} cleaned points, usable tags "
              f"{sorted(i for i, r in t.items() if r['ok'])}")
    print()
    poses, info = reg.chain_sweep(tags, [f["true_deg"] for f in frames], names=[f["stem"] for f in frames],
                                  loop=args.loop)
    Pm, Cm = geo.voxel_fuse(np.vstack([geo.apply(T, p) for T, p in zip(poses, P)]), np.vstack(C),
                            args.voxel_mm / 1000.0)
    return poses, Pm, np.clip(Cm, 0, 1), info


# ---------------------------------------------------------------------------
# one pick -> numbers
# ---------------------------------------------------------------------------
def describe(X, F, cam, ref_stem, args):
    pc = reg.plant_coords(X, F)
    uv = geo.project(X[None], cam["K"], cam["dist"])[0]
    rec = dict(cam_mm=X * 1000, pixel=uv, plant_mm=pc["xyz"] * 1000, radius_mm=pc["radius"] * 1000,
               azimuth_deg=pc["azimuth_deg"], height_mm=pc["height"] * 1000, face_deg=pc["faces_camera_deg"],
               report_mm=reg.at_turn(X, F, args.report_deg) * 1000)

    def xyz(v):
        return f"x {v[0]:+8.1f}  y {v[1]:+8.1f}  z {v[2]:+8.1f} mm"
    lines = [f"  camera  shot {ref_stem} colour frame (turntable {F['ref_deg']:.1f} deg):  {xyz(rec['cam_mm'])}"
             f"   range {np.linalg.norm(X) * 1000:.1f} mm, pixel ({uv[0]:.1f}, {uv[1]:.1f})",
             f"  plant   {xyz(rec['plant_mm'])}   ->  radius {rec['radius_mm']:.1f} mm, "
             f"azimuth {rec['azimuth_deg']:+.1f} deg, height {rec['height_mm']:.1f} mm above the pot base",
             f"  turn    faces the camera at turntable {rec['face_deg']:6.1f} deg:  "
             f"{xyz(reg.at_turn(X, F, rec['face_deg']) * 1000)}"]
    if abs(args.report_deg - F["ref_deg"]) > 1e-6:
        lines.append(f"  file    camera at turntable {args.report_deg:6.1f} deg:           {xyz(rec['report_mm'])}")
    lines += [f"          camera at turntable {d:6.1f} deg:           {xyz(reg.at_turn(X, F, d) * 1000)}"
              for d in parse_floats(args.at_deg)]
    return rec, lines


def parse_floats(text):
    return [float(x) for x in text.split(",") if x.strip()] if text else []


def screen_pick(P, view, proj, size, xy, radius_px, depth_tol_m):
    """Index of the point a click at screen pixel xy selects, and its pixel distance (or None).

    Every point is projected with the viewer's OpenGL view / projection matrices. Of the
    points within radius_px of the cursor, those within depth_tol_m of the nearest one form
    the visible surface (points seen through gaps between the drawn front points are
    behind it and never win); of those, the one closest to the cursor is picked."""
    M = proj @ view
    h = P @ M[:3, :3].T + M[:3, 3]
    w = P @ M[3, :3] + M[3, 3]
    ahead = w > 1e-9
    w = np.where(ahead, w, 1.0)
    px = np.hypot((h[:, 0] / w + 1) / 2 * size[0] - xy[0], (1 - h[:, 1] / w) / 2 * size[1] - xy[1])
    cand = np.nonzero(ahead & (px <= radius_px))[0]
    if not len(cand):
        return None
    eye = -view[:3, :3].T @ view[:3, 3]                 # camera centre, world coordinates
    dist = np.linalg.norm(P[cand] - eye, axis=1)
    front = cand[dist <= dist.min() + depth_tol_m]
    k = int(front[np.argmin(px[front])])
    return k, float(px[k])


# ---------------------------------------------------------------------------
# viewer
# ---------------------------------------------------------------------------
class Picker:
    PANEL_EM = 24

    def __init__(self, pcd, F, cam, ref_stem, args, out, capture):
        self.F, self.cam, self.ref_stem, self.args, self.out, self.capture = F, cam, ref_stem, args, out, capture
        self.P = np.asarray(pcd.points)
        self.picks = []                               # dicts: rec, marker name, label
        self.write_outputs()                          # a new session starts with an empty points file

        app = gui.Application.instance
        app.initialize()
        self.win = app.create_window("Flora point picker", 1500, 950)
        em = self.win.theme.font_size

        self.view = gui.SceneWidget()
        self.view.scene = rendering.Open3DScene(self.win.renderer)
        self.view.scene.set_background([0.12, 0.12, 0.12, 1.0])
        self.view.scene.show_axes(False)
        pts = rendering.MaterialRecord()
        pts.shader, pts.point_size = "defaultUnlit", args.point_size * self.win.scaling
        self.view.scene.add_geometry("cloud", pcd, pts)
        self.lit = rendering.MaterialRecord()
        self.lit.shader = "defaultLit"
        line = rendering.MaterialRecord()
        line.shader, line.line_width = "unlitLine", 2.0 * self.win.scaling

        a, O = F["axis"], F["origin"]
        h0, h1 = np.array(F["height_range_mm"]) / 1000.0
        axis = o3d.geometry.LineSet(o3d.utility.Vector3dVector([O + a * (h0 - 0.02), O + a * (h1 + 0.03)]),
                                    o3d.utility.Vector2iVector([[0, 1]]))
        axis.paint_uniform_color([0.2, 0.9, 0.9])
        self.view.scene.add_geometry("axis", axis, line)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        frame.transform(geo.make_T(F["R"], O))
        self.view.scene.add_geometry("plant frame", frame, self.lit)
        self.view.set_on_mouse(self.on_mouse)

        self.panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self.panel.add_child(gui.Label("Ctrl + left click: pick a point\nDrag: rotate    Wheel: zoom"))
        self.panel.add_child(gui.Label("Cyan line: turntable axis\nAxes at the pot base: plant frame\n"
                                       "  red x -> camera, green y, blue z up"))
        row = gui.Horiz(0.5 * em)
        for text, cb in (("Undo", self.undo), ("Clear", self.clear), ("Reset view", self.reset_view)):
            b = gui.Button(text)
            b.set_on_clicked(cb)
            row.add_child(b)
        self.panel.add_child(row)
        for text, name in (("Turntable axis", "axis"), ("Plant frame", "plant frame")):
            c = gui.Checkbox(text)
            c.checked = True
            c.set_on_checked(lambda on, n=name: self.view.scene.show_geometry(n, on))
            self.panel.add_child(c)
        self.info = gui.Label("No point picked yet")
        self.panel.add_child(self.info)

        self.win.add_child(self.view)
        self.win.add_child(self.panel)
        self.win.set_on_layout(self.on_layout)
        self.reset_view()

    def run(self):
        gui.Application.instance.run()

    def on_layout(self, ctx):
        r = self.win.content_rect
        pw = self.PANEL_EM * ctx.theme.font_size
        self.view.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    def reset_view(self):
        """Look at the plant from the reference camera."""
        bb = self.view.scene.bounding_box
        self.view.setup_camera(35.0, bb, bb.get_center())
        self.view.look_at(self.P.mean(0).astype(np.float32), np.zeros(3, np.float32),
                          np.array([0.0, -1.0, 0.0], np.float32))

    # --- picking ----------------------------------------------------------
    def on_mouse(self, ev):
        if not (ev.type == gui.MouseEvent.Type.BUTTON_DOWN and ev.is_modifier_down(gui.KeyModifier.CTRL)
                and ev.is_button_down(gui.MouseButton.LEFT)):
            return gui.Widget.EventCallbackResult.IGNORED
        self.pick_at(ev.x - self.view.frame.x, ev.y - self.view.frame.y)
        return gui.Widget.EventCallbackResult.HANDLED

    def pick_at(self, x, y):
        """Pick the visible point under view pixel (x, y) (window units)."""
        fr, cam = self.view.frame, self.view.scene.camera
        if fr.width <= 0 or fr.height <= 0:
            return
        hit = screen_pick(self.P, np.asarray(cam.get_view_matrix()), np.asarray(cam.get_projection_matrix()),
                          (fr.width, fr.height), (x, y), self.args.pick_radius_px, self.args.pick_depth_mm / 1000.0)
        if hit is None:
            self.set_info("No point under the cursor -- click on the cloud")
        else:
            self.add_pick(*hit)

    def add_pick(self, idx, snap_px):
        X = self.P[idx]
        k = len(self.picks) + 1
        rec, lines = describe(X, self.F, self.cam, self.ref_stem, self.args)
        print(f"\n[PICK {k}] fused point #{idx} ({snap_px:.1f} px from the click)")
        print("\n".join(lines), flush=True)

        name = f"pick {k}"
        sph = o3d.geometry.TriangleMesh.create_sphere(self.args.marker_mm / 1000.0)
        sph.translate(X)
        sph.paint_uniform_color([1.0, 0.15, 0.15])
        sph.compute_vertex_normals()
        self.view.scene.add_geometry(name, sph, self.lit)
        label = self.view.add_3d_label(X + self.F["axis"] * self.args.marker_mm / 500.0, f"P{k}")
        self.picks.append(dict(rec=rec, name=name, label=label))
        c, p = rec["report_mm"], rec["plant_mm"]
        self.set_info(f"P{k}  camera, turntable {self.args.report_deg:.1f} deg, mm\n"
                      f"  x {c[0]:+.1f}  y {c[1]:+.1f}  z {c[2]:+.1f}\n"
                      f"P{k}  plant frame, mm\n"
                      f"  x {p[0]:+.1f}  y {p[1]:+.1f}  z {p[2]:+.1f}\n"
                      f"  radius {rec['radius_mm']:.1f}, azimuth {rec['azimuth_deg']:+.1f} deg\n"
                      f"  height {rec['height_mm']:.1f} above the pot base\n"
                      f"  faces camera at turntable {rec['face_deg']:.1f} deg\n"
                      f"{len(self.picks)} point(s) in the points file")
        self.write_outputs()

    def undo(self):
        if self.picks:
            p = self.picks.pop()
            self.view.scene.remove_geometry(p["name"])
            self.view.remove_3d_label(p["label"])
            print(f"\n[PICK {len(self.picks) + 1}] removed")
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
        """picked_points.txt (camera xyz at --report-deg) and picks.csv (everything)."""
        ds.write_points_file(self.args.points_file, [p["rec"]["report_mm"] for p in self.picks], self.capture,
                             f"colour camera, turntable at {self.args.report_deg:.2f} deg (first shot "
                             f"{self.ref_stem} = {self.F['ref_deg']:.2f} deg); x right, y down, z forward; mm")
        cols = ["pick", "cam_x_mm", "cam_y_mm", "cam_z_mm", "pixel_u", "pixel_v", "report_deg",
                "report_x_mm", "report_y_mm", "report_z_mm", "plant_x_mm", "plant_y_mm", "plant_z_mm",
                "radius_mm", "azimuth_deg", "height_mm", "faces_camera_at_deg"]
        with open(self.out / "picks.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(cols)
            for k, p in enumerate(self.picks, 1):
                r = p["rec"]
                wr.writerow([k, *np.round(r["cam_mm"], 2), *np.round(r["pixel"], 1), self.args.report_deg,
                             *np.round(r["report_mm"], 2), *np.round(r["plant_mm"], 2), round(r["radius_mm"], 2),
                             round(r["azimuth_deg"], 2), round(r["height_mm"], 2), round(r["face_deg"], 2)])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Tag-pose stitch of a cleaned turntable sweep + 3D point picking",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--max-obliquity", type=float, default=55.0, help="ignore tags seen more obliquely (deg)")
    ap.add_argument("--tag-size-mm", type=float, default=24.0, help="printed tag size (black square edge)")
    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--loop", choices=["auto", "yes", "no"], default="auto",
                    help="close the loop last->first (auto: when the sweep covers a full turn)")
    ap.add_argument("--voxel-mm", type=float, default=1.0, help="fused cloud voxel size")
    ap.add_argument("--out", default="tag_pick", help="output sub-folder of the capture")
    ap.add_argument("--restitch", action="store_true", help="ignore a cached stitch")
    ap.add_argument("--base-percentile", type=float, default=0.5,
                    help="plant origin height: this percentile of the cloud along the axis")
    ap.add_argument("--points-file", default=None,
                    help="picked camera coordinates, rewritten every pick (default <capture>/<out>/picked_points.txt)")
    ap.add_argument("--report-deg", type=float, default=None,
                    help="turntable angle the points file refers to (default: the first shot's angle)")
    ap.add_argument("--at-deg", default="", help="also print camera coordinates at these turntable angles, e.g. 90,180")
    ap.add_argument("--point-size", type=float, default=3.0)
    ap.add_argument("--marker-mm", type=float, default=2.5)
    ap.add_argument("--pick-radius-px", type=float, default=6.0,
                    help="how far from the cursor (screen px) a click may land on a point")
    ap.add_argument("--pick-depth-mm", type=float, default=5.0,
                    help="points this close to the front-most one under the cursor count as the visible surface")
    ap.add_argument("--no-view", action="store_true", help="stitch + plant frame only, no viewer")
    args = ap.parse_args()

    capture = Path(args.capture)
    try:
        cam = ds.load_color_camera(capture)
        cleaning = ds.read_cleaning(capture)
    except ds.DatasetError as e:
        sys.exit(str(e))
    frames = [f for f in cleaning["frames"] if f["ok"]]
    skipped = [f["stem"] for f in cleaning["frames"] if not f["ok"]]
    if skipped:
        print(f"[WARN] cleaning gave nothing for {skipped} -- stitching without them")
    if len(frames) < 2:
        sys.exit("fewer than 2 cleaned shots")

    out = capture / args.out
    out.mkdir(parents=True, exist_ok=True)
    args.points_file = Path(args.points_file) if args.points_file else out / "picked_points.txt"
    if args.report_deg is None:
        args.report_deg = frames[0]["true_deg"]
    settings = {k: getattr(args, k) for k in SETTING_KEYS}
    settings.update(cleaning=cleaning["finished"], stems=[f["stem"] for f in frames])

    pose_file, ply_file = out / "poses.json", out / "fused.ply"
    cached = None
    if not args.restitch and pose_file.is_file() and ply_file.is_file():
        cached = json.loads(pose_file.read_text())
        if cached.get("settings") != settings:
            print("[CACHE] settings or cleaning changed since the cached stitch -- stitching again")
            cached = None
    if cached:
        poses = [np.asarray(s["T_shot_to_ref"]) for s in cached["shots"]]
        pcd = o3d.io.read_point_cloud(str(ply_file))
        info = cached["registration"]
        print(f"[CACHE] reusing the stitch in {out} (--restitch to redo it)")
    else:
        try:
            poses, Pm, Cm, info = stitch(capture, frames, cam, args)
        except reg.RegistrationError as e:
            sys.exit(f"[FAIL] {e}")
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm))
        pcd.colors = o3d.utility.Vector3dVector(Cm)
        o3d.io.write_point_cloud(str(ply_file), pcd)
        pose_file.write_text(json.dumps({
            "convention": "X_ref = T_shot_to_ref @ X_shot (metres, colour-camera frames)",
            "reference_index": frames[0]["index"], "method": "tag_pose_pick:pose", "settings": settings,
            "registration": info,
            "shots": [dict(index=f["index"], stem=f["stem"], true_deg=f["true_deg"], T_shot_to_ref=T.tolist())
                      for f, T in zip(frames, poses)]}, indent=2))

    try:
        F = reg.plant_frame(poses, [f["true_deg"] for f in frames], np.asarray(pcd.points), args.base_percentile)
    except reg.RegistrationError as e:
        sys.exit(f"[FAIL] {e}")
    tilt = np.degrees(np.arccos(np.clip(F["axis"] @ np.array([0.0, -1.0, 0.0]), -1, 1)))
    dr = np.abs(F["drift_deg"])
    print(f"\n[CLOUD] {len(pcd.points)} points from {len(frames)} shots, in shot {frames[0]['stem']}'s "
          f"colour-camera frame")
    if info.get("loop"):
        print(f"[LOOP ] error {info['loop']['error_deg']:.2f} deg / {info['loop']['error_mm']:.1f} mm "
              f"(spread over the loop)")
    print(f"[POSE ] registered turn vs logged turntable angle: median {np.median(dr):.2f}, max {dr.max():.2f} deg")
    print(f"[PLANT] turntable axis {np.round(F['axis'], 4).tolist()} (camera frame), {tilt:.1f} deg from the "
          f"camera's -y")
    print(f"[PLANT] origin (pot base on the axis) {np.round(F['origin'] * 1000, 1).tolist()} mm; cloud spans "
          f"{F['height_range_mm'][0]:.0f} to {F['height_range_mm'][1]:.0f} mm in height")
    (out / "plant_frame.json").write_text(json.dumps(dict(
        convention="metres, reference colour-camera frame; X_cam = R @ X_plant + origin",
        reference_stem=frames[0]["stem"], reference_deg=F["ref_deg"], axis=F["axis"].tolist(),
        origin=F["origin"].tolist(), R=F["R"].tolist(), turn_sign=F["turn_sign"],
        base_percentile=args.base_percentile, height_range_mm=F["height_range_mm"]), indent=2))
    print(f"[OUT  ] {out}  (poses.json, fused.ply, plant_frame.json)")

    if not args.no_view:
        print(f"\n[VIEW ] Ctrl + left click on the cloud to pick points; close the window when done.\n"
              f"[VIEW ] picked camera coordinates (turntable at {args.report_deg:.1f} deg) -> {args.points_file}")
        Picker(pcd, F, cam, frames[0]["stem"], args, out, capture).run()
        print(f"\n[DONE ] {args.points_file}")


if __name__ == "__main__":
    main()
