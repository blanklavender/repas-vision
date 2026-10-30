#!/usr/bin/env python3
"""
stitch_view.py  --  step 4 of top_view_combine: stitch the cleaned side shots with their
AprilTag poses, add the cleaned top shot through the robot-arm transform, and show it all
in a 3D viewer (= ../tag_pose_pick.py without the picking).

Frames (metres, colour-optical frames: x right, y down, z forward):
    ref  the first side shot's colour camera = C1, the camera at robot pose 1, with the
         turntable at the first shot's angle. Side shot k:  X_ref = T_shot_to_ref @ X_k
    C2   the colour camera at robot pose 2 (top view).    X_ref = C1_T_C2 @ X_C2
C1_T_C2 is read from the pose-transform file as an absolute value (no refinement): the
matrix itself, or inv(C2_T_C1), or inv(B_T_C1) @ B_T_C2. If the turntable did not stand at
the first shot's angle for the top shot, --top-deg turns the top cloud back about the
turntable axis to that angle.

Every side cloud is moved into ref and voxel-fused (--voxel-mm); the top cloud is moved
into ref and fused on top. Printed check: for the top points inside the side cloud's radial
extent (where both clouds exist), the distance to the nearest side point -- the transform
is only as good as the hand-eye / board calibration.

Viewer: drag rotates, wheel zooms; checkboxes show / hide the side sweep, the top shot (or
tint it magenta), the turntable axis, the plant frame and both camera poses.

    python stitch_view.py ../../../captures/femto_bolt/flora_unit/plant_10-16_deg_5_speed_5
    python stitch_view.py <capture> --voxel-mm 2 --snapshots --no-view
    python stitch_view.py <capture> --top-deg 355 --transforms <file>

I/O
---
in   <capture>/top_view_combine/registration.json                 register_side.py
     <capture>/top_view_combine/side/cleaning.json, pointcloud/    clean_side.py
     <capture>/top_view_combine/top/cleaning_top.json, cloud_top.ply   clean_top.py
     <capture>/robot_rgbd/camera_pose_transforms*.txt (or --transforms)
out  <capture>/top_view_combine/stitch/
        side_fused.ply     the side shots, ref frame
        top_in_ref.ply     the top shot, ref frame
        combined.ply       both, voxel-fused
        poses.json         T_shot_to_ref per side shot, T_top_to_ref + where it came from, check
        plant_frame.json   turntable axis, plant origin and axes (ref frame)
        snap_views.png     with --snapshots: front / side / top orthographic renders
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

import tv_common as tv
from flora_lib import dataset as ds
from flora_lib import geometry as geo
from flora_lib import registration as reg

TOP_TINT = np.array([1.0, 0.1, 0.85])


# ---------------------------------------------------------------------------
# stitching
# ---------------------------------------------------------------------------
def stitch_side(side, frames, poses, voxel):
    """Cleaned clouds of `frames`, moved by `poses` (T_shot_to_ref) -> fused points + colours."""
    P, C = [], []
    for f, T in zip(frames, poses):
        pcd = o3d.io.read_point_cloud(str(ds.clean_cloud_path(side, f["stem"])))
        P.append(geo.apply(T, np.asarray(pcd.points)))
        C.append(np.asarray(pcd.colors))
        print(f"[SIDE] {f['stem']}  {len(P[-1]):6d} cleaned points")
    Pm, Cm = geo.voxel_fuse(np.vstack(P), np.vstack(C), voxel)
    return Pm, np.clip(Cm, 0, 1)


def turn_back(A, top_deg):
    """4x4 that undoes the turntable standing at top_deg instead of the reference angle."""
    rad = -A["turn_sign"] * np.radians(top_deg - A["ref_deg"])
    R = geo.rotate_about(A["axis"], rad)
    return geo.make_T(R, A["point"] - R @ A["point"])


def to_pcd(P, C):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.colors = o3d.utility.Vector3dVector(C)
    return pcd


# ---------------------------------------------------------------------------
# orthographic snapshots (no GUI needed)
# ---------------------------------------------------------------------------
def render(q, C, right, up, near, mm_per_px, pad=10):
    """Plant-frame points q -> orthographic image (BGR), nearest point wins, 2 px splats."""
    r, u, n = q @ right, q @ up, q @ near
    res = mm_per_px / 1000.0
    col = ((r - r.min()) / res).astype(int) + pad
    row = ((u.max() - u) / res).astype(int) + pad
    img = np.full((row.max() + pad + 2, col.max() + pad + 2, 3), 30, np.uint8)
    o = np.argsort(n)                                   # far first, near last (overwrites)
    bgr = (np.clip(C[o][:, ::-1], 0, 1) * 255).astype(np.uint8)
    for dy in (0, 1):
        for dx in (0, 1):
            img[row[o] + dy, col[o] + dx] = bgr
    return img


def snapshots(path, Ps, Cs, Pt, Ct, F, mm_per_px):
    q_all = (np.vstack([Ps, Pt]) - F["origin"]) @ F["R"]
    C_true = np.vstack([Cs, Ct])
    C_src = np.vstack([np.tile([0.55, 0.75, 0.75], (len(Ps), 1)), np.tile(TOP_TINT, (len(Pt), 1))])
    X, Y, Z = np.eye(3)
    views = [("front (from camera)", Y, Z, X), ("side", -X, Z, Y), ("top", Y, -X, Z)]
    rows = []
    for C, tag in ((C_true, "colour"), (C_src, "side = grey, top shot = magenta")):
        tiles = []
        for name, right, up, near in views:
            im = render(q_all, C, right, up, near, mm_per_px)
            cv2.putText(im, f"{name} | {tag}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            tiles.append(im)
        h = max(t.shape[0] for t in tiles)
        rows.append(cv2.hconcat([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                                 for t in tiles]))
    w = max(r.shape[1] for r in rows)
    cv2.imwrite(str(path), cv2.vconcat([cv2.copyMakeBorder(r, 0, 4, 0, w - r.shape[1], cv2.BORDER_CONSTANT,
                                                           value=(0, 0, 0)) for r in rows]))


# ---------------------------------------------------------------------------
# viewer
# ---------------------------------------------------------------------------
class Viewer:
    PANEL_EM = 20

    def __init__(self, side_pcd, top_pcd, F, T_top, cam, args, title="Flora side + top view"):
        from open3d.visualization import gui, rendering
        self.gui, self.F, self.T_top = gui, F, T_top
        self.P = np.vstack([np.asarray(side_pcd.points), np.asarray(top_pcd.points)])

        app = gui.Application.instance
        app.initialize()
        self.win = app.create_window(title, 1500, 950)
        em = self.win.theme.font_size

        self.view = gui.SceneWidget()
        self.view.scene = rendering.Open3DScene(self.win.renderer)
        self.view.scene.set_background([0.12, 0.12, 0.12, 1.0])
        self.view.scene.show_axes(False)
        pts = rendering.MaterialRecord()
        pts.shader, pts.point_size = "defaultUnlit", args.point_size * self.win.scaling
        lit = rendering.MaterialRecord()
        lit.shader = "defaultLit"
        line = rendering.MaterialRecord()
        line.shader, line.line_width = "unlitLine", 2.0 * self.win.scaling

        tint = o3d.geometry.PointCloud(top_pcd)
        tint.paint_uniform_color(TOP_TINT)
        self.view.scene.add_geometry("side", side_pcd, pts)
        self.view.scene.add_geometry("top", top_pcd, pts)
        self.view.scene.add_geometry("top tint", tint, pts)
        self.view.scene.show_geometry("top tint", False)

        a, O = F["axis"], F["origin"]
        h0, h1 = np.array(F["height_range_mm"]) / 1000.0
        axis = o3d.geometry.LineSet(o3d.utility.Vector3dVector([O + a * (h0 - 0.02), O + a * (h1 + 0.03)]),
                                    o3d.utility.Vector2iVector([[0, 1]]))
        axis.paint_uniform_color([0.2, 0.9, 0.9])
        self.view.scene.add_geometry("axis", axis, line)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        frame.transform(geo.make_T(F["R"], O))
        self.view.scene.add_geometry("plant frame", frame, lit)
        for name, T, colour in (("camera C1", np.eye(4), [1.0, 0.85, 0.2]), ("camera C2", T_top, list(TOP_TINT))):
            fr = o3d.geometry.LineSet.create_camera_visualization(
                cam["size"][0], cam["size"][1], cam["K"], np.linalg.inv(T), scale=0.06)
            fr.paint_uniform_color(colour)
            self.view.scene.add_geometry(name, fr, line)

        self.panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self.panel.add_child(gui.Label("Drag: rotate    Wheel: zoom\nCyan line: turntable axis\n"
                                       "Axes at the pot base: plant frame\n  red x -> camera, green y, blue z up\n"
                                       "Yellow frustum: camera pose 1 (side)\nMagenta frustum: camera pose 2 (top)"))
        row = gui.Horiz(0.5 * em)
        for text, cb in (("Side camera view", self.reset_view), ("Top camera view", self.top_view)):
            b = gui.Button(text)
            b.set_on_clicked(cb)
            row.add_child(b)
        self.panel.add_child(row)
        for text, name, on in (("Side sweep", "side", True), ("Top shot", "top", True),
                               ("Tint top shot magenta", "top tint", False), ("Turntable axis", "axis", True),
                               ("Plant frame", "plant frame", True), ("Camera pose 1", "camera C1", True),
                               ("Camera pose 2", "camera C2", True)):
            c = gui.Checkbox(text)
            c.checked = on
            c.set_on_checked(lambda v, n=name: self.toggle(n, v))
            self.panel.add_child(c)
        self.panel.add_child(gui.Label(args.info))

        self.win.add_child(self.view)
        self.win.add_child(self.panel)
        self.win.set_on_layout(self.on_layout)
        self.tint_on = False
        self.top_on = True
        self.reset_view()

    def run(self):
        self.gui.Application.instance.run()

    def toggle(self, name, on):
        if name in ("top", "top tint"):                  # one of the two top clouds at a time
            if name == "top":
                self.top_on = on
            else:
                self.tint_on = on
            self.view.scene.show_geometry("top", self.top_on and not self.tint_on)
            self.view.scene.show_geometry("top tint", self.top_on and self.tint_on)
        else:
            self.view.scene.show_geometry(name, on)

    def on_layout(self, ctx):
        r = self.win.content_rect
        pw = self.PANEL_EM * ctx.theme.font_size
        self.view.frame = self.gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = self.gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    def look(self, eye, up):
        bb = self.view.scene.bounding_box
        self.view.setup_camera(35.0, bb, bb.get_center())
        self.view.look_at(self.P.mean(0).astype(np.float32), np.asarray(eye, np.float32), np.asarray(up, np.float32))

    def reset_view(self):
        """Look at the plant from camera pose 1."""
        self.look(np.zeros(3), [0.0, -1.0, 0.0])

    def top_view(self):
        """Look at the plant from camera pose 2."""
        self.look(self.T_top[:3, 3], -self.T_top[:3, 1])


# ---------------------------------------------------------------------------
# inputs (shared with downsample_view.py)
# ---------------------------------------------------------------------------
def add_scene_args(ap):
    ap.add_argument("capture")
    ap.add_argument("--work", default=tv.WORK_DIR, help="work sub-folder of the capture")
    ap.add_argument("--robot-dir", default=tv.ROBOT_DIR, help="sub-folder of the capture holding the top shot")
    ap.add_argument("--transforms", default=None,
                    help="pose-transform file (default: newest robot_rgbd/camera_pose_transforms*)")
    ap.add_argument("--side-frame", default="C1", help="frame name of the side camera in the transform file")
    ap.add_argument("--top-frame", default="C2", help="frame name of the top camera in the transform file")
    ap.add_argument("--top-deg", type=float, default=None,
                    help="turntable angle during the top shot (default: the first side shot's angle)")
    ap.add_argument("--base-percentile", type=float, default=0.5,
                    help="plant origin height: this percentile of the side cloud along the axis")
    ap.add_argument("--point-size", type=float, default=3.0)
    ap.add_argument("--snap-mm-per-px", type=float, default=0.4, help="snapshot resolution")


def load_scene(args):
    """Registration, side cleaning, top cleaning and the top transform -> dict (exits on error)."""
    capture = Path(args.capture)
    work = tv.work_dir(capture, args.work)
    side, top = work / tv.SIDE_DIR, work / tv.TOP_DIR
    try:
        regn = ds.read_registration(work)
        cleaning = ds.read_cleaning(side)
        cam = ds.load_color_camera(capture)
        tpath = tv.find_transforms_file(capture, args.robot_dir, args.transforms)
        mats = tv.read_transforms(tpath)
        T_c1c2, how, check = tv.top_to_side(mats, args.side_frame, args.top_frame)
    except (ds.DatasetError, KeyError) as e:
        sys.exit(str(e))
    top_summary = top / tv.TOP_CLEANING
    if not top_summary.is_file() or not json.loads(top_summary.read_text()).get("ok"):
        sys.exit(f"missing or failed {top_summary} -- run clean_top.py first")
    if cleaning["settings"].get("registration") != regn["finished"]:
        sys.exit(f"[FAIL] {side / ds.CLEANING} was made with another registration than "
                 f"{work / ds.REGISTRATION} -- run clean_side.py again")
    frames = [f for f in cleaning["frames"] if f["ok"] and f["stem"] in regn["by_stem"]]
    skipped = [f["stem"] for f in cleaning["frames"] if f not in frames]
    if skipped:
        print(f"[WARN] cleaning gave nothing for {skipped} -- stitching without them")
    if len(frames) < 2:
        sys.exit("fewer than 2 cleaned side shots")
    ref = regn["shots"][0]
    poses = [regn["by_stem"][f["stem"]]["T"] for f in frames]
    A = regn["axis"]
    top_deg = A["ref_deg"] if args.top_deg is None else args.top_deg
    T_top = turn_back(A, top_deg) @ T_c1c2

    print(f"[TF  ] {tpath.name}: {sorted(mats)}")
    print(f"[TF  ] X_ref = {how} @ X_top  -- top camera {np.round(T_c1c2[:3, 3] * 1000, 1).tolist()} mm from the "
          f"side camera, viewing {geo.rot_deg(T_c1c2[:3, :3]):.1f} deg away from it")
    if check:
        print(f"[TF  ] consistency with the base-frame matrices: {check['rot_deg']:.4f} deg / "
              f"{check['trans_mm']:.3f} mm")
    if abs(top_deg - A["ref_deg"]) > 1e-6:
        print(f"[TF  ] top shot at turntable {top_deg:.1f} deg -> turned back to {A['ref_deg']:.1f} deg about the axis")
    return dict(capture=capture, work=work, side=side, top=top, regn=regn, cleaning=cleaning, cam=cam,
                frames=frames, poses=poses, A=A, ref=ref, T_top=T_top, T_c1c2=T_c1c2, how=how, check=check,
                tpath=tpath, top_deg=top_deg)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stitch the side sweep + the robot-arm top shot and view it",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_scene_args(ap)
    ap.add_argument("--voxel-mm", type=float, default=1.0, help="fused cloud voxel size")
    ap.add_argument("--snapshots", action="store_true", help="also write stitch/snap_views.png")
    ap.add_argument("--no-view", action="store_true", help="stitch + outputs only, no viewer")
    args = ap.parse_args()

    S = load_scene(args)
    work, side, top, frames, poses, A, ref = (S[k] for k in ("work", "side", "top", "frames", "poses", "A", "ref"))
    regn, cleaning, cam, T_top, T_c1c2, how, check, tpath, top_deg = (
        S[k] for k in ("regn", "cleaning", "cam", "T_top", "T_c1c2", "how", "check", "tpath", "top_deg"))

    out = work / tv.STITCH_DIR
    out.mkdir(parents=True, exist_ok=True)
    voxel = args.voxel_mm / 1000.0
    Ps, Cs = stitch_side(side, frames, poses, voxel)
    top_pcd = o3d.io.read_point_cloud(str(top / "cloud_top.ply"))
    Pt = geo.apply(T_top, np.asarray(top_pcd.points))
    Ct = np.asarray(top_pcd.colors)
    Pm, Cm = geo.voxel_fuse(np.vstack([Ps, Pt]), np.vstack([Cs, Ct]), voxel)
    side_pcd, top_ref = to_pcd(Ps, Cs), to_pcd(Pt, Ct)
    o3d.io.write_point_cloud(str(out / "side_fused.ply"), side_pcd)
    o3d.io.write_point_cloud(str(out / "top_in_ref.ply"), top_ref)
    o3d.io.write_point_cloud(str(out / "combined.ply"), to_pcd(Pm, np.clip(Cm, 0, 1)))

    # the side cloud only reaches its radial crop: judge the top shot where both exist
    r_side = np.percentile(reg.radial_distance(Ps, A["axis"], A["point"]), 99)
    overlap = reg.radial_distance(Pt, A["axis"], A["point"]) < r_side
    d = np.asarray(to_pcd(Pt[overlap], Ct[overlap]).compute_point_cloud_distance(side_pcd)) * 1000 \
        if overlap.any() else np.full(1, np.nan)
    fit = dict(side_radius_mm=float(r_side * 1000), n_overlap=int(overlap.sum()), median_mm=float(np.median(d)),
               p90_mm=float(np.percentile(d, 90)), within_3mm=float((d < 3).mean()),
               within_5mm=float((d < 5).mean()), within_10mm=float((d < 10).mean()))
    F = reg.plant_frame(A, Ps, args.base_percentile)
    ht = (Pt - F["origin"]) @ F["axis"] * 1000

    (out / "poses.json").write_text(json.dumps(dict(
        convention="X_ref = T @ X (metres, colour-optical frames); ref = first side shot = camera pose 1",
        reference_stem=ref["stem"], settings=dict(voxel_mm=args.voxel_mm, top_deg=top_deg,
                                                  registration=regn["finished"], cleaning=cleaning["finished"]),
        top=dict(transforms_file=str(tpath), from_matrix=how, T_C1_C2=T_c1c2.tolist(), T_top_to_ref=T_top.tolist(),
                 base_frame_check=check, fit_to_side=fit),
        shots=[dict(index=f["index"], stem=f["stem"], true_deg=f["true_deg"], T_shot_to_ref=T.tolist())
               for f, T in zip(frames, poses)]), indent=2))
    (out / "plant_frame.json").write_text(json.dumps(dict(
        convention="metres, reference colour-camera frame; X_cam = R @ X_plant + origin",
        reference_stem=ref["stem"], reference_deg=F["ref_deg"], axis=F["axis"].tolist(),
        origin=F["origin"].tolist(), R=F["R"].tolist(), turn_sign=F["turn_sign"],
        base_percentile=args.base_percentile, height_range_mm=F["height_range_mm"]), indent=2))

    print(f"\n[CLOUD] side {len(Ps):,} points from {len(frames)} shots + top {len(Pt):,} points -> "
          f"combined {len(Pm):,} ({args.voxel_mm:g} mm voxels), shot {ref['stem']}'s colour-camera frame")
    print(f"[PLANT] origin (pot base on the axis) {np.round(F['origin'] * 1000, 1).tolist()} mm; side cloud "
          f"{F['height_range_mm'][0]:.0f} to {F['height_range_mm'][1]:.0f} mm, top shot "
          f"{np.percentile(ht, 0.1):.0f} to {np.percentile(ht, 99.9):.0f} mm above the pot base")
    print(f"[FIT  ] {fit['n_overlap']:,} of {len(Pt):,} top points lie inside the side cloud's "
          f"{fit['side_radius_mm']:.0f} mm radius (the rest is plant only the top shot sees)")
    print(f"[FIT  ] those -> nearest side point: median {fit['median_mm']:.1f} mm, p90 {fit['p90_mm']:.1f} mm; "
          f"{fit['within_3mm']:.0%} within 3 mm, {fit['within_5mm']:.0%} within 5 mm, "
          f"{fit['within_10mm']:.0%} within 10 mm")
    if args.snapshots:
        snapshots(out / "snap_views.png", Ps, Cs, Pt, Ct, F, args.snap_mm_per_px)
    print(f"[OUT  ] {out}  (side_fused.ply, top_in_ref.ply, combined.ply, poses.json, plant_frame.json"
          f"{', snap_views.png' if args.snapshots else ''})")

    if not args.no_view:
        args.info = (f"side {len(Ps):,} pts, {len(frames)} shots\ntop {len(Pt):,} pts\n"
                     f"top -> side fit (overlap):\n  median {fit['median_mm']:.1f} mm, "
                     f"{fit['within_5mm']:.0%} within 5 mm")
        print("\n[VIEW ] close the window when done")
        Viewer(side_pcd, top_ref, F, T_top, cam, args).run()


if __name__ == "__main__":
    main()
