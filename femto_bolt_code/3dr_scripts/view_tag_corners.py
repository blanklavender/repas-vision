#!/usr/bin/env python3
"""
view_tag_corners.py  --  where does one tag's corners land in every shot, after the
tag_stitch.py poses are applied?

Nothing is re-registered here. Each shot's pose T_shot_to_ref comes straight from
tag_stitch.py (all shared tags jointly, per adjacent pair, then chain / pose graph).
This script takes ONE tag's 4 corners as measured in each shot, applies that shot's
pose, and draws them -- a coloured marker set per shot (spheres on the corners,
corner 0 larger, thick-ish edges), toggleable per shot. If the poses are right, all
shots' squares sit on top of each other.

    python view_tag_corners.py ../../captures/femto_bolt/flora_unit/yellow_pot_09-30_deg_10_speed_5/stitch_pnp
    python view_tag_corners.py <stitch_dir> --tag-id 15 --corners plane --with-fused

I/O contract
------------
in   <stitch_dir>/poses.json           T_shot_to_ref per shot (tag_stitch.py)
     <stitch_dir>/tags_per_shot.json   per-shot tag corners, each shot's colour-camera frame
     <stitch_dir>/fused.ply            only with --with-fused (context cloud)
out  console: per shot pose rotation vs logged turntable angle, and each corner's
     distance from its mean over shots (mm); an Open3D viewer
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from single_tag_masked import corner_markers, shot_colors


def main():
    ap = argparse.ArgumentParser(description="One tag's corners in every shot, under the tag_stitch poses",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("stitch", help="tag_stitch.py output folder, e.g. <capture>/stitch_pnp")
    ap.add_argument("--tag-id", type=int, default=15)
    ap.add_argument("--corners", choices=["pnp", "plane"], default="pnp",
                    help="which per-shot corner measurement to draw")
    ap.add_argument("--include-unusable", action="store_true",
                    help="also draw shots where tag_stitch marked the tag unusable (drawn as cubes)")
    ap.add_argument("--with-fused", action="store_true", help="show fused.ply for context")
    ap.add_argument("--corner-size-mm", type=float, default=1.5)
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--no-view", action="store_true", help="print the numbers only")
    args = ap.parse_args()

    stitch = Path(args.stitch)
    for f in ("poses.json", "tags_per_shot.json"):
        if not (stitch / f).is_file():
            sys.exit(f"missing {stitch / f} -- run tag_stitch.py first")
    poses = json.loads((stitch / "poses.json").read_text())
    T = {s["index"]: np.asarray(s["T_shot_to_ref"]) for s in poses["shots"]}
    ref_idx = poses["reference_index"]
    ref_deg = next((s["true_deg"] for s in poses["shots"] if s["index"] == ref_idx), 0.0)

    key, ok_key = ("corners_pnp", "ok_pnp") if args.corners == "pnp" else ("corners_plane", "ok")
    shots = []
    for s in json.loads((stitch / "tags_per_shot.json").read_text())["shots"]:
        rec = s["tags"].get(str(args.tag_id))
        if rec is None or rec.get(key) is None:
            continue
        usable = bool(rec.get(ok_key))
        if not usable and not args.include_unusable:
            continue
        if s["index"] not in T:
            print(f"[SKIP] {s['index']:03d}: shot has no pose (not connected in tag_stitch)")
            continue
        Tk = T[s["index"]]
        C = np.asarray(rec[key]) @ Tk[:3, :3].T + Tk[:3, 3]
        rot = float(np.degrees(np.arccos(np.clip((np.trace(Tk[:3, :3]) - 1) / 2, -1, 1))))
        logged = abs(((s["true_deg"] - ref_deg) + 180) % 360 - 180)
        shots.append(dict(index=s["index"], C=C, usable=usable, rot=rot, logged=logged))
    if not shots:
        sys.exit(f"tag {args.tag_id} has no {args.corners} corners in any posed shot")

    cols = shot_colors(len(shots))
    for s, c in zip(shots, cols):
        s["pal"] = c

    # ---- numbers ------------------------------------------------------------
    good = [s for s in shots if s["usable"]]
    mean = np.mean([s["C"] for s in good], 0) if good else np.mean([s["C"] for s in shots], 0)
    print(f"tag {args.tag_id}, {args.corners} corners, poses from {stitch.name} "
          f"({poses.get('method')}), reference shot {ref_idx:03d}\n")
    print("  shot   pose_rot  logged  diff   colour (RGB)         distance from mean corner (mm)")
    print("                                                         c0     c1     c2     c3   mean")
    for s in shots:
        d = np.linalg.norm(s["C"] - mean, axis=1) * 1000
        rgb = str(tuple(int(255 * x) for x in s["pal"]))
        print(f"   {s['index']:03d}  {s['rot']:7.2f}  {s['logged']:6.2f} {s['rot'] - s['logged']:+6.2f}"
              f"   {rgb:<18} " + " ".join(f"{x:6.2f}" for x in d) + f" {d.mean():6.2f}"
              + ("" if s["usable"] else "   (unusable in tag_stitch)"))
    if len(good) > 1:
        dev = np.stack([np.linalg.norm(s["C"] - mean, axis=1) for s in good]) * 1000
        print(f"\n  RMS spread over {len(good)} usable shots: "
              + " ".join(f"c{k} {x:.2f}" for k, x in enumerate(np.sqrt((dev ** 2).mean(0))))
              + f"  | overall {np.sqrt((dev ** 2).mean()):.2f} mm")

    # ---- viewer -------------------------------------------------------------
    if args.no_view:
        return
    size = args.corner_size_mm / 1000.0
    items, names = [], {}
    if args.with_fused:
        f = stitch / "fused.ply"
        if not f.is_file():
            sys.exit(f"missing {f}")
        items.append(dict(name="fused cloud", geometry=o3d.io.read_point_cloud(str(f)), is_visible=True))
    for s in shots:
        m, ls = corner_markers(s["C"], s["pal"], size, cubes=not s["usable"])
        tube = edge_tubes(s["C"], s["pal"], size * 0.25)
        i = s["index"]
        names[i] = [f"corners {i:03d}", f"edges {i:03d}"]
        items.append(dict(name=names[i][0], geometry=m, group="corners", is_visible=True))
        items.append(dict(name=names[i][1], geometry=tube, group="edges", is_visible=True))
    idxs = [s["index"] for s in shots]
    top = mean[mean[:, 1].argmin()]
    label_pos = {s["index"]: top + np.array([0.0, -0.004 - 0.003 * k, 0.0]) for k, s in enumerate(shots)}

    def labels(vis, sel):
        if args.no_labels:
            return
        vis.clear_3d_labels()
        for s in shots:
            if s["index"] in sel:
                lab = vis.add_3d_label(label_pos[s["index"]], f"{s['index']:03d}")
                try:
                    lab.color = o3d.visualization.gui.Color(*[float(x) for x in s["pal"]])
                except Exception:
                    pass

    state = {"k": 0}

    def show(vis, sel):
        for i in idxs:
            for n in names[i]:
                vis.show_geometry(n, i in sel)
        labels(vis, sel)

    def step(vis, d):
        state["k"] = (state["k"] + d) % len(idxs)
        show(vis, {idxs[state["k"]]})
        print(f"[VIEW] solo {idxs[state['k']]:03d}")

    def add_next(vis):
        state["k"] = (state["k"] + 1) % len(idxs)
        sel = set(idxs[:state["k"] + 1])
        show(vis, sel)
        print(f"[VIEW] shots {sorted(sel)}")

    actions = [("All", lambda v: show(v, set(idxs))),
               ("None", lambda v: show(v, set())),
               ("Solo next", lambda v: step(v, +1)),
               ("Solo prev", lambda v: step(v, -1)),
               ("Add next", add_next)]
    print("\n[VIEW] tick shots in the scene list or use the buttons; corner 0 = larger marker"
          + ("; cubes = tag unusable in tag_stitch" if args.include_unusable else ""))
    center = mean.mean(0)
    o3d.visualization.draw(items, title=f"tag {args.tag_id} corners per shot", show_ui=True,
                           point_size=2, lookat=center.astype(np.float32),
                           eye=(center + np.array([0, 0, -0.12])).astype(np.float32),
                           up=np.array([0, -1, 0], np.float32), show_skybox=False,
                           actions=actions, on_init=lambda vis: labels(vis, set(idxs)))


def edge_tubes(C, color, radius):
    """The 4 tag edges as thin cylinders -- visible at any zoom, unlike 1-px lines."""
    mesh = o3d.geometry.TriangleMesh()
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        v = C[b] - C[a]
        L = float(np.linalg.norm(v))
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius, L, resolution=12)
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z, v / L)
        if np.linalg.norm(axis) > 1e-9:
            ang = np.arccos(np.clip(z @ (v / L), -1, 1))
            cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * ang),
                       center=np.zeros(3))
        cyl.translate((C[a] + C[b]) / 2)
        mesh += cyl
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


if __name__ == "__main__":
    main()
