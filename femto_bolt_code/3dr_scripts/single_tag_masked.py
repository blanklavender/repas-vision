#!/usr/bin/env python3
"""
single_tag_masked.py  --  single-tag stitching test on the new burst captures.

Same idea as tag_stitch_single_tag.py, but on the new folder structure and the
cleaned, SAM3-masked clouds:

1. Follow ONE tag (--tag-id) through every shot where it is usable.
2. Per shot, get its 4 corners in 3D (that shot's colour-camera frame):
       --corners pnp    colour-only PnP with the known --tag-size-mm (default;
                        immune to the ToF depth bias on the tag cards)
       --corners plane  colour rays on a plane fitted to the tag's depth points
3. Register each shot's 4 corners onto the reference shot's 4 corners (Kabsch,
   or translation only with --fit translation). No turntable angle or axis is
   used; the logged turntable angle is only printed next to the fitted
   rotation, for comparison.
4. Move each shot's cloud (averaged depth -> weight/validity filter -> SAM3
   mask) with that transform and show them in a viewer where every shot can be
   toggled on/off.

    python single_tag_masked.py ../../captures/femto_bolt/flora_unit/yellow_pot_09-30_deg_10_speed_5
    python single_tag_masked.py <capture> --tag-id 18 --corners plane --fit translation
    python single_tag_masked.py <capture> --tag-id 16 --indices 0-6,32-35 --save --no-view

I/O contract
------------
in   <capture>/camera_params.json, manifest.csv, color_*.png       (sweep_capture.py)
     <capture>/stops/<stop>/avg/ depth_mean_mm.npy, weight.npy, valid_mask.png
                                                                    (average_confidence.py)
     <capture>/stops/<stop>/mask.png                                (sam3_batch_masks.py;
                                                                     skip with --no-mask)
out  console table: per shot fitted rotation vs logged turntable rotation, corner RMS
     with --save, <capture>/<--out, default single_tag<ID>>/:
        aligned_<stop>.ply   masked cloud moved into the reference shot's frame
        merged.ply           all of them together (1 mm confidence-weighted voxels)
        transforms.json      T_shot_to_ref per shot + the numbers in the table
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from pupil_apriltags import Detector

import tag_stitch as ts            # cloud building, tag measurement, PnP, masks

def shot_colors(n):
    """n clearly different colours (hue steps, alternating brightness so neighbours differ)."""
    out = []
    for k in range(n):
        hsv = np.uint8([[[int(180 * k / max(n, 1)) % 180, 230, 255 if k % 2 == 0 else 170]]])
        out.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0] / 255.0)
    return np.array(out)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
def fit_transform(src, dst, mode):
    """src, dst: (4,3) corresponding corners -> 4x4 T with T*src ~= dst."""
    if mode == "rigid":
        R, t = ts.kabsch(src, dst)
    else:
        R, t = np.eye(3), dst.mean(0) - src.mean(0)
    return ts.make_T(R, t)


def spread_report(shots, key, label):
    """How far each corner lands from its mean over shots (reference frame, mm)."""
    have = [s for s in shots if s.get(key) is not None]
    if len(have) < 2:
        return
    C = np.stack([s[key] for s in have]) * 1000                   # (shots, 4, 3)
    dev = np.linalg.norm(C - C.mean(0), axis=2)                   # (shots, 4)
    print(f"\n  {label}: distance of each corner from its mean position (mm)")
    print("  shot    colour (RGB)          c0      c1      c2      c3    mean")
    for s, d in zip(have, dev):
        rgb = tuple(int(255 * x) for x in s["pal"])
        print(f"   {s['index']:03d}   {str(rgb):<18} " + "  ".join(f"{x:6.2f}" for x in d)
              + f"  {d.mean():6.2f}")
    print("   RMS over shots:            " + "  ".join(f"{x:6.2f}" for x in np.sqrt((dev ** 2).mean(0)))
          + f"  {np.sqrt((dev ** 2).mean()):6.2f}")


# ---------------------------------------------------------------------------
# viewer (same layout as tag_stitch_single_tag.py)
# ---------------------------------------------------------------------------
def corner_markers(C, color, size_m, cubes=False):
    """Spheres (or cubes) on the 4 corners + outline; corner 0 drawn larger to show orientation."""
    m = o3d.geometry.TriangleMesh()
    for k, p in enumerate(C):
        r = size_m * (1.8 if k == 0 else 1.0)
        g = (o3d.geometry.TriangleMesh.create_box(r, r, r).translate(-np.full(3, r / 2)) if cubes
             else o3d.geometry.TriangleMesh.create_sphere(r / 2))
        g.translate(p)
        m += g
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(C),
                              o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]]))
    m.paint_uniform_color(color)
    ls.paint_uniform_color(color)
    m.compute_vertex_normals()
    return m, ls


def viewer(shots, ref_idx, center, tag_id, args):
    idxs = [s["index"] for s in shots]
    size = args.corner_size_mm / 1000.0
    items = []
    for s in shots:
        i, col = s["index"], s["pal"]
        items.append(dict(name=f"cloud {i:03d}", geometry=s["cloud"], group="clouds", is_visible=True))
        m, ls = corner_markers(s["C_ref"], col, size)
        items.append(dict(name=f"corners {i:03d}", geometry=m, group="corners", is_visible=True))
        items.append(dict(name=f"outline {i:03d}", geometry=ls, group="corners", is_visible=True))
        if s.get("Cp_ref") is not None:
            m, ls = corner_markers(s["Cp_ref"], col, size * 0.8, cubes=True)
            items.append(dict(name=f"depth corners {i:03d}", geometry=m, group="depth corners",
                              is_visible=True))
            items.append(dict(name=f"depth outline {i:03d}", geometry=ls, group="depth corners",
                              is_visible=True))
    have = {it["name"] for it in items}
    names = {i: [n for n in (f"corners {i:03d}", f"outline {i:03d}", f"depth corners {i:03d}",
                             f"depth outline {i:03d}") if n in have] for i in idxs}

    # shot labels: a column of shot numbers above the reference tag, each in its shot's colour
    ref = next(s for s in shots if s["index"] == ref_idx)
    top = ref["C_ref"][ref["C_ref"][:, 1].argmin()]
    label_pos = {s["index"]: top + np.array([0.0, -0.004 - 0.003 * k, 0.0])
                 for k, s in enumerate(shots)}

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

    def show(vis, sel, clouds=True):
        for i in idxs:
            vis.show_geometry(f"cloud {i:03d}", clouds and i in sel)
            for n in names[i]:
                vis.show_geometry(n, i in sel)
        labels(vis, sel)

    def step(vis, d):
        state["k"] = (state["k"] + d) % len(idxs)
        show(vis, {idxs[state["k"]]})
        print(f"[VIEW] solo {idxs[state['k']]:03d}")

    def ref_plus(vis):
        state["k"] = (state["k"] + 1) % len(idxs)
        sel = {ref_idx, idxs[state["k"]]}
        show(vis, sel)
        print(f"[VIEW] {sorted(sel)}")

    actions = [("All", lambda v: show(v, set(idxs))),
               ("Corners only", lambda v: show(v, set(idxs), clouds=False)),
               ("Ref + next", ref_plus),
               ("Solo next", lambda v: step(v, +1)),
               ("Solo prev", lambda v: step(v, -1))]
    print("[VIEW] tick individual shots in the scene list, or use the action buttons; corner 0 is "
          "the larger marker" + ("; cubes = depth-plane corners" if args.compare_plane else ""))
    o3d.visualization.draw(items, title=f"tag {tag_id} masked stitch", show_ui=True,
                           point_size=args.point_size, lookat=center.astype(np.float32),
                           eye=np.zeros(3, np.float32), up=np.array([0, -1, 0], np.float32),
                           show_skybox=False, actions=actions,
                           on_init=lambda vis: labels(vis, set(idxs)))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Single-tag stitching test on masked, averaged clouds",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture", help="capture folder (sweep_capture -> average_confidence -> sam3_batch_masks)")
    ap.add_argument("--tag-id", type=int, default=16)
    ap.add_argument("--ref-index", type=int, default=None,
                    help="reference shot (default: first shot where the tag is usable)")
    ap.add_argument("--indices", default=None, help="manifest indices to consider, e.g. 0-8,32-35")
    ap.add_argument("--corners", choices=["pnp", "plane"], default="pnp")
    ap.add_argument("--fit", choices=["rigid", "translation"], default="rigid")
    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--tag-size-mm", type=float, default=24.0)
    ap.add_argument("--max-obliquity", type=float, default=55.0,
                    help="deg; skip shots where the tag is seen more edge-on than this")
    ap.add_argument("--w-min", type=float, default=0.2, help="drop points with weight below this")
    ap.add_argument("--no-mask", action="store_true", help="use the whole cleaned cloud, not the SAM3 mask")
    ap.add_argument("--mask-erode", type=int, default=3, help="px (colour) to erode the SAM3 mask")
    ap.add_argument("--palette", action="store_true", help="colour each shot instead of true RGB")
    ap.add_argument("--view-voxel", type=float, default=0.0, help="m; display downsample (0 = off)")
    ap.add_argument("--point-size", type=int, default=2)
    ap.add_argument("--corner-size-mm", type=float, default=1.5, help="corner marker size")
    ap.add_argument("--no-labels", action="store_true", help="no shot-number labels in the viewer")
    ap.add_argument("--compare-plane", action="store_true",
                    help="also draw each shot's depth-plane corners (cubes, same colour, same "
                         "transform) -- shows the depth bias even when PnP corners coincide")
    ap.add_argument("--save", action="store_true", help="write aligned clouds + transforms")
    ap.add_argument("--out", default=None, help="output sub-folder (default single_tag<ID>)")
    ap.add_argument("--no-view", action="store_true")
    # tag_stitch.measure_tags settings (plane mode), kept at tag_stitch.py defaults
    ap.set_defaults(tag_ids=set(), tag_erode=3, tag_min_pts=30, plane_inlier_mm=1.5,
                    ransac_iters=200, corner_win=6.0, tag_side_tol=0.10)
    args = ap.parse_args()
    args.mask = not args.no_mask

    capture = Path(args.capture)
    if not capture.is_dir():
        sys.exit(f"capture folder not found: {capture}")
    cam = ts.load_camera(capture)
    rows = ts.load_manifest(capture, ts.parse_indices(args.indices) if args.indices else None)
    detector = Detector(families=args.tag_family, quad_decimate=1.0, refine_edges=True)
    rng = np.random.default_rng(0)
    mode = args.corners

    # ---- 1 + 2: tag corners + masked cloud per shot ---------------------
    shots = []
    for r in rows:
        color = cv2.imread(str(capture / r["color"]))
        if color is None:
            sys.exit(f"unreadable colour image {capture / r['color']}")
        P, C, w, uv, inb = ts.shot_cloud(capture, r, cam, args, color)
        tags = ts.measure_tags(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), P, w, uv, inb,
                               cam, args, detector, rng)        # tags on the UNMASKED data
        rec = tags.get(args.tag_id)
        if rec is None:
            print(f"[SKIP] {r['index']:03d}: tag {args.tag_id} not detected")
            continue
        if not ts.tag_usable(rec, mode):
            why = (f"seen {rec.get('obliquity_pnp_deg', float('nan')):.0f} deg edge-on "
                   f"> --max-obliquity {args.max_obliquity:g}") if mode == "pnp" else rec.get("error")
            print(f"[SKIP] {r['index']:03d}: tag {args.tag_id} unusable ({why})")
            continue
        n_all = len(P)
        if args.mask:
            m = ts.in_mask(ts.load_mask(capture, r, args), uv, inb)
            P, C, w = P[m], C[m], w[m]
        Cn = np.asarray(ts.tag_points(rec, mode))
        Cp = (np.asarray(rec["corners_plane"]) if args.compare_plane and mode == "pnp"
              and "corners_plane" in rec else None)
        shots.append(dict(index=r["index"], true_deg=r["true_deg"], stem=Path(r["stop_dir"]).name,
                          C=Cn, Cp=Cp, P=P, col=C, w=w))
        if mode == "pnp":     # PnP corners are an exact tag-size square by construction
            info = (f"obliquity {rec['obliquity_pnp_deg']:4.1f} deg  "
                    f"reprojection {rec['pnp_reproj_px']:.2f} px")
        else:
            sides = np.linalg.norm(Cn - np.roll(Cn, -1, 0), axis=1) * 1000
            info = f"sides {np.round(sides, 1)} mm (expected {args.tag_size_mm:g})"
        print(f"[TAG ] {r['index']:03d}  {info}  cloud {n_all:,} -> {len(P):,} pts")

    if len(shots) < 2:
        sys.exit(f"tag {args.tag_id} is usable in only {len(shots)} shot(s) -- try another --tag-id")
    ref = next((s for s in shots if s["index"] == args.ref_index), None) if args.ref_index is not None \
        else shots[0]
    if ref is None:
        sys.exit(f"tag {args.tag_id} is not usable in reference shot {args.ref_index}")

    # ---- 3: register each shot's 4 corners onto the reference's --------
    print(f"\n  shot   logged_rot   fit_rot   diff   corner_rms_mm   (corners = {mode}, fit = {args.fit})")
    if mode == "pnp" and args.fit == "rigid":
        print("  (pnp: corner_rms is 0 by construction -- judge by diff vs the logged turntable angle)")
    colours = shot_colors(len(shots))
    for k, s in enumerate(shots):
        s["T"] = fit_transform(s["C"], ref["C"], args.fit)
        s["C_ref"] = ts.apply(s["T"], s["C"])
        s["Cp_ref"] = ts.apply(s["T"], s["Cp"]) if s["Cp"] is not None else None
        rms = float(np.sqrt(np.mean(np.sum((s["C_ref"] - ref["C"]) ** 2, 1))) * 1000)
        rot = ts.rot_deg(s["T"][:3, :3])
        logged = abs(((s["true_deg"] - ref["true_deg"]) + 180) % 360 - 180)
        s.update(rms_mm=rms, fit_rot_deg=rot, logged_rot_deg=logged, pal=colours[k])
        print(f"   {s['index']:03d}   {logged:8.2f}   {rot:8.2f}  {rot - logged:+6.2f}   {rms:9.3f}")

    spread_report(shots, "C_ref", f"{mode} corners")
    if args.compare_plane:
        spread_report(shots, "Cp_ref", "depth-plane corners (same transforms)")
    if mode == "pnp" and args.fit == "rigid" and not args.compare_plane:
        print("  (pnp + rigid: the fitted corners coincide by construction -- use --compare-plane "
              "or --fit translation to see real variation)")

    # ---- 4: move the clouds ---------------------------------------------
    for s in shots:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ts.apply(s["T"], s["P"])))
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile(s["pal"], (len(s["P"]), 1)) if args.palette else np.clip(s["col"], 0, 1))
        if args.view_voxel > 0:
            pcd = pcd.voxel_down_sample(args.view_voxel)
        s["cloud"] = pcd

    if args.save:
        out = capture / (args.out or f"single_tag{args.tag_id}")
        out.mkdir(parents=True, exist_ok=True)
        Ps, Cs, Ws = [], [], []
        for s in shots:
            o3d.io.write_point_cloud(str(out / f"aligned_{s['stem']}.ply"), s["cloud"])
            Ps.append(ts.apply(s["T"], s["P"])), Cs.append(s["col"]), Ws.append(s["w"])
        Pm, Cm, _, _ = ts.voxel_weighted(np.vstack(Ps), np.vstack(Cs), np.concatenate(Ws), 0.001)
        merged = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm))
        merged.colors = o3d.utility.Vector3dVector(np.clip(Cm, 0, 1))
        o3d.io.write_point_cloud(str(out / "merged.ply"), merged)
        (out / "transforms.json").write_text(json.dumps({
            "tag_id": args.tag_id, "corners": mode, "fit": args.fit, "reference_index": ref["index"],
            "convention": "X_ref = T_shot_to_ref @ X_shot (metres, colour-camera frames)",
            "shots": [dict(index=s["index"], true_deg=s["true_deg"], T_shot_to_ref=s["T"].tolist(),
                           fit_rot_deg=s["fit_rot_deg"], logged_rot_deg=s["logged_rot_deg"],
                           corner_rms_mm=s["rms_mm"]) for s in shots]}, indent=2))
        print(f"\n[OUT ] {len(shots)} aligned clouds + merged.ply ({len(Pm):,} pts) -> {out}")

    if not args.no_view:
        viewer(shots, ref["index"], ref["C"].mean(0), args.tag_id, args)


if __name__ == "__main__":
    main()
