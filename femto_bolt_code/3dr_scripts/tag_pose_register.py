#!/usr/bin/env python3
"""
tag_pose_register.py  --  register a run of shots from whole-tag POSES, compared with
the corner-based Kabsch that tag_stitch.py uses.

For every adjacent pair (i, i+1) and every tag usable in both shots:
    tag pose in a shot   P = [R | c]   R from the tag's PnP corner frame, c = tag centre
    per-tag transform    T_i->i+1 = P_(i+1) @ P_i^-1
The per-tag transforms are then combined:
    pose      rotation = chordal L2 mean of the per-tag rotations (SVD projection of
              the weighted sum), translation = weighted mean of c_(i+1) - R c_i
    corners   tag_stitch.py's Kabsch over all shared corners (reference)
Weights (pose method): equal (default), or cos^2 of the tag's larger obliquity in the two
shots (--weights cos2). Equal was slightly better on the yellow-pot capture.

Both chains are composed into the first shot's frame, compared with the logged
turntable angles, and written as fused clouds you can open side by side.

    python tag_pose_register.py ../../captures/femto_bolt/flora_unit/yellow_pot_09-30_deg_10_speed_5
    python tag_pose_register.py <capture> --indices 0-9 --max-obliquity 40 --weights equal

Viewer (skip with --no-view): every selected shot's masked cloud under either method,
with its usable tags' corners (spheres, corner 0 larger) and edges in the shot's colour.
Buttons: All / None / Solo next / Solo prev / Add next, switch method, RGB vs per-shot
colour, clouds on/off, tags on/off; single shots can also be ticked in the scene list.

Depth calibration: --depth-correction <tag_stitch out>/depth_correction.json moves the depth
clouds onto the PnP tags (fitted over the whole capture by tag_stitch.py --refine-depth);
--refine-depth estimates it from the selected shots only. Poses are unaffected either way.

in   <capture>: camera_params.json, manifest.csv, color_*.png, stops/*/avg/, stops/*/mask.png
out  <capture>/<--out, default tag_pose_first10>/
        fused_pose.ply, fused_corners.ply   masked clouds of the selected shots in the first
                                            shot's frame (1 mm confidence-weighted voxels)
        transforms.json                     per-edge transforms + errors for both methods
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from pupil_apriltags import Detector

import tag_stitch as ts
from single_tag_masked import corner_markers, shot_colors
from view_tag_corners import edge_tubes


def tag_pose(rec):
    """4x4 tag->camera pose from the PnP corners (pupil order TR, TL, BL, BR)."""
    C = np.asarray(rec["corners_pnp"])
    x = ts.unit(C[0] - C[1])                          # TL -> TR
    y = ts.unit(C[1] - C[2])                          # BL -> TL
    z = ts.unit(np.cross(x, y))
    y = np.cross(z, x)
    return ts.make_T(np.column_stack([x, y, z]), C.mean(0))


def pose_average(Ts, w):
    """Chordal L2 rotation mean + weighted translation (see module docstring)."""
    M = sum(wi * T[:3, :3] for wi, T in zip(w, Ts))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return R


def main():
    ap = argparse.ArgumentParser(description="Tag-pose vs tag-corner registration of a run of shots",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("--indices", default="0-9")
    ap.add_argument("--max-obliquity", type=float, default=55.0)
    ap.add_argument("--weights", choices=["equal", "cos2"], default="equal")
    ap.add_argument("--tag-size-mm", type=float, default=24.0)
    ap.add_argument("--out", default="tag_pose_first10")
    ap.add_argument("--loop", choices=["auto", "yes", "no"], default="auto",
                    help="add the last->first edge and spread the loop error (auto: full turn)")
    ap.add_argument("--no-view", action="store_true", help="numbers + files only, no viewer")
    ap.add_argument("--no-labels", action="store_true", help="no shot-number labels in the viewer")
    ap.add_argument("--corner-size-mm", type=float, default=1.5, help="tag corner marker size")
    ap.add_argument("--depth-correction", default=None,
                    help="depth_correction.json from tag_stitch.py --refine-depth (whole capture; preferred)")
    ap.add_argument("--refine-depth", action="store_true",
                    help="estimate the depth->colour correction from the selected shots' own tags")
    ap.set_defaults(tag_ids=set(), tag_erode=3, tag_min_pts=30, plane_inlier_mm=1.5, ransac_iters=200,
                    corner_win=6.0, tag_side_tol=0.10, w_min=0.2, mask=True, mask_erode=3,
                    use_tag_centers=False, corner_depth="pnp", exclude_tags=set(), tag_outlier_mm=4.0,
                    min_corr=8)
    args = ap.parse_args()

    capture = Path(args.capture)
    cam = ts.load_camera(capture)
    rows = ts.load_manifest(capture, ts.parse_indices(args.indices))
    det = Detector(families="tag36h11", quad_decimate=1.0, refine_edges=True)
    rng = np.random.default_rng(0)

    def build(cam):
        shots = []
        for r in rows:
            col = cv2.imread(str(capture / r["color"]))
            P, C, w, uv, inb = ts.shot_cloud(capture, r, cam, args, col)
            tags = ts.measure_tags(cv2.cvtColor(col, cv2.COLOR_BGR2GRAY), P, w, uv, inb, cam, args, det, rng)
            m = ts.in_mask(ts.load_mask(capture, r, args), uv, inb)
            shots.append(dict(index=r["index"], true_deg=r["true_deg"], tags=tags, P=P[m], C=C[m], w=w[m]))
        return shots

    # depth->colour correction: poses come from PnP and do not change; the clouds move onto the tags
    if args.depth_correction:
        f = Path(args.depth_correction)
        if not f.is_file():
            sys.exit(f"--depth-correction: {f} not found")
        cam = ts.corrected_camera(cam, np.asarray(json.loads(f.read_text())["T"]))
        print(f"[CAL ] depth->colour correction loaded from {f}")
        shots = build(cam)
    elif args.refine_depth:
        shots = build(cam)
        T_corr, st = ts.estimate_depth_correction([s["tags"] for s in shots])
        if T_corr is None:
            sys.exit(f"--refine-depth: {st['error']}")
        print(f"[CAL ] from {st['n_used']} tags in {st['n_shots']} shots: tag mismatch "
              f"{st['before_mm']:.2f} -> {st['after_heldout_mm']:.2f} mm (held-out)")
        if st["after_heldout_mm"] < st["before_mm"]:
            cam = ts.corrected_camera(cam, T_corr)
            shots = build(cam)
        else:
            print("[CAL ] held-out check did not improve -- correction NOT applied")
    else:
        shots = build(cam)
    plates = [np.linalg.norm(np.asarray(r["corners_plane"]) - np.asarray(r["corners_pnp"]), axis=1).mean() * 1000
              for s in shots for r in s["tags"].values() if r.get("ok") and r.get("ok_pnp")]
    if plates:
        print(f"[CAL ] depth tag plates vs PnP tags: mean {np.mean(plates):.2f} mm over {len(plates)} tags\n")

    def register(a, b, logged):
        ids = sorted(i for i in set(a["tags"]) & set(b["tags"])
                     if a["tags"][i].get("ok_pnp") and b["tags"][i].get("ok_pnp"))
        e = dict(src=a["index"], tgt=b["index"], ids=ids, logged=logged, per_tag_deg=[],
                 T_pose=None, T_corners=None)
        if not ids:
            return e
        Ts, wts = [], []
        for i in ids:
            Ts.append(tag_pose(b["tags"][i]) @ np.linalg.inv(tag_pose(a["tags"][i])))
            ob = max(a["tags"][i]["obliquity_pnp_deg"], b["tags"][i]["obliquity_pnp_deg"])
            wts.append(np.cos(np.radians(ob)) ** 2 if args.weights == "cos2" else 1.0)
        wts = np.asarray(wts) / np.sum(wts)
        R = pose_average(Ts, wts)
        ca = np.array([a["tags"][i]["center_pnp"] for i in ids])
        cb = np.array([b["tags"][i]["center_pnp"] for i in ids])
        e["T_pose"] = ts.make_T(R, (wts[:, None] * (cb - ca @ R.T)).sum(0))
        e["T_corners"], _ = ts.tag_align(a["tags"], b["tags"], args)
        e["per_tag_deg"] = [ts.rot_deg(T[:3, :3]) for T in Ts]
        return e

    step_log = [b["true_deg"] - a["true_deg"] for a, b in zip(shots, shots[1:])]
    edges = [register(a, b, step_log[k]) for k, (a, b) in enumerate(zip(shots, shots[1:]))]
    closed = {"yes": True, "no": False, "auto": ts.is_closed_loop(rows)}[args.loop]
    loop = register(shots[-1], shots[0], shots[0]["true_deg"] + 360.0 - shots[-1]["true_deg"]) \
        if closed else None

    # consensus axis for the tilt numbers: mean rotation axis over both methods
    def axis(T):
        r = cv2.Rodrigues(T[:3, :3])[0].ravel()
        return r / np.linalg.norm(r)
    axm = ts.unit(np.mean([axis(e[m]) for e in edges for m in ("T_pose", "T_corners")
                           if e[m] is not None], 0))

    print(f"shots {shots[0]['index']}-{shots[-1]['index']}, weights {args.weights}, "
          f"max obliquity {args.max_obliquity:g} deg; logged step {np.mean(step_log):.3f} deg"
          f"{'; closed loop' if closed else ''}\n")
    print(" edge     tags        per-tag rotations (deg)       POSE  err  tilt   |  CORNERS  err  tilt")
    res = {"pose": [], "corners": []}
    for e in edges + ([loop] if loop else []):
        line = f" {e['src']:03d}->{e['tgt']:03d}  {str(e['ids']):<11} {str([round(x, 2) for x in e['per_tag_deg']]):<28}"
        for m, key in (("pose", "T_pose"), ("corners", "T_corners")):
            T = e[key]
            if T is None:
                line += "   ---  (too few tags)  |"
                continue
            a = ts.rot_deg(T[:3, :3]); err = a - e["logged"]
            tilt = float(np.degrees(np.arccos(np.clip(abs(axis(T) @ axm), -1, 1))))
            res[m].append((err, tilt))
            line += f"  {a:6.2f} {err:+5.2f} {tilt:5.1f}   |"
        print(line.rstrip("|"))
    print()
    for m, v in res.items():
        v = np.array(v)
        print(f" {m:<8} per-edge rotation error: mean {v[:, 0].mean():+.2f}, sd {v[:, 0].std():.2f}, "
              f"max |{np.abs(v[:, 0]).max():.2f}| deg   axis tilt median {np.median(v[:, 1]):.1f}, "
              f"max {v[:, 1].max():.1f} deg")

    # ---- chain, close the loop, fuse ----------------------------------------
    def signed_angle(T):
        """rotation about the consensus axis, signed, in degrees."""
        r = cv2.Rodrigues(T[:3, :3])[0].ravel()
        return float(np.degrees(np.linalg.norm(r)) * np.sign(r @ axm or 1.0))

    out = capture / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    n = len(shots)
    for m, key in (("pose", "T_pose"), ("corners", "T_corners")):
        bad = [f"{e['src']:03d}->{e['tgt']:03d}" for e in edges if e[key] is None]
        if bad:
            print(f"\n[WARN] {m}: no transform for {bad} -- chain broken, no fused cloud")
            continue
        poses = [np.eye(4)]
        for e in edges:
            poses.append(poses[-1] @ np.linalg.inv(e[key]))            # T_shot_to_first
        info = {}
        if loop is not None and loop[key] is not None:
            # shot 0 reached again through the loop edge; it should be the identity.
            # Spread that discrepancy evenly over the loop (shot k gets k/n of the fix).
            D = poses[-1] @ np.linalg.inv(loop[key])
            rD, tD = cv2.Rodrigues(D[:3, :3])[0].ravel(), D[:3, 3]
            info = dict(loop_error_deg=ts.rot_deg(D[:3, :3]), loop_error_mm=float(np.linalg.norm(tD) * 1000))
            poses = [ts.make_T(cv2.Rodrigues(-(k / n) * rD)[0], -(k / n) * tD) @ P
                     for k, P in enumerate(poses)]
        elif loop is not None:
            print(f"[WARN] {m}: loop edge failed -- chained only")
        sgn = np.sign(signed_angle(poses[1])) or 1.0
        drift = []
        for k, (sh, P) in enumerate(zip(shots, poses)):
            sh[f"T_{m}"] = P
            logged = sh["true_deg"] - shots[0]["true_deg"]
            drift.append(((sgn * signed_angle(P) - logged + 180.0) % 360.0) - 180.0)
        Ps = [ts.apply(sh[f"T_{m}"], sh["P"]) for sh in shots]
        Pm, Cm, _, _ = ts.voxel_weighted(np.vstack(Ps), np.vstack([sh["C"] for sh in shots]),
                                         np.concatenate([sh["w"] for sh in shots]), 0.001)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm))
        pcd.colors = o3d.utility.Vector3dVector(np.clip(Cm, 0, 1))
        o3d.io.write_point_cloud(str(out / f"fused_{m}.ply"), pcd)
        summary[m] = dict(pose_error_vs_turntable_deg=drift, **info)
        dr = np.abs(drift)
        print(f"\n {m:<8}" + (f" loop error {info['loop_error_deg']:.2f} deg / {info['loop_error_mm']:.1f} mm"
                             " spread over the loop;" if info else " chained, no loop closure;")
              + f" pose vs turntable: median {np.median(dr):.2f}, max {dr.max():.2f} deg")
        print("   per shot: " + " ".join(f"{d:+.2f}" for d in drift))
    (out / "transforms.json").write_text(json.dumps(dict(
        convention="X_tgt = T @ X_src, metres, colour-camera frames",
        edges=[dict(src=e["src"], tgt=e["tgt"], tags=e["ids"], logged_deg=e["logged"],
                    per_tag_deg=e["per_tag_deg"], T_pose=e["T_pose"].tolist(),
                    T_corners=None if e["T_corners"] is None else e["T_corners"].tolist())
               for e in edges + ([loop] if loop else []) if e["T_pose"] is not None],
        chains=summary), indent=2))
    print(f"\n[OUT ] {out}  ({', '.join(f'fused_{m}.ply' for m in summary)}, transforms.json)")

    if not args.no_view:
        viewer(shots, [m for m in ("pose", "corners") if m in summary], args)


# ---------------------------------------------------------------------------
# viewer: every shot under either method, toggle shots / method / colouring
# ---------------------------------------------------------------------------
def viewer(shots, methods, args):
    idxs = [s["index"] for s in shots]
    pal = shot_colors(len(shots))
    size = args.corner_size_mm / 1000.0
    items, names = [], {}
    for m in methods:
        for s, col in zip(shots, pal):
            i, T = s["index"], s[f"T_{m}"]
            P = ts.apply(T, s["P"])
            for mode, colors in (("rgb", np.clip(s["C"], 0, 1)), ("shot", np.tile(col, (len(P), 1)))):
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
                pcd.colors = o3d.utility.Vector3dVector(colors)
                n = f"{m} cloud {mode} {i:03d}"
                items.append(dict(name=n, geometry=pcd, group=f"{m} clouds", is_visible=False))
                names[(m, "cloud", mode, i)] = n
            mesh = o3d.geometry.TriangleMesh()
            for rec in s["tags"].values():                  # every usable tag's corners, this shot
                if rec.get("ok_pnp"):
                    mk, _ = corner_markers(ts.apply(T, np.asarray(rec["corners_pnp"])), col, size)
                    mesh += mk
                    mesh += edge_tubes(ts.apply(T, np.asarray(rec["corners_pnp"])), col, size * 0.25)
            n = f"{m} tags {i:03d}"
            items.append(dict(name=n, geometry=mesh, group=f"{m} tags", is_visible=False))
            names[(m, "tags", i)] = n

    state = dict(method=methods[0], colour="rgb", sel=set(idxs), clouds=True, tags=True, k=0)

    def refresh(vis):
        for m in methods:
            for i in idxs:
                on = m == state["method"] and i in state["sel"]
                for mode in ("rgb", "shot"):
                    vis.show_geometry(names[(m, "cloud", mode, i)],
                                      on and state["clouds"] and mode == state["colour"])
                vis.show_geometry(names[(m, "tags", i)], on and state["tags"])
        vis.clear_3d_labels()
        if not args.no_labels:
            for s, col in zip(shots, pal):
                if s["index"] in state["sel"]:
                    lab = vis.add_3d_label(label_pos[s["index"]], f"{s['index']:03d}")
                    try:
                        lab.color = o3d.visualization.gui.Color(*[float(x) for x in col])
                    except Exception:
                        pass
        print(f"[VIEW] {state['method']} | colour {state['colour']} | shots {sorted(state['sel'])}")

    first_tags = [np.asarray(r["corners_pnp"]) for r in shots[0]["tags"].values() if r.get("ok_pnp")]
    top = np.vstack(first_tags)[np.vstack(first_tags)[:, 1].argmin()] if first_tags else np.zeros(3)
    label_pos = {s["index"]: top + np.array([0.0, -0.006 - 0.004 * k, 0.0]) for k, s in enumerate(shots)}

    def setter(**kw):
        def f(vis):
            state.update(kw)
            refresh(vis)
        return f

    def solo(d):
        def f(vis):
            state["k"] = (state["k"] + d) % len(idxs)
            state["sel"] = {idxs[state["k"]]}
            refresh(vis)
        return f

    def add_next(vis):
        state["k"] = (state["k"] + 1) % len(idxs)
        state["sel"] = set(idxs[:state["k"] + 1])
        refresh(vis)

    def flip(key, a, b):
        def f(vis):
            state[key] = b if state[key] == a else a
            refresh(vis)
        return f

    actions = [("All shots", setter(sel=set(idxs))), ("No shots", setter(sel=set())),
               ("Solo next", solo(+1)), ("Solo prev", solo(-1)), ("Add next", add_next),
               ("Method pose/corners", flip("method", *methods) if len(methods) > 1 else setter()),
               ("RGB / per-shot colour", flip("colour", "rgb", "shot")),
               ("Clouds on/off", lambda v: (state.update(clouds=not state["clouds"]), refresh(v))),
               ("Tags on/off", lambda v: (state.update(tags=not state["tags"]), refresh(v)))]
    print("\n[VIEW] buttons: shots All/None/Solo/Add next, switch method, colouring, clouds/tags."
          " Individual shots: tick them in the scene list (groups '<method> clouds' / '<method> tags').")
    center = np.vstack([s["P"] for s in shots[:1]]).mean(0)
    o3d.visualization.draw(items, title="tag pose vs tag corner registration", show_ui=True,
                           point_size=2, lookat=center.astype(np.float32), eye=np.zeros(3, np.float32),
                           up=np.array([0, -1, 0], np.float32), show_skybox=False,
                           actions=actions, on_init=refresh)


if __name__ == "__main__":
    main()
