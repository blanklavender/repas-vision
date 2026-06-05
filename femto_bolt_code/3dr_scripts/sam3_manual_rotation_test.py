import glob
import json
import os
import re

import numpy as np
import open3d as o3d

INPUT_DIR = "../captures/manual_rotation_test/"

PCD_SUBDIR = "pointcloud"
OUT_SUBDIR = "aligned"
AXIS_JSON = "axis_refined.json"
PLY_GLOB = "cloud_*.ply"

ANGLE_SOURCE = "measured"
REF_ANGLE_DEG = 0
TAG_TO_AXIS_OVERRIDE_M = None

LOAD_ANGLES = None
VISIBLE_ANGLES = None

VOXEL_M = 0.002
REFINE_ICP = False
ICP_MAX_CORR_M = 0.008
ICP_ITERS = 50

VIEWER = "gui"
COLOR_BY_FRAME = True
MERGED_VISIBLE = True
MERGED_VOXEL_M = 0.0
POINT_SIZE = 3
AXIS_HALF_LEN_M = 0.15
FRAME_AXIS_SIZE_M = 0.02
SAVE_MERGED = True

PALETTE = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.70, 0.20], [0.15, 0.35, 0.95], [1.00, 0.75, 0.00],
    [0.85, 0.15, 0.85], [0.00, 0.80, 0.80], [1.00, 0.45, 0.00], [0.55, 0.85, 0.55],
])


def axis_angle_matrix(axis, theta):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def rotation_about(P, u_hat, deg):
    R = axis_angle_matrix(u_hat, np.deg2rad(deg))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = P - R @ P
    return T


MERGED_NAME = "merged_true_colour"


def gui_viewer(clouds, shot_frames, statics, visible, merged):
    items = [dict(name=MERGED_NAME, geometry=merged, group="merged",
                  is_visible=MERGED_VISIBLE)]
    for a in sorted(clouds):
        items.append(dict(name=f"cloud_{a:03d}", geometry=clouds[a],
                          group="clouds", is_visible=a in visible))
        items.append(dict(name=f"frame_{a:03d}", geometry=shot_frames[a],
                          group="shot frames", is_visible=a in visible))
    for name, geom in statics:
        items.append(dict(name=name, geometry=geom, group="axis", is_visible=True))

    order = sorted(clouds)
    focus = {"i": 0}

    def toggle(vis, angles, on):
        for a in angles:
            vis.show_geometry(f"cloud_{a:03d}", on)
            vis.show_geometry(f"frame_{a:03d}", on)

    def merged_only(vis):
        toggle(vis, order, False)
        vis.show_geometry(MERGED_NAME, True)
        print("[VIEW] merged, true colour")

    def show_all(vis):
        vis.show_geometry(MERGED_NAME, False)
        toggle(vis, order, True)
        print("[VIEW] all frames")

    def hide_all(vis):
        vis.show_geometry(MERGED_NAME, False)
        toggle(vis, order, False)

    def only_ref(vis):
        vis.show_geometry(MERGED_NAME, False)
        toggle(vis, order, False)
        toggle(vis, [REF_ANGLE_DEG] if REF_ANGLE_DEG in clouds else order[:1], True)

    def step(vis):
        vis.show_geometry(MERGED_NAME, False)
        toggle(vis, order, False)
        focus["i"] = (focus["i"] + 1) % len(order)
        pair = {order[focus["i"]]}
        if REF_ANGLE_DEG in clouds:
            pair.add(REF_ANGLE_DEG)
        toggle(vis, sorted(pair), True)
        print(f"[VIEW] showing {sorted(pair)}")

    o3d.visualization.draw(items, title="turntable alignment", show_ui=True,
                           point_size=POINT_SIZE,
                           actions=[("Merged", merged_only), ("All frames", show_all),
                                    ("All off", hide_all), ("Only ref", only_ref),
                                    ("Ref + next", step)])


def legacy_viewer(clouds, shot_frames, statics, visible, merged):
    order = sorted(clouds)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="turntable alignment")
    for _, g in statics:
        vis.add_geometry(g)
    shown = set()

    def apply(target):
        for a in order:
            if a in shown and a not in target:
                vis.remove_geometry(clouds[a], reset_bounding_box=False)
                vis.remove_geometry(shot_frames[a], reset_bounding_box=False)
            elif a not in shown and a in target:
                vis.add_geometry(clouds[a], reset_bounding_box=False)
                vis.add_geometry(shot_frames[a], reset_bounding_box=False)
        shown.clear()
        shown.update(target)
        print(f"[VIEW] showing {sorted(shown)}")

    merged_on = {"v": False}

    def set_merged(on):
        if on and not merged_on["v"]:
            vis.add_geometry(merged, reset_bounding_box=False)
        elif not on and merged_on["v"]:
            vis.remove_geometry(merged, reset_bounding_box=False)
        merged_on["v"] = on

    apply(set() if MERGED_VISIBLE else set(visible))
    set_merged(MERGED_VISIBLE)
    focus = {"i": 0}

    def pair():
        s = {order[focus["i"]]}
        if REF_ANGLE_DEG in clouds:
            s.add(REF_ANGLE_DEG)
        return s

    def k_all(_):
        set_merged(False)
        apply(set(order))
        return False

    def k_none(_):
        set_merged(False)
        apply(set())
        return False

    def k_next(_):
        set_merged(False)
        focus["i"] = (focus["i"] + 1) % len(order)
        apply(pair())
        return False

    def k_prev(_):
        set_merged(False)
        focus["i"] = (focus["i"] - 1) % len(order)
        apply(pair())
        return False

    def k_solo(_):
        set_merged(False)
        apply({order[focus["i"]]})
        return False

    def k_merged(_):
        apply(set())
        set_merged(True)
        return False

    vis.register_key_callback(ord("A"), k_all)
    vis.register_key_callback(ord("X"), k_none)
    vis.register_key_callback(ord("N"), k_next)
    vis.register_key_callback(ord("B"), k_prev)
    vis.register_key_callback(ord("S"), k_solo)
    vis.register_key_callback(ord("M"), k_merged)
    print("[KEYS] M=merged true colour  A=all frames  X=none  "
          "N=ref+next  B=ref+prev  S=solo current  Q=quit")
    vis.run()
    vis.destroy_window()


def main():
    pcd_dir = os.path.join(INPUT_DIR, PCD_SUBDIR)
    out_dir = os.path.join(INPUT_DIR, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    axis_path = os.path.join(out_dir, AXIS_JSON)
    if not os.path.exists(axis_path):
        raise SystemExit(f"missing {axis_path} -- run axis_debug.py first")
    with open(axis_path) as f:
        ax = json.load(f)
    P = np.asarray(ax["axis_point"], dtype=np.float64)
    u_hat = np.asarray(ax["axis_dir"], dtype=np.float64)
    u_hat /= np.linalg.norm(u_hat)
    angle_sign = int(ax["angle_sign"])
    undo_deg = ax.get("undo_angle_deg", {})
    depth_src = ax.get("axis_depth_source", "circle")
    lat_src = ax.get("axis_lateral_source", "circle")
    tag_off = ax.get("tag_to_axis_m")
    z_tag = ax.get("axis_z_tag")
    z_circ = ax.get("axis_z_circle")

    print(f"[AXIS] point={P.round(4)}  dir={u_hat.round(4)}  angle_sign={angle_sign:+d}")
    print(f"[AXIS] depth from '{depth_src}', lateral from '{lat_src}'  "
          f"-> axis camera z = {P[2]:.4f} m")
    if z_tag is not None and z_circ is not None:
        print(f"[AXIS] z(tag+{tag_off*100:.2f}cm)={z_tag:.4f} m  z(circle)={z_circ:.4f} m  "
              f"delta={(z_tag - z_circ)*1000:+.1f} mm")
    if depth_src == "tag" and z_tag is not None and abs(P[2] - z_tag) > 1e-6:
        print("[WARN] axis_point z does not match axis_z_tag -- stale axis_refined.json?")
    print(f"[AXIS] circle-fit RMS={ax['circle_rms_m']*1000:.2f} mm  "
          f"vertical spread={ax['vertical_spread_deg_median']:.2f} deg")
    print(f"[AXIS] measured angles available for {len(undo_deg)} frames")

    plys = glob.glob(os.path.join(pcd_dir, PLY_GLOB))
    if plys and os.path.getmtime(axis_path) < max(os.path.getmtime(q) for q in plys):
        print("[WARN] axis_refined.json is older than the point clouds -- re-run axis_debug.py")

    if TAG_TO_AXIS_OVERRIDE_M is not None:
        if z_tag is None or tag_off is None:
            raise SystemExit("axis_refined.json has no tag depth; cannot override the offset")
        want_z = (z_tag - tag_off) + TAG_TO_AXIS_OVERRIDE_M
        e1 = np.cross(u_hat, np.array([0.0, 0.0, 1.0]))
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u_hat, e1)
        P = P + ((want_z - P[2]) / e2[2]) * e2
        print(f"[AXIS] offset override {tag_off*100:.2f} -> {TAG_TO_AXIS_OVERRIDE_M*100:.2f} cm  "
              f"-> axis camera z = {P[2]:.4f} m  point={P.round(4)}")

    frames = []
    for path in sorted(glob.glob(os.path.join(pcd_dir, PLY_GLOB))):
        stem = os.path.splitext(os.path.basename(path))[0]
        suffix = stem.split("cloud_", 1)[-1]
        m = re.search(r"(\d{3})$", stem)
        if not m:
            print(f"[SKIP] no 3-digit angle: {stem}")
            continue
        angle = int(m.group(1)) % 360
        if LOAD_ANGLES is not None and angle not in LOAD_ANGLES:
            continue
        frames.append((angle, suffix, path))
    frames.sort()
    if not frames:
        raise SystemExit(f"no clouds matching {PLY_GLOB} in {pcd_dir} (check LOAD_ANGLES)")
    if LOAD_ANGLES is not None:
        print(f"[LOAD] restricted to {sorted(LOAD_ANGLES)} -> {len(frames)} cloud(s)")

    clouds, shot_frames, transforms, used_measured = {}, {}, {}, 0
    for angle, suffix, path in frames:
        if ANGLE_SOURCE == "measured" and suffix in undo_deg:
            deg = undo_deg[suffix]
            src = "meas"
            used_measured += 1
        else:
            deg = angle_sign * angle
            src = "file"

        pcd = o3d.io.read_point_cloud(path)
        if VOXEL_M > 0:
            pcd = pcd.voxel_down_sample(VOXEL_M)
        T = rotation_about(P, u_hat, deg)
        pcd.transform(T)
        clouds[angle] = pcd
        transforms[f"{angle:03d}"] = dict(suffix=suffix, undo_deg=float(deg),
                                          source=src, T=T.tolist())
        print(f"[XFRM] {angle:03d}  undo={deg:+8.3f} deg ({src})  pts={len(pcd.points):,}")

    print(f"[XFRM] {used_measured}/{len(frames)} frames used measured angles")

    have_ref = REF_ANGLE_DEG in clouds
    if not have_ref:
        print(f"[WARN] reference angle {REF_ANGLE_DEG:03d} not loaded -- "
              f"skipping residuals and using {sorted(clouds)[0]:03d} for viewer stepping")

    if REFINE_ICP and have_ref:
        target = clouds[REF_ANGLE_DEG]
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
        crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_ITERS)
        print("[ICP ] point-to-plane refinement against reference cloud")
        for angle in sorted(clouds):
            if angle == REF_ANGLE_DEG:
                continue
            res = o3d.pipelines.registration.registration_icp(
                clouds[angle], target, ICP_MAX_CORR_M, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(), crit)
            clouds[angle].transform(res.transformation)
            key = f"{angle:03d}"
            transforms[key]["T"] = (np.asarray(res.transformation)
                                    @ np.asarray(transforms[key]["T"])).tolist()
            transforms[key]["icp_fitness"] = float(res.fitness)
            transforms[key]["icp_rmse"] = float(res.inlier_rmse)
            print(f"  {angle:03d}  fitness={res.fitness:.3f}  "
                  f"inlier_rmse={res.inlier_rmse*1000:6.2f} mm")

    if have_ref:
        target = clouds[REF_ANGLE_DEG]
        print("[RESID] median NN distance to reference cloud")
        meds = []
        for angle in sorted(clouds):
            if angle == REF_ANGLE_DEG:
                continue
            d = np.asarray(clouds[angle].compute_point_cloud_distance(target))
            meds.append(np.median(d))
            print(f"  {angle:03d}  median={np.median(d)*1000:6.2f} mm  "
                  f"p90={np.percentile(d, 90)*1000:6.2f} mm")
        if meds:
            print(f"[RESID] overall median = {np.median(meds)*1000:.2f} mm")

    merged = o3d.geometry.PointCloud()
    for angle in sorted(clouds):
        merged += clouds[angle]
    if MERGED_VOXEL_M > 0:
        merged = merged.voxel_down_sample(MERGED_VOXEL_M)
    print(f"[MERGE] true-colour merge: {len(merged.points):,} pts")

    for i, angle in enumerate(sorted(clouds)):
        if COLOR_BY_FRAME:
            clouds[angle].paint_uniform_color(PALETTE[i % len(PALETTE)])
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=FRAME_AXIS_SIZE_M)
        cf.transform(np.asarray(transforms[f"{angle:03d}"]["T"]))
        shot_frames[angle] = cf

    a0, a1 = P - u_hat * AXIS_HALF_LEN_M, P + u_hat * AXIS_HALF_LEN_M
    axis_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.stack([a0, a1])),
        lines=o3d.utility.Vector2iVector([[0, 1]]))
    axis_ls.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])
    world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    world.translate(P)
    statics = [("rotation_axis", axis_ls), ("axis_origin", world)]

    if SAVE_MERGED:
        o3d.io.write_point_cloud(os.path.join(out_dir, "merged.ply"), merged)
        with open(os.path.join(out_dir, "transforms_applied.json"), "w") as f:
            json.dump(dict(axis_point=P.tolist(), axis_dir=u_hat.tolist(),
                           angle_source=ANGLE_SOURCE, refine_icp=REFINE_ICP,
                           load_angles=sorted(clouds), frames=transforms), f, indent=2)
        print(f"\n[SAVE] {out_dir}/merged.ply  ({len(merged.points):,} pts)")
        print(f"[SAVE] {out_dir}/transforms_applied.json")

    visible = set(clouds) if VISIBLE_ANGLES is None else (set(VISIBLE_ANGLES) & set(clouds))
    if MERGED_VISIBLE:
        visible = set()
    if VIEWER == "gui":
        try:
            gui_viewer(clouds, shot_frames, statics, visible, merged)
        except Exception as e:
            print(f"[VIEW] gui viewer unavailable ({e}); falling back to legacy")
            legacy_viewer(clouds, shot_frames, statics, visible, merged)
    elif VIEWER == "legacy":
        legacy_viewer(clouds, shot_frames, statics, visible, merged)


if __name__ == "__main__":
    main()