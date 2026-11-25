#!/usr/bin/env python3
"""
surface_pick.py  --  everything about one point on the 2DGS surface, in camera coordinates.

Core of surface_pick_viewer.py (no GUI). Given a vertex of the RGB-only 2DGS TSDF mesh it returns:
    2dgs      the point, and the 2DGS normal (the rendered surfel normal of the shot that sees the
              point most head-on), plus the leaf ID from leaf_track.py
    pixel     that best shot, and the point's pixel in it: undistorted (= cameras.json K) and raw
    tof       the ToF depth on the same camera ray (median of a 5x5 window of cleaned ToF pixels,
              kept to the same SAM3 leaf as the centre pixel) -> a 3D point
    depth2dgs optional (--depth-run): the same camera ray hitting a second 2DGS run's mesh, e.g. one
              trained WITH ToF depth. Off by default; everything else works without it.
    gaps      distances between those points (all lie on one camera ray, so each gap is a
              depth difference; + = farther from the camera than the RGB-only 2DGS point)
    turntable the turntable command (whole degrees, what the firmware takes) at which the point
              faces the camera most directly while visible and in the image, and the point and
              normal in the camera frame there

Frames: "ref camera" = the colour camera with the turntable at the first shot (000_ang000,
0 deg), metres internally, mm in the output; x right, y down, z forward. This is the frame of
the 2DGS mesh, cameras.json, the ToF stitch and tag_pick/plant_frame.json (checked: identical).

Turntable: the firmware takes whole degrees and turns whole motor steps (8192 per turn, 0.044 deg),
truncating each move (flora_turntable_control_scripts/sweep_capture.py). The command reported here is
an absolute goto from home (turntable.Turntable.home() then goto(cmd)), which moves the shorter way:
cmd <= 180 turns +cmd, larger ones turn -(360 - cmd), each truncated to whole steps. (The sweep's
logged true_deg differ: there every 5 deg move lost 0.034 deg, which added up over 71 moves.)
Rotating the table to true angle t moves a plant point X to  O + Rot(axis, turn_sign * (t - ref)) (X - O)  in the camera frame
(flora_lib.registration.at_turn). Equivalently the camera sits at the inverse rotation of the
reference camera in the plant's frame; the self-test checks that against every shot's 2DGS camera.

    python surface_pick.py --selftest
    python surface_pick.py --vertex 12345
    python surface_pick.py --xyz -20.5 -110.2 520.3        # mm, ref camera; snaps to the mesh
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "flora_pipeline"))
from flora_lib import registration as reg  # noqa: E402
from flora_lib import tof_clean  # noqa: E402

CAP = REPO / "captures" / "femto_bolt"
DEFAULTS = dict(
    run=CAP / "plant_10-16_clean55/results/2dgs_7k",
    raw=CAP / "plant_10-16_deg_5_speed_5",
    tof_cache=CAP / "plant_10-16/results_10-16/plant_10-16_side/diag/tof_cache",
    clean_dir=CAP / "plant_10-16_tof_clean_obl55",
)
STEPS_PER_DEG = (2048 * 4) / 360.0   # turntable firmware, as sweep_capture.py
VIS_TOL_M = 0.003          # a ray hit within 3 mm of the point counts as "the point is visible"
SUSPECT_MM = 35.0          # gaps larger than this are flagged as probably a different surface


def rot(axis, rad):
    return reg.rotate_about(np.asarray(axis, float), rad)


class SurfaceQuery:
    def __init__(self, run, depth_run, raw, tof_cache, clean_dir, normal_radius_mm=3.0, tof_window=5):
        """depth_run: None (default) or a second 2DGS run whose mesh point on the same ray is reported."""
        self.run, self.raw = Path(run), Path(raw)
        self.depth_run = Path(depth_run) if depth_run else None
        self.tof_cache, self.normal_r, self.win = Path(tof_cache), normal_radius_mm / 1000.0, tof_window

        mesh = o3d.io.read_triangle_mesh(str(self.run / "mesh/tsdf_mesh_metres.ply"))
        mesh.compute_vertex_normals()
        self.mesh = mesh
        self.V = np.asarray(mesh.vertices)
        self.Nv = np.asarray(mesh.vertex_normals)
        self.tree = cKDTree(self.V)
        self.scene = self._scene(mesh)
        self.dscene = None
        if self.depth_run is not None:
            dpath = self.depth_run / "mesh/tsdf_mesh_metres.ply"
            if not dpath.is_file():
                sys.exit(f"--depth-run: no mesh at {dpath}")
            self.dscene = self._scene(o3d.io.read_triangle_mesh(str(dpath)))
        lt = self.run / "leaf_track/vertex_leaf.npy"
        self.leaf = np.load(lt) if lt.is_file() else np.zeros(len(self.V), np.int32)
        tj = self.run / "leaf_track/tracks.json"
        self.inst_leaf = ({s["stem"]: s["instance_leaf"] for s in json.loads(tj.read_text())["shots"]}
                          if tj.is_file() else {})

        cams = json.loads((self.run / "maps/cameras.json").read_text())
        self.K, self.wh = np.array(cams["K"], float), tuple(cams["wh"])
        self.views = [dict(stem=v["stem"], c2w=np.array(v["c2w_metres"], float)) for v in cams["views"]]
        c = json.loads((self.raw / "camera_params.json").read_text())
        self.dist = np.array(c["color_distortion"]["opencv_order"], float)
        self.undist = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, self.wh, cv2.CV_32FC1)
        self.clean = tof_clean.CleanRun(clean_dir, self.K, self.dist, self.wh)

        pf = json.loads((self.raw / "tag_pick/plant_frame.json").read_text())
        self.F = dict(axis=np.array(pf["axis"]), origin=np.array(pf["origin"]), R=np.array(pf["R"]),
                      turn_sign=float(pf["turn_sign"]), ref_deg=float(pf["reference_deg"]))
        rows = list(csv.DictReader(open(self.raw / "manifest.csv")))
        self.true_of_stem = {Path(r["color"]).stem.removeprefix("color_"): float(r["true_deg"]) for r in rows}

    @staticmethod
    def _scene(mesh):
        s = o3d.t.geometry.RaycastingScene()
        s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        return s

    # --- geometry helpers ---------------------------------------------------
    def camera_at(self, true_deg):
        """Ref-frame pose (c2w) of the camera when the turntable stands at true_deg."""
        R = rot(self.F["axis"], -self.F["turn_sign"] * np.radians(true_deg - self.F["ref_deg"]))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.F["origin"] - R @ self.F["origin"]
        return T

    def project(self, X, c2w):
        """-> (camera xyz, undistorted pixel, raw distorted pixel, in_image)."""
        T = np.linalg.inv(c2w)
        Xc = T[:3, :3] @ X + T[:3, 3]
        uv = self.K[:2, :2] @ (Xc[:2] / Xc[2]) + self.K[:2, 2]
        raw, _ = cv2.projectPoints(Xc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        w, h = self.wh
        inside = Xc[2] > 0 and 0 <= uv[0] < w and 0 <= uv[1] < h
        return Xc, uv, raw.reshape(2), bool(inside)

    def visible(self, X, C):
        """Is X the first surface on the ray from camera centre C (RGB-only mesh)?"""
        d = X - C
        L = np.linalg.norm(d)
        ray = o3d.core.Tensor(np.hstack([C, d / L])[None].astype(np.float32))
        t = float(self.scene.cast_rays(ray)["t_hit"].numpy()[0])
        return np.isfinite(t) and t >= L - VIS_TOL_M

    def ray_hit_near(self, scene, C, X, max_m=0.04):
        """Hit of the ray C->X on `scene` closest to X (not just the first one), or None."""
        d = X - C
        L = np.linalg.norm(d)
        ray = o3d.core.Tensor(np.hstack([C, d / L])[None].astype(np.float32))
        hits = scene.list_intersections(ray)
        t = hits["t_hit"].numpy()
        if not len(t):
            return None
        k = np.argmin(np.abs(t - L))
        return C + d / L * float(t[k]) if abs(t[k] - L) <= max_m else None

    def mesh_normal(self, i):
        nb = self.tree.query_ball_point(self.V[i], self.normal_r)
        n = self.Nv[nb].sum(0)
        return n / (np.linalg.norm(n) + 1e-12)

    # --- the query ------------------------------------------------------------
    def snap(self, xyz_m):
        return int(self.tree.query(np.asarray(xyz_m, float))[1])

    def query(self, i):
        X = self.V[i]
        n_mesh = self.mesh_normal(i)

        # shots that see the point, ranked by how head-on (|cos| between normal and view ray)
        seen = []
        for v in self.views:
            C = v["c2w"][:3, 3]
            Xc, uv, raw, inside = self.project(X, v["c2w"])
            if inside and self.visible(X, C):
                to_cam = (C - X) / np.linalg.norm(C - X)
                seen.append(dict(v, uv=uv, raw=raw, Xc=Xc, cos=float(n_mesh @ to_cam)))
        if not seen:
            return dict(vertex=i, error="no shot sees this point (hidden in every view)")
        best = max(seen, key=lambda s: abs(s["cos"]))
        side = np.sign(best["cos"]) or 1.0                 # orient toward the side the best shot sees
        n_mesh = n_mesh * side
        seen = sorted([s for s in seen if s["cos"] * side > 0], key=lambda s: -abs(s["cos"]))

        # 2DGS normal: rendered surfel normals around the pixel in the best shot
        n2, n2_src = self._render_normal(best), "2DGS rendered normal, " + best["stem"]
        if n2 is None:
            n2, n2_src = n_mesh, "mesh vertex normals (no rendered surfel of this surface at the pixel)"
        elif n2 @ n_mesh < np.cos(np.radians(60)):
            n2, n2_src = n_mesh, "mesh vertex normals (rendered normal disagreed by > 60 deg)"

        tof = self._tof_point(seen, X, int(self.leaf[i]))
        C = best["c2w"][:3, 3]
        Xd = None if self.dscene is None else self.ray_hit_near(self.dscene, C, X)
        t0 = float(np.linalg.norm(X - C))

        def along(P):
            return None if P is None else float(np.linalg.norm(P - C) - t0) * 1000.0

        gaps = dict(tof_minus_2dgs_mm=None if tof is None else float(np.linalg.norm(tof["X"] - tof["C"])
                                                                     - np.linalg.norm(X - tof["C"])) * 1000.0)
        if self.dscene is not None:
            gaps["depth2dgs_minus_2dgs_mm"] = along(Xd)
        if tof is not None and Xd is not None:
            gaps["tof_minus_depth2dgs_mm"] = (gaps["tof_minus_2dgs_mm"] - gaps["depth2dgs_minus_2dgs_mm"]
                                              if tof["stem"] == best["stem"] else
                                              float(np.linalg.norm(tof["X"] - Xd)) * 1000.0)

        big = [k for k, v in gaps.items() if v is not None and abs(v) > SUSPECT_MM]
        if big:
            gaps["suspect"] = f"|gap| > {SUSPECT_MM:.0f} mm in {', '.join(big)}: probably not the same surface"
        pc = reg.plant_coords(X, self.F)
        out = dict(
            vertex=int(i), leaf=int(self.leaf[i]),
            ref_camera=dict(point_mm=X * 1000, normal=n2, normal_source=n2_src,
                            mesh_normal_angle_deg=float(np.degrees(np.arccos(np.clip(n2 @ n_mesh, -1, 1))))),
            plant=dict(height_mm=pc["height"] * 1000, radius_mm=pc["radius"] * 1000, azimuth_deg=pc["azimuth_deg"]),
            best_shot=dict(stem=best["stem"], true_deg=self.true_of_stem.get(best["stem"]),
                           incidence_deg=float(np.degrees(np.arccos(min(1.0, abs(best["cos"]))))),
                           pixel_undistorted=best["uv"], pixel_raw=best["raw"], point_cam_mm=best["Xc"] * 1000,
                           n_shots_seeing=len(seen)),
            tof=None if tof is None else dict({k: v for k, v in tof.items() if k not in ("X", "C")},
                                              point_mm=tof["X"] * 1000),
            depth2dgs=None if Xd is None else dict(point_mm=Xd * 1000),
            has_depth_run=self.dscene is not None,
            gaps=gaps,
            turntable=self.best_turn(X, n2),
        )
        return out

    def _render_normal(self, shot):
        m = np.load(self.run / "maps" / f"{shot['stem']}.npz")
        u, v = int(round(shot["uv"][0])), int(round(shot["uv"][1]))
        r = self.win // 2
        sl = (slice(max(v - r, 0), v + r + 1), slice(max(u - r, 0), u + r + 1))
        nc, a = m["normal_cam"][sl].astype(np.float64).reshape(-1, 3), m["alpha"][sl].reshape(-1)
        same = np.abs(m["median_depth_mm"][sl].reshape(-1) - shot["Xc"][2] * 1000.0) < 3.0
        nc = nc[(a > 0.5) & same]
        if not len(nc):
            return None
        n = np.median(nc, 0)
        if n[2] > 0:                                      # rendered normals face the camera (z < 0)
            n = -n
        return shot["c2w"][:3, :3] @ (n / np.linalg.norm(n))

    def _tof_point(self, seen, X, leaf):
        """ToF depth on the camera ray through X, from the most head-on shot with cleaned ToF there.
        On a leaf vertex the ToF pixels must lie in that shot's SAM3 mask of the SAME tracked leaf."""
        r = self.win // 2
        for s in seen:
            stem = s["stem"]
            dpath = self.tof_cache / f"{stem}_depth_mm.npy"
            if not dpath.is_file() or stem in self.clean.dropped:
                continue
            u, v = int(round(s["uv"][0])), int(round(s["uv"][1]))
            sl = (slice(max(v - r, 0), v + r + 1), slice(max(u - r, 0), u + r + 1))
            d = np.load(dpath)[sl].astype(np.float64)
            ok = np.load(self.tof_cache / f"{stem}_valid.npy")[sl] & self.clean.undistorted_valid(stem)[sl] & (d > 0)
            lm = self.raw / "leaf_masks" / f"{stem}.npz"
            same_leaf = False
            if leaf > 0:
                if not lm.is_file() or stem not in self.inst_leaf:
                    continue
                lab = np.load(lm)["labels"]
                c = int(lab[v, u])
                il = self.inst_leaf[stem]
                if c == 0 or c > len(il) or il[c - 1] != leaf:
                    continue                                   # pixel is not this leaf in this shot
                ok &= lab[sl] == c
                same_leaf = True
            if ok.sum() < 5:
                continue
            z = float(np.median(d[ok])) / 1000.0
            ray = np.array([(s["uv"][0] - self.K[0, 2]) / self.K[0, 0], (s["uv"][1] - self.K[1, 2]) / self.K[1, 1], 1.0])
            c2w = s["c2w"]
            Xt = c2w[:3, :3] @ (ray * z) + c2w[:3, 3]
            return dict(X=Xt, C=c2w[:3, 3], stem=stem, n_px=int(ok.sum()), same_leaf_mask=same_leaf,
                        best_shot=stem == seen[0]["stem"],
                        spread_mm=float(np.percentile(d[ok], 90) - np.percentile(d[ok], 10)),
                        point_cam_mm=(np.linalg.inv(c2w)[:3, :3] @ Xt + np.linalg.inv(c2w)[:3, 3]) * 1000)
        return None

    @staticmethod
    def true_of_command(cmd):
        """True angle after home() + goto(cmd): one move the shorter way, truncated to whole steps."""
        delta = cmd if cmd <= 180 else cmd - 360
        steps = int(abs(delta) * STEPS_PER_DEG)
        return (np.sign(delta) * steps / STEPS_PER_DEG) % 360.0

    def best_turn(self, X, n, max_cmd=359):
        """Whole-degree turntable command that shows the point most head-on, visible and in the image."""
        rows = []
        for cmd in range(0, max_cmd + 1):
            true = self.true_of_command(cmd)
            c2w = self.camera_at(true)
            C = c2w[:3, 3]
            Xc, uv, raw, inside = self.project(X, c2w)
            cos = float(n @ (C - X) / np.linalg.norm(C - X))
            ok = inside and cos > 0 and self.visible(X, C)
            rows.append((cmd, true, cos, ok, Xc, uv, raw, c2w))
        good = [r for r in rows if r[3]]
        if not good:
            return dict(error="the point is never visible from the camera at any turntable angle")
        cmd, true, cos, _, Xc, uv, raw, c2w = max(good, key=lambda r: r[2])
        ok30 = [r[0] for r in good if r[2] >= np.cos(np.radians(30))]
        n_cam = np.linalg.inv(c2w)[:3, :3] @ n
        return dict(command_deg=int(cmd), true_deg=float(true),
                    incidence_deg=float(np.degrees(np.arccos(min(1.0, cos)))),
                    point_cam_mm=Xc * 1000, normal_cam=n_cam, pixel_undistorted=uv, pixel_raw=raw,
                    commands_within_30deg=_ranges(ok30), n_visible_commands=len(good))

    # --- check against the shots --------------------------------------------
    def selftest(self):
        err_t, err_r = [], []
        for v in self.views:
            T = self.camera_at(self.true_of_stem[v["stem"]])
            err_t.append(np.linalg.norm(T[:3, 3] - v["c2w"][:3, 3]) * 1000)
            R = T[:3, :3].T @ v["c2w"][:3, :3]
            err_r.append(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        return dict(n=len(err_t), centre_mm_median=float(np.median(err_t)), centre_mm_max=float(np.max(err_t)),
                    rot_deg_median=float(np.median(err_r)), rot_deg_max=float(np.max(err_r)))

    def shot_image(self, stem, uv, half=160):
        """Undistorted colour of a shot, cropped around uv, with a crosshair (RGB uint8)."""
        img = cv2.imread(str(self.raw / f"color_{stem}.png"), cv2.IMREAD_COLOR)
        img = cv2.remap(img, *self.undist, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        u, v = int(round(uv[0])), int(round(uv[1]))
        cv2.circle(img, (u, v), 9, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(img, (u - 22, v), (u - 11, v), (0, 0, 255), 2)
        cv2.line(img, (u + 11, v), (u + 22, v), (0, 0, 255), 2)
        cv2.line(img, (u, v - 22), (u, v - 11), (0, 0, 255), 2)
        cv2.line(img, (u, v + 11), (u, v + 22), (0, 0, 255), 2)
        h, w = img.shape[:2]
        x0, y0 = min(max(u - half, 0), w - 2 * half), min(max(v - half, 0), h - 2 * half)
        crop = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
        cv2.putText(crop, stem, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def _ranges(vals):
    """[3,4,5,9,10] -> '3-5, 9-10'"""
    if not vals:
        return ""
    out, s, p = [], vals[0], vals[0]
    for x in vals[1:] + [None]:
        if x is None or x != p + 1:
            out.append(f"{s}-{p}" if s != p else f"{s}")
            if x is not None:
                s = x
        if x is not None:
            p = x
    return ", ".join(out)


def fmt(v, nd=1):
    return "(" + ", ".join(f"{x:+.{nd}f}" for x in np.asarray(v, float)) + ")"


def report_lines(r):
    """Human-readable lines for one query result."""
    if "error" in r:
        return [f"vertex {r['vertex']}: {r['error']}"]
    rc, b, t, g = r["ref_camera"], r["best_shot"], r["turntable"], r["gaps"]
    L = [f"vertex {r['vertex']}   leaf {('L%d' % r['leaf']) if r['leaf'] > 0 else 'none'}   "
         f"height {r['plant']['height_mm']:.0f} mm above pot base, radius {r['plant']['radius_mm']:.0f} mm",
         "REF CAMERA FRAME (turntable 0 deg, shot 000), mm:",
         f"  2DGS point         {fmt(rc['point_mm'])}",
         f"  2DGS normal        {fmt(rc['normal'], 3)}   [{rc['normal_source']}; "
         f"{rc['mesh_normal_angle_deg']:.1f} deg from mesh normal]"]
    L.append(f"  ToF point          {fmt(r['tof']['point_mm'])}   [shot {r['tof']['stem']}"
             f"{'' if r['tof']['best_shot'] else ' (not the best shot)'}, {r['tof']['n_px']} px, "
             f"spread {r['tof']['spread_mm']:.1f} mm{', same tracked leaf' if r['tof']['same_leaf_mask'] else ''}]"
             if r["tof"] else "  ToF point          none (no cleaned ToF pixel of this leaf on the ray in any shot)")
    if r.get("has_depth_run"):
        L.append(f"  2DGS+ToF-depth pt  {fmt(r['depth2dgs']['point_mm'])}" if r["depth2dgs"] else
                 "  2DGS+ToF-depth pt  none (that mesh has no surface within 40 mm on this ray)")
    gl = []
    for k, name in (("tof_minus_2dgs_mm", "ToF - 2DGS"), ("depth2dgs_minus_2dgs_mm", "depth-2DGS - 2DGS"),
                    ("tof_minus_depth2dgs_mm", "ToF - depth-2DGS")):
        if isinstance(g.get(k), float):
            gl.append(f"{name} {g[k]:+.1f}")
    L.append("  GAPS along the ray (+ = farther from camera), mm:  " + (";  ".join(gl) or "-"))
    if g.get("suspect"):
        L.append("  !! " + g["suspect"])
    L += [f"BEST SHOT {b['stem']} (turntable {b['true_deg']:.1f} deg), {b['incidence_deg']:.0f} deg off head-on, "
          f"seen by {b['n_shots_seeing']} shots",
          f"  pixel undistorted {fmt(b['pixel_undistorted'])}   raw image {fmt(b['pixel_raw'])}"]
    if "error" in t:
        L.append(f"TURNTABLE: {t['error']}")
    else:
        L += [f"TURNTABLE: command {t['command_deg']} deg (true {t['true_deg']:.1f} deg), "
              f"{t['incidence_deg']:.0f} deg off head-on; within 30 deg at commands {t['commands_within_30deg'] or '-'}",
              f"  camera frame there: point {fmt(t['point_cam_mm'])} mm, normal {fmt(t['normal_cam'], 3)}",
              f"  pixel undistorted {fmt(t['pixel_undistorted'])}   raw image {fmt(t['pixel_raw'])}"]
    return L


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return np.round(o.astype(float), 4).tolist()
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def add_args(ap):
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=str(v))
    ap.add_argument("--depth-run", default=None,
                    help="optional second 2DGS run (e.g. trained with ToF depth): also report where the same "
                         "camera ray hits its mesh. Off by default.")
    ap.add_argument("--normal-radius-mm", type=float, default=3.0, help="mesh-normal averaging radius")
    ap.add_argument("--tof-window", type=int, default=5, help="ToF median window, px (odd)")


def from_args(a):
    return SurfaceQuery(a.run, a.depth_run, a.raw, a.tof_cache, a.clean_dir, a.normal_radius_mm, a.tof_window)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--vertex", type=int)
    g.add_argument("--xyz", type=float, nargs=3, help="mm, ref camera frame; snapped to the nearest vertex")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full record as JSON")
    a = ap.parse_args()
    q = from_args(a)
    if a.selftest:
        print(json.dumps(q.selftest(), indent=2))
        return
    i = a.vertex if a.vertex is not None else q.snap(np.array(a.xyz) / 1000.0)
    r = q.query(i)
    print(json.dumps(to_jsonable(r), indent=2) if a.json else "\n".join(report_lines(r)))


if __name__ == "__main__":
    main()
