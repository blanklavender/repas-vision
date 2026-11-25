#!/usr/bin/env python3
"""
tsdf_mesh_2dgs.py -- TSDF mesh from a 2DGS run's rendered median depth and colour, in metres.

gsplat has no mesh export, so this ports the defaults of the original 2DGS "extract_mesh_bounded":
    depth_trunc = 2 * radius (radius: closest camera to the centre of the camera path)
    voxel       = depth_trunc / 1024
    sdf_trunc   = 5 * voxel
    training views only; depth outside the object mask dropped; 50 largest clusters kept
Also dropped here: pixels with alpha <= 0.5, because the median depth is not defined on them.

It reads what `diag_depth.py export` wrote and needs only Open3D, so it runs outside the gsplat env:
    ~/.venvs/repas_open3d/bin/python tsdf_mesh_2dgs.py --run_dir <2dgs run> --data_dir <colmap> \
        --tof_cache <3dgs run>/diag/tof_cache

Pixel grids: gsplat's pixel i is K coordinate i + 0.5, Open3D's and the ToF's is i. So integration
uses cx - 0.5, cy - 0.5 to keep the mesh on the splats' geometry, and the ToF check casts rays with K.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

REGIONS = {"plant": 1, "pot_no_tags": 2, "tags": 3}
PANEL_VIEWS = ["000", "024", "048", "064"]


def read_png(path):
    return np.asarray(o3d.io.read_image(str(path)))


def error_stats(e):
    a = np.abs(e)
    return dict(n=int(len(e)), median_abs=float(np.median(a)), bias_mean=float(np.mean(e)),
                p90_abs=float(np.percentile(a, 90)), pct_3mm=float((a <= 3).mean() * 100),
                pct_5mm=float((a <= 5).mean() * 100), pct_10mm=float((a <= 10).mean() * 100))


def crop(im, mask, pad=8):
    y, x = np.nonzero(mask)
    return im[max(y.min() - pad, 0):y.max() + pad, max(x.min() - pad, 0):x.max() + pad]


def integrate(maps, cams, data_dir):
    K = np.array(cams["K"])
    w, h = cams["wh"]
    # gsplat pixel i = K coordinate i + 0.5; Open3D pixel i = K coordinate i.
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2] - 0.5, K[1, 2] - 0.5)
    train = [v for v in cams["views"] if v["split"] == "train"]
    centres = np.array([np.array(v["c2w_metres"])[:3, 3] for v in cams["views"]])
    radius = float(np.linalg.norm(centres - centres.mean(0), axis=1).min())
    depth_trunc = 2.0 * radius
    voxel = depth_trunc / 1024
    sdf_trunc = 5.0 * voxel
    print(f"[TSDF] radius {radius * 1000:.1f} mm, depth_trunc {depth_trunc * 1000:.1f} mm, "
          f"voxel {voxel * 1000:.3f} mm, sdf_trunc {sdf_trunc * 1000:.3f} mm, {len(train)} training views")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=sdf_trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for v in train:
        d = np.load(maps / f"{v['stem']}.npz")
        object_mask = read_png(data_dir / "images" / f"color_{v['stem']}.png").sum(-1) > 0
        depth = d["median_depth_mm"].astype(np.float32).copy()
        depth[~object_mask | (d["alpha"] <= 0.5)] = 0
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.io.read_image(str(maps / f"{v['stem']}_rgb.png")), o3d.geometry.Image(depth),
            depth_scale=1000.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(np.array(v["c2w_metres"])))
    return volume.extract_triangle_mesh(), dict(radius_m=radius, depth_trunc_m=depth_trunc, voxel_m=voxel,
                                                sdf_trunc_m=sdf_trunc, n_views=len(train))


def keep_largest(mesh, n=50):
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    keep = set(np.argsort(counts)[::-1][:n].tolist())
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_triangles_by_mask(~np.isin(labels, list(keep)))
    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out, int(len(counts))


def check_views(mesh, cams, cache, data_dir, out):
    """Cast each view's ToF-grid rays at the mesh: holes per region, and depth disagreement with the ToF."""
    K = np.array(cams["K"])
    w, h = cams["wh"]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    dirs_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    pools, hits, rows = {}, {}, []
    for view in cams["views"]:
        stem, split = view["stem"], view["split"]
        c2w = np.array(view["c2w_metres"])
        rays = np.concatenate([np.broadcast_to(c2w[:3, 3], (h, w, 3)), dirs_cam @ c2w[:3, :3].T], -1).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        z_mm = hit["t_hit"].numpy() * 1000.0  # direction has z = 1 in the camera, so t is z-depth
        hit_ok = np.isfinite(z_mm)
        tof = np.load(cache / f"{stem}_depth_mm.npy").astype(np.float32)
        valid = np.load(cache / f"{stem}_valid.npy")
        region = np.load(cache / f"{stem}_region.npy")
        row = dict(stem=stem, split=split)
        for name, label in REGIONS.items():
            cand = (region == label) & valid
            pick = cand & hit_ok
            hits.setdefault((split, name), [0, 0])
            hits[(split, name)][0] += int(pick.sum())
            hits[(split, name)][1] += int(cand.sum())
            pools.setdefault((split, name), []).append(z_mm[pick] - tof[pick])
            row[f"{name}_mesh_coverage"] = float(pick.sum() / max(cand.sum(), 1))
        rows.append(row)

        if stem[:3] in PANEL_VIEWS:
            mask = region > 0
            normals = hit["primitive_normals"].numpy()
            unit_rays = rays[..., 3:] / np.linalg.norm(rays[..., 3:], axis=-1, keepdims=True)
            shade = np.abs((normals * unit_rays).sum(-1))
            mesh_vis = (np.nan_to_num(shade) * 255).astype(np.uint8)
            mesh_vis = np.repeat(mesh_vis[..., None], 3, -1)
            mesh_vis[~hit_ok] = 0
            miss = read_png(data_dir / "images" / f"color_{stem}.png").copy()
            miss[(region == 1) & ~hit_ok] = (255, 0, 255)  # plant pixel where the mesh has nothing
            panel = np.hstack([crop(read_png(data_dir / "images" / f"color_{stem}.png"), mask), crop(mesh_vis, mask), crop(miss, mask)])
            o3d.io.write_image(str(out / f"mesh_panel_{stem}.png"), o3d.geometry.Image(panel))
    report = {}
    for (split, name), v in pools.items():
        report.setdefault(split, {})[name] = dict(error_stats(np.concatenate(v)),
                                                  mesh_coverage=hits[(split, name)][0] / hits[(split, name)][1])
    return report, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--tof_cache", type=Path, required=True)
    args = ap.parse_args()
    maps = args.run_dir / "maps"
    out = args.run_dir / "mesh"
    out.mkdir(exist_ok=True)
    cams = json.loads((maps / "cameras.json").read_text())

    raw, params = integrate(maps, cams, args.data_dir)
    mesh, n_clusters = keep_largest(raw)
    o3d.io.write_triangle_mesh(str(out / "tsdf_mesh_metres.ply"), mesh)
    print(f"[TSDF] {len(raw.triangles)} triangles in {n_clusters} clusters; kept {len(mesh.triangles)} in 50 largest")
    report, rows = check_views(mesh, cams, args.tof_cache, args.data_dir, out)
    (out / "tsdf_report.json").write_text(json.dumps(dict(
        params=params, n_triangles_raw=len(raw.triangles), n_triangles=len(mesh.triangles), n_clusters=n_clusters,
        watertight=bool(mesh.is_watertight()), disagreement_with_tof=report, per_view=rows), indent=2))
    for split in ("test", "train"):
        for name, r in report[split].items():
            print(f"{split:5s} {name:12s} coverage {r['mesh_coverage'] * 100:5.1f}%  |e| med {r['median_abs']:6.2f}  bias {r['bias_mean']:7.2f}  "
                  f"p90 {r['p90_abs']:6.2f}  <=3 {r['pct_3mm']:5.1f}  <=5 {r['pct_5mm']:5.1f}  <=10 {r['pct_10mm']:5.1f}")
    print(f"[OUT ] {out}")


if __name__ == "__main__":
    main()
