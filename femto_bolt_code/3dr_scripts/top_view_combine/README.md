# top_view_combine

Side turntable sweep + one robot-arm (UR5e) top shot → one stitched plant cloud, viewed in 3D.

```
python main_pipeline.py ../../../captures/femto_bolt/flora_unit/plant_10-16_deg_5_speed_5
python main_pipeline.py <capture> --from clean_top --force          # redo from a step
python main_pipeline.py <capture> --only stitch --top-deg 355       # top shot taken at another turntable angle
python main_pipeline.py <capture> --args clean_side="--axis-radius 0.10" --args stitch="--snapshots"
```

| script | role |
|---|---|
| `main_pipeline.py` | runs the steps below as separate processes, logs to `top_view_combine/pipeline.log` |
| `register_side.py` | step 1: AprilTag poses of the side shots + turntable axis |
| `clean_side.py` | step 2: SAM3 "yellow cylinder" + "green plant", radial crop around the axis |
| `clean_top.py` | step 3: SAM3 "green plant" on the top shot, no radial crop |
| `stitch_view.py` | step 4: stitch side + top, write the clouds, 3D viewer |
| `downsample_view.py` | opt-in step 5: raw overlap of every shot, or `--downsample` to the significant points |
| `overlap_view.py` | opt-in step 6: every shot as its own layer: toggle, scrub, colour by shot / overlap; `--source cleaned|downsampled` |
| `tv_common.py` | shared: work-folder layout, SAM3, depth filters, transform-file parser, top camera |

All outputs go to `<capture>/top_view_combine/`. The capture's own files are only read.

## Input

```
<capture>/
  manifest.csv, camera_params.json            side sweep (sweep_capture.py)
  color_<stem>.png  1280x720 BGR              side colour, <stem> = 000_ang000 ...
  depth_<stem>.png  1280x720 uint16 mm        side depth, D2C-aligned to colour
  robot_rgbd/
    color.png (or color_bgr.npy)  1280x720    top colour
    depth_mm.npy (or depth_raw.npy/.png)      top depth, 640x576 RAW depth-sensor image (not aligned)
    camera_params.json                        depth intrinsics + depth scale
    camera_pose_transforms_<date>.txt         B_T_C1, B_T_C2, C1_T_C2, C2_T_C1 (4x4, metres)
```

## Flow

### Step 1 — `register_side.py` → `registration.json`
1. Detect tag36h11 AprilTags on every raw side colour image. Each tag's pose comes from IPPE-square PnP on its 4 corners (colour only; tags seen at more than 55° are dropped).
2. Adjacent shots: per shared tag `T = P_(i+1) · P_i⁻¹`. The pair transform is the chordal-L2 mean of those rotations plus the mean tag-centre translation.
3. Chain the pairs into shot 0's frame. On a full 360° turn, close the loop and spread the error evenly over the shots.
4. Turntable axis: the dominant rotation axis of all poses (SVD), pointing up, through a least-squares point.

**Output**: `T_shot_to_ref` per shot, the axis (direction + point), loop error, per-shot drift against the logged angle, and the tag radius (= pot radius, ≈ 46 mm here).

### Step 2 — `clean_side.py` → `side/`
For each side shot:
1. SAM3 union mask of "yellow cylinder" and "green plant".
2. **Radial crop**: back-project the depth and move the axis into this shot's frame. Keep only points within `--axis-radius` (default 150 mm) of the axis: wide enough for the leaves, still cutting the wall and floor.
3. Depth-gradient filter (Sobel > 20 mm/px → flying pixels).
4. Erode the mask by 2 px.
5. Occlusion filter: drop points more than 40 mm behind the nearest kept pixel within a 24 px window.
6. Back-project through the colour intrinsics, then statistical outlier removal (20 neighbours, 2σ).

**Output**: `mask/`, `rgb/`, `depth/`, `pointcloud/cloud_<stem>.ply` (that shot's colour frame, metres), `cleaning.json`.

### Step 3 — `clean_top.py` → `top/`
1. SAM3 mask, prompt "green plant" only, on the 1280x720 colour image.
2. Depth-gradient filter on the 640x576 depth image.
3. Back-project every valid depth pixel through the **depth** intrinsics and its rational distortion. Move it into the colour camera: `X_c = R_dc·X_d + t_dc` (depth_to_color from camera_params.json).
4. Project into the colour image (colour K + distortion).
5. Visibility z-buffer: drop a point when another projected point within 5 px is more than 15 mm nearer. Otherwise it would pick up the mask or colour of whatever hides it from the colour camera.
6. Keep points that land on the eroded "green plant" mask, take their colour, then statistical outlier removal.

**Output**: `mask_top.png`, `color_top.png`, `depth_top.png`, `cloud_top.ply` (**C2** colour-optical frame, metres), `cleaning_top.json`.

### Step 4 — `stitch_view.py` → `stitch/`
1. Side: `X_ref = T_shot_to_ref · X_shot` for every cleaned shot, then voxel-fuse (1 mm).
2. Top: parse the transform file. Use `C1_T_C2` as is (else `inv(C2_T_C1)`, else `inv(B_T_C1)·B_T_C2`). The base-frame consistency is printed. `X_ref = turn_back(top_deg) · C1_T_C2 · X_C2`, where ref = shot 0's colour camera = C1, and `turn_back` is the identity unless `--top-deg` is given.
3. Voxel-fuse side + top into `combined.ply`.
4. Fit check: for top points inside the side cloud's radius, report the distance to the nearest side point.
5. Plant frame: origin on the axis at the pot base, z up the axis.
6. Viewer. Toggles: side sweep, top shot (or magenta tint), axis, plant frame, both camera frustums. Buttons look from camera pose 1 or pose 2.

**Output**: `side_fused.ply`, `top_in_ref.ply`, `combined.ply`, `poses.json`, `plant_frame.json`, and `snap_views.png` with `--snapshots`.

### Step 5 (opt-in) — `downsample_view.py` → `downsample/`
Where the pipeline thins points: **only** `stitch_view.py`. It takes a 1 mm voxel mean, unweighted: every shot's noise is averaged in rather than rejected. The cleaning steps' SOR removes outliers but does not thin. Nothing else decimates (camera `decimation_factor` = 1).

```
python downsample_view.py <capture>                                 # RAW overlap: all 72 shots + top, untouched (~2.0 M points)
python downsample_view.py <capture> --color shot                    # raw, one hue per shot (top = white)
python downsample_view.py <capture> --color views                   # raw, coloured by how many shots see each spot
python downsample_view.py <capture> --downsample                    # significant points (~123 k, 6 %)
python downsample_view.py <capture> --downsample --color views
python downsample_view.py <capture> --downsample --method voxel     # plain voxel mean, for comparison
```

Significant points, a point-cloud version of the mesh clean-up (no triangles):

| stage | what | paper analogue |
|---|---|---|
| 1 incidence | per shot, PCA normal; drop points seen > 75° off-normal (mixed / flying ToF pixels); weight = cos(incidence) | triangles nearly perpendicular to the surface |
| 2 consensus | count the distinct shots (top included) within a 2 mm cell + its 26 neighbours; side points need ≥ 4 | suspended / redundant triangles; noise seen by 1–2 shots |
| 3 fuse | one point per 2 mm voxel, incidence-weighted mean position + colour | — |
| 4 plane fit | project onto the local PCA plane (6 mm) where the neighbourhood is not isotropic: thins the multi-shot shell; edges stay | boundary smoothing (`BoundarySmoothR`) |
| 5 components | DBSCAN at 5 mm; drop clusters < 100 points or > 30 mm off the main plant | removing small components; `TriEdgeLength` |

The top shot is a single view, so stage 2 does not filter it (it only counts as one vote for the side points). Stage 5 then removes its background fragments.

### Step 6 (opt-in) — `overlap_view.py`: shot-by-shot overlap inspector
Each of the 72 side shots and the top shot is a separate layer in the reference frame. It writes nothing, apart from `--screenshot`.

```
python overlap_view.py <capture>                                     # cleaned per-shot clouds, one hue per shot
python overlap_view.py <capture> --source downsampled                # per-shot clouds after the significance steps
python overlap_view.py <capture> --scrub --focus 18 --window 2 --color overlap
python overlap_view.py <capture> --shots 0-5,36 --no-top --cameras
```

- `--source cleaned` uses each shot's cloud from `clean_side.py` / `clean_top.py` as is.
- `--source downsampled` runs each shot through the incidence filter, consensus and its own weighted 2 mm voxel grid. It keeps only points within 5 mm of the final significant cloud, and adds the fused significant cloud as an extra layer.
- **Colour modes:**
  - `rgb`: camera colour.
  - `shot`: one hue per shot around the turn; the top shot is white.
  - `focus`: the focus shot is orange, the others grey.
  - `overlap`: only the focus shot is drawn. It is green where another shown shot agrees within `--overlap-mm`, red where only this shot sees the surface.
- **Panel:** focus slider, Prev / Next (or the ←/→ keys in the 3D view), Scrub (focus ± window only), window slider, point size, and a checkbox per shot with All / None / Invert. Layers: top shot, fused cloud, shot cameras, axis, plant frame. You can look from the focus camera, pose 1 or the top camera.
- **Info text:** the focus shot's point count, and its overlap with the previous shot, the next shot and the top shot (fraction within `--overlap-mm`, and the median distance there).
