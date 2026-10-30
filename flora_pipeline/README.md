# Flora pipeline: capture → clean → pick

This pipeline does one turntable sweep of a potted plant with the Orbbec Femto Bolt. It registers the shots using the AprilTags on the pot and fits the turntable axis. Then it cleans every shot with SAM3 text prompts, keeping only the points within a fixed radius of that axis, stitches the shots into one point cloud and opens a 3D viewer. Every point you pick there is written, in camera coordinates (mm), to a text file. That file is rewritten on every run.

```
 sweep_capture.py ──► tag_register.py ──► sam3_data_cleaning.py ──► tag_pose_pick.py ──► picked_points.txt
 (turntable + camera)  (tag poses + axis)  (SAM3 + axis crop)        (stitch + 3D picking)
            └──────────────── all controlled by run_pipeline.py + pipeline.toml ────────────────┘
```

| Step | Script | Does | Typical time |
|---|---|---|---|
| `capture` | `flora_turntable_control_scripts/sweep_capture.py` | homes the turntable, then takes one colour + aligned-depth shot per stop | ~4 min for 36 stops at 10° |
| `register` | `femto_bolt_code/3dr_scripts/tag_register.py` | AprilTag poses from the raw colour images, loop closure, turntable axis | ~10–30 s |
| `clean` | `femto_bolt_code/3dr_scripts/sam3_data_cleaning.py` | SAM3 mask from the prompts, crop to a cylinder around the turntable axis, gradient/occlusion filters, one cleaned cloud per shot | ~1.5 min (GPU) |
| `pick` | `femto_bolt_code/3dr_scripts/tag_pose_pick.py` | fused cloud from the registered poses, plant frame, 3D picking | ~10 s, then interactive |

Each step is its own script and process. Steps hand data to each other only through files in the capture folder, so you can run any step on its own, repeat it, or debug it. Code the steps share lives in the library `flora_pipeline/flora_lib/`.

---

## 1. Setup on a new computer

### Hardware
- **Orbbec Femto Bolt** on a USB 3 port.
- **Turntable** with its Arduino running the serial-enabled turntable firmware. That is Noah's sketch with the serial layer; it is not in this repository. `turntable.py` sends it `H`, `M<deg>`, `G<deg>`, `S<1-9>` and waits for `OK ANG:… SPD:… INCR:…`.
- **AprilTags** of family `tag36h11`, with a 24 mm black square, stuck around the pot. Neighbouring shots must share at least one tag seen within 55° of head-on.
- An **NVIDIA GPU** is strongly recommended for SAM3. It runs on CPU, but very slowly.

### Software
Use Python **3.11** (3.11 or newer is needed for `tomllib`; open3d 0.19 supports 3.11). One virtual environment holds every step. The commands below are for Windows; on Linux use `python3.11` and `source .venv/bin/activate`.

```bash
git clone https://github.com/blanklavender/repas-vision.git
cd repas-vision
py -3.11 -m venv femto_bolt_code/.venv
femto_bolt_code\.venv\Scripts\activate
python -m pip install --upgrade pip
```

1. **PyTorch with CUDA.** Install this first, so later installs don't pull a CPU-only build. The tested build is torch 2.11 for CUDA 12.8:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   Check it with `python -c "import torch; print(torch.cuda.is_available())"`, which should print `True`.
2. **Pipeline packages:**
   ```bash
   pip install -r flora_pipeline/requirements.txt
   ```
3. **Orbbec SDK (`pyorbbecsdk`, v2.x; tested with 2.0.15).** Follow https://github.com/orbbec/pyorbbecsdk for your OS. On Linux, also install the udev rules from that repository so the camera opens without root. Check it with `python -c "import pyorbbecsdk"`.
4. **SAM3.** Both `sam3/` and `sam3_weights/` are git-ignored, so on a new computer you create them yourself:
   ```bash
   git clone https://github.com/facebookresearch/sam3.git sam3
   pip install -e sam3
   ```
   Request access to the SAM3 checkpoint on Hugging Face (`facebook/sam3`), download `sam3.pt`, and put it at `repas-vision/sam3_weights/sam3.pt`. To keep it somewhere else, set `[clean]` `checkpoint = "path/to/sam3.pt"` in `pipeline.toml`.
5. **Turntable port.** Find the Arduino's port: Device Manager → Ports on Windows (e.g. `COM7`), or `/dev/ttyACM0` on Linux. On Linux, add yourself to the `dialout` group (`sudo usermod -aG dialout $USER`, then log out and in). Put the port in `pipeline.toml` → `[capture] port`. Close the Arduino IDE Serial Monitor before running, because only one program can hold the port.
6. **Check the setup.**
   ```bash
   cd flora_pipeline
   python run_pipeline.py --dry-run
   python ../flora_turntable_control_scripts/turntable.py COM7 --step 90
   ```
   - The first prints the four commands the pipeline would run, and moves nothing.
   - The second is the turntable smoke test: it homes the table and turns it in 90° steps.
   - Finally, run `python run_pipeline.py --latest --only pick` on an existing capture to try the viewer.

### Tune `pipeline.toml` for the new rig (once)
- `[clean] axis_radius`: how far from the turntable axis, in metres, a point may lie and still be kept (default 0.06). It must be larger than the pot radius, which `register` prints as the tags' distance from the axis, and large enough for the widest leaves. Because it follows the axis, it works at any camera height or tilt and needs no depth limits.
- `[clean] prompts`: what SAM3 should keep, e.g. `["yellow cylinder", "green plant"]`. Name the pot so the tags on it are kept.
- `[register] tag_size_mm`: the edge length of the black square of your printed tags.

---

## 2. Running it

Run everything from `flora_pipeline/` with the venv active.

```bash
python run_pipeline.py
```

This makes a new capture folder, `<captures_root>/<name>_<MM-DD>_deg_<step>_speed_<speed>`. The `_2`, `_3`, … suffix is added if that folder already exists, so a capture is never overwritten. Then:

1. **capture**: the turntable homes, the camera warms up, and one shot is taken per stop after waiting `settle` seconds. With `home_after = true` the table re-homes at the end, so the plant stands exactly as in the first shot.
2. **register**: detects the tags in every raw colour image, chains the shot poses, closes the loop and fits the turntable axis.
3. **clean**: loads SAM3 and cleans every shot, keeping only points within `axis_radius` of the axis.
4. **pick**: stitches the shots and opens the viewer. Ctrl + left click a point on the plant; it gets a red marker, its coordinates are printed in the terminal, and the points file is rewritten. Use Undo, Clear and Reset view as needed, and close the window when done. The runner then prints the points file.

Common variations:

```bash
# different plant name and sweep settings
python run_pipeline.py --name basil --step 10 --speed 5 --settle 3 --port COM7

# re-process an existing capture (skip the camera), e.g. after changing settings
python run_pipeline.py --capture-dir ../captures/femto_bolt/flora_unit/basil_10-06_deg_10_speed_5 --from register

# just pick points again on the newest capture (cleaning + stitch are cached, so it opens fast)
python run_pipeline.py --latest --only pick

# capture only, process later
python run_pipeline.py --to capture

# change any setting for one run
python run_pipeline.py --latest --from clean --set clean.axis_radius=0.07 --set clean.force=true
python run_pipeline.py --latest --only pick --set pick.at_deg=90,180
```

If a step fails, the runner stops and prints the command to continue from that step, for example `python run_pipeline.py --capture-dir "<capture>" --from clean`. Everything a run prints is also appended to `<capture>/pipeline.log`.

**Caching.** `register` skips a capture that `registration.json` says was already registered with the same settings. `clean` skips a capture that `cleaning.json` says was already cleaned with the same settings and the same registration. `pick` reuses its stitch while its settings, the registration and the cleaning run are unchanged. To force a step to redo its work, use `--set register.force=true`, `--set clean.force=true` or `--set pick.restitch=true`. A new registration always makes `clean` run again, because the crop depends on the axis.

### Running a step by hand
Run these from the repository root. Each script takes `--help`.

```bash
python flora_turntable_control_scripts/sweep_capture.py COM7 --step 10 --speed 5 --settle 3 --out <capture> --home-after --no-ply
python femto_bolt_code/3dr_scripts/tag_register.py <capture>
python femto_bolt_code/3dr_scripts/sam3_data_cleaning.py <capture>
python femto_bolt_code/3dr_scripts/tag_pose_pick.py <capture> --points-file picked_points.txt
```

---

## 3. The output: picked points

`[paths] points_file` defaults to `captures/femto_bolt/flora_unit/picked_points.txt`. A per-capture copy is in `<capture>/tag_pick/picks.csv`, which has more columns. The file is **emptied when the viewer opens** and **rewritten after every pick or undo**. It is written to a temporary file and then renamed into place, so a program polling it never reads half a file.

```
# Flora pipeline picked points (rewritten on every pick)
# capture  C:\...\captures\femto_bolt\flora_unit\basil_10-06_deg_10_speed_5
# frame    colour camera, turntable at 0.00 deg (first shot 000_ang000 = 0.00 deg); x right, y down, z forward; mm
# written  2026-10-06T11:48:09
# id  x_mm  y_mm  z_mm
P1 1.90 75.45 389.16
P2 3.60 62.55 399.52
```

**Frame.** Coordinates are in the Femto Bolt **colour camera** frame (OpenCV convention: x right, y down, z forward, origin at the colour camera's optical centre), in millimetres, with the turntable at the angle of the **first shot**. With `home_after = true` that is where the plant is after the run. If the plant stands at another angle, pass `--set pick.report_deg=<logged angle>` and the file is written for that angle instead. To parse it, skip lines starting with `#` and split the rest on spaces.

The terminal (and `picks.csv`) also shows, for each pick:
- **plant frame**: z is the turntable axis pointing up; the origin is on the axis at the pot base; x points horizontally toward the camera. From this you get the radius from the axis, the azimuth (0° = facing the camera), and the height above the pot base.
- the **turntable angle at which the point faces the camera**, and its camera coordinates at that angle;
- the **pixel** where the point appears in the first shot's colour image (`color_000_ang000.png`), useful for checking;
- with `pick.at_deg = "90,180"`, its camera coordinates at those turntable angles too.

---

## 4. Manual: every control option

### 4.1 `run_pipeline.py`

| Option | Meaning |
|---|---|
| `--config FILE` | settings file (default `flora_pipeline/pipeline.toml`) |
| `--from STEP` / `--to STEP` | first / last step to run (`capture`, `register`, `clean`, `pick`; default all four) |
| `--only STEP` | run just this step |
| `--capture-dir DIR` | work on this existing capture (needed when `capture` is not run) |
| `--latest` | work on the newest capture (folder with `manifest.csv`) under `captures_root` |
| `--name NAME` | prefix of the new capture folder (`paths.name`) |
| `--port`, `--step`, `--speed`, `--settle` | shortcuts for `capture.port/step/speed/settle` |
| `--points-file FILE` | where the picked points go (`paths.points_file`) |
| `--set SECTION.KEY=VALUE` | override any `pipeline.toml` key for this run; repeatable. Values are TOML: `10`, `0.5`, `true`, `"text"`, `["a", "b"]` |
| `--dry-run` | print the commands only |

### 4.2 `pipeline.toml`
Every key in `[capture]`, `[register]`, `[clean]` and `[pick]` is passed to that step's script as `--key-with-dashes`:
- `true` → the flag is given,
- `false` or `""` → it is left out,
- a list → several values.

So you can add any script option from the tables below to its section. Relative paths are relative to `pipeline.toml`.

| Section / key | Default | Meaning |
|---|---|---|
| `paths.captures_root` | `../captures/femto_bolt/flora_unit` | where new captures are created and `--latest` looks |
| `paths.points_file` | `<captures_root>/picked_points.txt` | the picked-points text file |
| `paths.name` | `plant` | capture folder prefix |
| `python.capture` / `register` / `clean` / `pick` | `""` | Python interpreter per step (`""` = the one running the runner); lets a step use a different venv |

### 4.3 `capture`: `sweep_capture.py PORT [options]`

| Option | Default | Meaning |
|---|---|---|
| `PORT` | | turntable serial port (`COM7`, `/dev/ttyACM0`) |
| `--step` | 10 | degrees between shots (whole degrees; the firmware truncates to motor steps, and the true angle is logged) |
| `--speed` | 5 | turntable speed 1–9 |
| `--settle` | 3.0 | seconds to wait after each move before the shot (lets leaves stop swinging; the camera queue is drained meanwhile) |
| `--out` | `captures/sweep_<time>` | capture folder (the runner sets it); refuses a folder that already has a `manifest.csv` |
| `--total` | 360 | degrees to sweep (number of shots = total // step) |
| `--warmup` | 3.0 | seconds of camera auto-exposure warm-up |
| `--home-after` | off | re-home the turntable after the sweep |
| `--no-ply` | off | skip the SDK point clouds (`cloud_*.ply`, not used downstream) |
| `--decimate` | 1 | SDK point-cloud decimation |
| `--units` | `mm` | SDK point-cloud units (`mm` or `m`) |
| `--ascii-ply` | off | ASCII instead of binary SDK clouds |
| `--no-preview` | off | no preview windows. With preview on, **q / ESC aborts** the sweep. |

It writes `color_<stem>.png`, `depth_<stem>.png` (16-bit, aligned to the colour image), `manifest.csv` (rewritten after every stop, so an aborted sweep stays usable), `camera_params.json` and `intrinsics.json`. `<stem>` is `<index>_ang<commanded deg>`, e.g. `012_ang120`.

### 4.4 `register`: `tag_register.py CAPTURE [options]`

It runs on the raw colour images, before any cleaning:
1. AprilTag poses by colour-only PnP;
2. pairwise registration of neighbouring shots;
3. chaining into the first shot's frame, with loop closure on a full turn;
4. turntable axis fit: the common rotation axis of all poses.

| Option | Default | Meaning |
|---|---|---|
| `--max-obliquity` | 55 | ignore tags seen more than this many degrees from head-on (oblique PnP is unreliable) |
| `--tag-size-mm` | 24 | black-square edge of the printed tags |
| `--tag-family` | `tag36h11` | AprilTag family |
| `--loop` | `auto` | close the loop last → first shot (`auto` = when the sweep covers 360°) |
| `--max-turn-error` | 2 | reject a shot whose registered rotation is more than this many degrees off its logged turntable turn (a bad tag pose); `clean` skips it, so it is never stitched. 0 = keep every shot |
| `--force` | off | register again even if `registration.json` is up to date |

**Rejected shots.** Each shot's registered rotation is compared with its own logged turntable angle, around the axis and in tilt. This catches a shot that a 10° sweep places at, say, 47° instead of 40°. The step drops the worst shot beyond `--max-turn-error` and chains the sweep again, with that shot's neighbours registered to each other directly (a 20° edge). It repeats until every remaining shot is within tolerance.

Re-chaining is needed because a bad tag pose doesn't cancel out exactly over its two edges, since each edge averages several tags. Without it, every later shot is bent by a few degrees as well. The axis is fitted from the kept shots only. Each dropped shot prints a `[DROP ]` line. A shot whose neighbours share no usable tag can't be bridged, so it is kept with a `[WARN]`. Shot 0 is the reference and is never dropped.

It writes `registration.json`, which holds:
- `T_shot_to_ref`, `turn_error_deg` and `rejected` for every shot;
- the edges and the loop error;
- the axis (direction + a point on it, in the first shot's colour-camera frame);
- the registered-vs-logged turn per shot;
- the tags' distance from the axis, which is the pot radius.

### 4.5 `clean`: `sam3_data_cleaning.py CAPTURE [options]`

It needs `registration.json`. Filters, in order:
1. SAM3 union mask of all prompts;
2. axis crop: keep depth whose 3D point lies within `--axis-radius` of the turntable axis, with the axis moved into each shot's own camera frame;
3. depth-gradient filter;
4. mask erosion;
5. occlusion filter;
6. back-projection;
7. statistical outlier removal.

| Option | Default | Meaning |
|---|---|---|
| `--prompts` | `"yellow cylinder" "green plant"` | SAM3 text prompts; the union of all detections is kept |
| `--confidence` | 0.5 | SAM3 detection threshold (lower finds more, including wrong things) |
| `--checkpoint` | `<repo>/sam3_weights/sam3.pt` | SAM3 weights |
| `--axis-radius` | 0.06 | keep points within this many metres of the turntable axis (must exceed the pot radius) |
| `--grad-max` | 20 | drop depth edges steeper than this many mm per pixel (flying pixels on silhouettes); 0 = off |
| `--hole-guard` | 1 | px of hole filling before the gradient test |
| `--occlusion-tol-mm`, `--occlusion-window-px` | 40, 24 | drop depth lying more than tol behind the nearest kept pixel within the window. This catches the wall seen past the pot edge, which lands on pot pixels because the depth and colour cameras sit apart. 0 = off |
| `--mask-erode` | 2 | px the SAM3 mask shrinks before it cuts the depth |
| `--sor-neighbors`, `--sor-std` | 20, 2.0 | statistical outlier removal; `--sor-std 0` = off |
| `--intrinsics-ini` | | Orbbec `CameraParam_*.ini`, only for old captures without `camera_params.json` |
| `--force` | off | clean again even if `cleaning.json` is up to date (same settings and registration) |

It writes `mask/`, `rgb/` (masked colour), `depth/` (cleaned aligned depth), `pointcloud/cloud_<stem>.ply` (cleaned cloud, colour-camera frame, metres) and `cleaning.json`. A shot whose mask comes out empty is listed there and left out of the stitch.

### 4.6 `pick`: `tag_pose_pick.py CAPTURE [options]`

It moves every cleaned shot into the first shot's frame with its pose from `registration.json`, fuses them, and puts the plant origin on the axis at the pot base.

| Option | Default | Meaning |
|---|---|---|
| `--points-file` | `<capture>/tag_pick/picked_points.txt` | picked camera coordinates (the runner passes `paths.points_file`) |
| `--report-deg` | first shot's angle | turntable angle the points file refers to |
| `--at-deg` | | also print coordinates at these turntable angles, e.g. `90,180` |
| `--voxel-mm` | 1.0 | voxel size of the fused cloud |
| `--base-percentile` | 0.5 | plant origin height: this percentile of the cloud's height along the axis |
| `--out` | `tag_pick` | output sub-folder of the capture |
| `--restitch` | off | ignore the cached stitch |
| `--no-view` | off | stitch only, no viewer |
| `--point-size`, `--marker-mm` | 3, 2.5 | drawn point size (px) and pick marker radius (mm) |
| `--pick-radius-px` | 6 | how far from the cursor a click may land on a point |
| `--pick-depth-mm` | 5 | depth band of the visible surface under the cursor (points behind it are never picked) |

**Viewer controls:**
- **Ctrl + left click**: pick a point.
- **Drag**: rotate. **Wheel**: zoom.
- **Undo / Clear**: remove picks; the points file is updated.
- **Reset view**: back to the camera's viewpoint.
- **Checkboxes**: show or hide the turntable axis (cyan line) and the plant frame axes (red x toward the camera, green y, blue z up).

It writes `tag_pick/poses.json` (the transforms of the fused shots, copied from the registration), `fused.ply`, `plant_frame.json`, `picks.csv`, and the points file.

**Quality numbers printed by `register` (and repeated by `pick`):**
- *loop error*: how far the chain misses itself after a full turn. It is spread over the shots; about 1° or less is good.
- *registered turn vs logged turntable angle*: the median and max difference per kept shot; under 1° is good.
- *shots within N deg of their logged turn*: how many shots survived `register.max_turn_error`. If many are dropped, the tags are being misread. Check the tag size, raise `register.max_obliquity`, or recapture with better light.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `No Orbbec camera found` | USB 3 port/cable; on Linux the udev rules; close Orbbec Viewer |
| `Could not open COM7` / `never saw READY` | wrong port, or the Arduino IDE Serial Monitor holds it |
| `[SKIP] …: empty SAM3 mask` for some shots | other wording in `clean.prompts`, or a lower `clean.confidence` |
| cloud has wall/background in it | lower `clean.axis_radius` (keep it above the pot radius that `register` prints) |
| leaves missing | raise `clean.axis_radius`, lower `clean.mask_erode`, or set `clean.occlusion_tol_mm` higher |
| pot wall cut away | `clean.axis_radius` is below the pot radius printed by `register`; raise it |
| ghost copies / doubled surfaces in the stitch | a shot with a bad pose got through: lower `register.max_turn_error` (e.g. 1.5) and look at the `[DROP ]` lines |
| `fewer than 2 shots within … deg of their logged turn` | `register.max_turn_error` is too tight for this capture: raise it, or set 0 to keep every shot |
| `no shared usable tag between shots …` | smaller `capture.step`, more tags around the pot, or `register.max_obliquity` up to ~65 |
| clean step slow | SAM3 is on the CPU: check that `torch.cuda.is_available()` prints `True` |
| viewer does not open (remote / headless session) | Open3D needs a desktop OpenGL session; run `pick` with `--set pick.no_view=true` to stitch only |

---

## 6. Code layout (for changing the pipeline)

```
flora_pipeline/
  run_pipeline.py        the controller (steps, config, logging)
  pipeline.toml          settings
  requirements.txt
  flora_lib/             shared library, imported by the step scripts
    dataset.py           capture-folder layout: file names, manifest, camera_params, registration.json, cleaning.json, points file
    geometry.py          transforms, camera rays / projection, voxel fusion
    tags.py              AprilTag detection, colour-only PnP tag poses, pairwise tag-pose registration
    registration.py      chaining + loop closure, turntable axis / plant frame
flora_turntable_control_scripts/sweep_capture.py, turntable.py
femto_bolt_code/3dr_scripts/tag_register.py, sam3_data_cleaning.py, tag_pose_pick.py
```

- The step scripts put `flora_pipeline/` on `sys.path` themselves, so nothing needs installing.
- To **add a step**: write a script that takes the capture folder as its first argument, reads and writes files through `flora_lib.dataset`, and exits non-zero on failure. Then add it to `STEPS` in `run_pipeline.py` and give it a section in `pipeline.toml`.

The research scripts in `3dr_scripts/` (`tag_stitch.py`, `tag_pose_register.py`, `tag_graph_register.py`, …) are not part of this pipeline. They need the old multi-frame captures (`stops/*/avg`, from `average_confidence.py`), which `sweep_capture.py` no longer records.
