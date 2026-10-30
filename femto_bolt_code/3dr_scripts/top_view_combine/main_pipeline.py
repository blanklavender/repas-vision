#!/usr/bin/env python3
"""
main_pipeline.py  --  side sweep + robot-arm top shot -> one stitched, viewable plant cloud.

    register    register_side.py   AprilTag poses of the side shots + turntable axis
    clean_side  clean_side.py      SAM3 "yellow cylinder" + "green plant", radial crop around the axis
    clean_top   clean_top.py       SAM3 "green plant" on the top shot (robot_rgbd/), no radial crop
    stitch      stitch_view.py     side shots by tag poses, top shot by C1_T_C2, fused + 3D viewer
    downsample  downsample_view.py (opt-in) raw overlap of every shot, or --downsample to the
                                   significant points (multi-view consensus, plane fit, components)
    overlap     overlap_view.py    (opt-in) every shot as its own layer: toggle, scrub, colour by
                                   shot / overlap; --downsample = per-shot downsampled clouds

Each step is its own script in this folder, run as a separate process; they talk only
through files in <capture>/top_view_combine/ (see tv_common.py), so any step can be run or
repeated alone. register / clean_side / clean_top skip themselves when their output is
current (--force redoes them).

    python main_pipeline.py ../../../captures/femto_bolt/flora_unit/plant_10-16_deg_5_speed_5
    python main_pipeline.py <capture> --from clean_top --force
    python main_pipeline.py <capture> --only stitch --top-deg 355 --voxel-mm 2
    python main_pipeline.py <capture> --args clean_top="--confidence 0.4" --args stitch="--snapshots"
    python main_pipeline.py <capture> --only downsample                  # raw overlap, no downsampling
    python main_pipeline.py <capture> --only downsample --downsample --args downsample="--color views"
    python main_pipeline.py <capture> --only overlap                     # shot-by-shot inspector, cleaned
    python main_pipeline.py <capture> --only overlap --downsample --args overlap="--scrub --color overlap"
    python main_pipeline.py <capture> --dry-run
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = {
    "register": HERE / "register_side.py",
    "clean_side": HERE / "clean_side.py",
    "clean_top": HERE / "clean_top.py",
    "stitch": HERE / "stitch_view.py",
    "downsample": HERE / "downsample_view.py",
    "overlap": HERE / "overlap_view.py",
}
ORDER = list(STEPS)
WORK_DIR = "top_view_combine"                 # = tv_common.WORK_DIR (not imported: no cv2 needed here)


def main():
    ap = argparse.ArgumentParser(description="Side sweep + top shot: register -> clean_side -> clean_top -> stitch",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("capture", help="capture folder (manifest.csv, color_/depth_<stem>.png, robot_rgbd/)")
    ap.add_argument("--from", dest="start", choices=ORDER, default=ORDER[0], help="first step to run")
    ap.add_argument("--to", dest="stop", choices=ORDER, default=None,
                    help="last step to run (default stitch; downsample and overlap are opt-in: --to / --only)")
    ap.add_argument("--only", choices=ORDER, help="run just this step")
    ap.add_argument("--work", default=WORK_DIR, help="work sub-folder of the capture")
    ap.add_argument("--force", action="store_true", help="redo register / clean steps even if current")
    ap.add_argument("--checkpoint", help="SAM3 weights for both clean steps")
    ap.add_argument("--transforms", help="pose-transform file (default: newest robot_rgbd/camera_pose_transforms*)")
    ap.add_argument("--top-deg", type=float, help="turntable angle during the top shot (default: first shot's)")
    ap.add_argument("--voxel-mm", type=float, help="fused cloud voxel size")
    ap.add_argument("--no-view", action="store_true", help="stitch / downsample without opening the viewer")
    ap.add_argument("--downsample", action="store_true",
                    help="downsample step: significant points (else raw overlap); overlap step: per-shot downsampled clouds")
    ap.add_argument("--args", action="append", default=[], metavar='STEP="--flag value ..."',
                    help="extra arguments for one step, repeatable")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the steps")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = ap.parse_args()

    capture = Path(args.capture).resolve()
    if not capture.is_dir():
        sys.exit(f"{capture} does not exist")
    stop = args.stop or (args.start if args.start in ("downsample", "overlap") else "stitch")
    steps = [args.only] if args.only else ORDER[ORDER.index(args.start):ORDER.index(stop) + 1]
    if not steps:
        sys.exit("--from comes after --to: nothing to run")

    extra = {s: [] for s in ORDER}
    for item in args.args:
        step, sep, text = item.partition("=")
        if not sep or step not in extra:
            sys.exit(f"--args {item!r}: expected STEP=\"...\" with STEP one of {ORDER}")
        extra[step] += shlex.split(text, posix=os.name != "nt")
    common = ["--work", args.work]
    force = ["--force"] if args.force else []
    sam = ["--checkpoint", args.checkpoint] if args.checkpoint else []
    scene = ((["--transforms", args.transforms] if args.transforms else [])
             + (["--top-deg", str(args.top_deg)] if args.top_deg is not None else [])
             + (["--voxel-mm", str(args.voxel_mm)] if args.voxel_mm is not None else [])
             + (["--no-view"] if args.no_view else []))
    commands = {
        "register": [*common, *force],
        "clean_side": [*common, *force, *sam],
        "clean_top": [*common, *force, *sam],
        "stitch": [*common, *scene],
        "downsample": [*common, *scene, *(["--downsample"] if args.downsample else [])],
        "overlap": [*common, *[a for a in scene if a != "--no-view"],
                    "--source", "downsampled" if args.downsample else "cleaned"],
    }
    commands = {s: [args.python, str(STEPS[s]), str(capture), *a, *extra[s]] for s, a in commands.items()}

    work = capture / args.work
    print(f"[PIPE] capture {capture}")
    print(f"[PIPE] work    {work}")
    print(f"[PIPE] steps   {' -> '.join(steps)}")
    if args.dry_run:
        for s in steps:
            print(f"\n[{s}]\n  " + subprocess.list2cmdline(commands[s]))
        return

    env = dict(os.environ, PYTHONUNBUFFERED="1")          # child output appears live
    work.mkdir(parents=True, exist_ok=True)
    log = open(work / "pipeline.log", "a", encoding="utf-8")
    log.write(f"\n===== {datetime.now().isoformat(timespec='seconds')}  steps {steps}\n")
    t_all = time.time()
    for s in steps:
        cmd = commands[s]
        print(f"\n{'=' * 78}\n[PIPE] {s}: {subprocess.list2cmdline(cmd)}\n{'=' * 78}", flush=True)
        log.write(f"\n----- {s}: {subprocess.list2cmdline(cmd)}\n")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
        log.flush()
        if code != 0:
            print(f"\n[FAIL] step '{s}' exited with code {code} after {time.time() - t0:.0f} s "
                  f"(log: {work / 'pipeline.log'}).\n[FAIL] after fixing it, continue with:\n"
                  f"       python {Path(__file__).name} \"{capture}\" --from {s}")
            sys.exit(code)
        print(f"[PIPE] {s} done in {time.time() - t0:.0f} s")
    log.close()
    print(f"\n[PIPE] finished in {time.time() - t_all:.0f} s -- outputs in {work}")


if __name__ == "__main__":
    main()
