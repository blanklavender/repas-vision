#!/usr/bin/env python3
"""
run_pipeline.py  --  run the Flora pipeline end to end, or any part of it.

    capture   flora_turntable_control_scripts/sweep_capture.py   turntable sweep, one shot per stop
    clean     femto_bolt_code/3dr_scripts/sam3_data_cleaning.py  SAM3 prompts + depth thresholds
    pick      femto_bolt_code/3dr_scripts/tag_pose_pick.py       tag-pose stitch + 3D point picking

Every step is its own script, run as a separate process with the settings from
pipeline.toml, and talks to the next one only through files in the capture folder
(flora_lib/dataset.py), so any step can also be run, debugged or repeated on its own.
The picked points (camera coordinates, mm) end up in [paths] points_file, which is
emptied at the start of the pick step and rewritten after every pick.

    python run_pipeline.py                                   # capture -> clean -> pick
    python run_pipeline.py --name basil --step 10 --speed 5 --settle 3 --port COM7
    python run_pipeline.py --capture-dir ../captures/femto_bolt/flora_unit/plant_10-06_deg_10_speed_5 --from clean
    python run_pipeline.py --latest --only pick              # pick again on the newest capture
    python run_pipeline.py --latest --from clean --set clean.z_max=0.5 --set clean.force=true
    python run_pipeline.py --dry-run                         # print the commands only
"""

import argparse
import os
import subprocess
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STEPS = {
    "capture": REPO / "flora_turntable_control_scripts" / "sweep_capture.py",
    "clean": REPO / "femto_bolt_code" / "3dr_scripts" / "sam3_data_cleaning.py",
    "pick": REPO / "femto_bolt_code" / "3dr_scripts" / "tag_pose_pick.py",
}
ORDER = list(STEPS)


def parse_value(text):
    """TOML literal if it parses (10, 0.5, true, ["a", "b"]), else the raw string."""
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def load_config(path, overrides):
    cfg = tomllib.loads(Path(path).read_text())
    for item in overrides:
        key, sep, value = item.partition("=")
        section, dot, name = key.strip().partition(".")
        if not (sep and dot and name):
            sys.exit(f"--set {item!r}: expected section.key=value, e.g. clean.z_max=0.5")
        cfg.setdefault(section, {})[name] = parse_value(value.strip())
    return cfg


def flags(section, skip=()):
    """{key: value} -> ['--key-name', 'value', ...] (true: flag only, false / "": omitted)."""
    out = []
    for key, value in section.items():
        if key in skip or value is False or value == "" or value is None:
            continue
        out.append("--" + key.replace("_", "-"))
        if value is not True:
            out.extend(str(v) for v in value) if isinstance(value, list) else out.append(str(value))
    return out


def resolve(base, p):
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def new_capture_dir(root, cfg):
    c = cfg["capture"]
    base = root / f"{cfg['paths']['name']}_{datetime.now():%m-%d}_deg_{c['step']}_speed_{c['speed']}"
    path, k = base, 2
    while path.exists():
        path, k = base.with_name(f"{base.name}_{k}"), k + 1
    return path


def latest_capture(root):
    found = sorted(root.glob("*/manifest.csv"), key=lambda p: p.stat().st_mtime)
    if not found:
        sys.exit(f"--latest: no capture (folder with manifest.csv) under {root}")
    return found[-1].parent


def main():
    ap = argparse.ArgumentParser(description="Flora pipeline: capture -> clean -> pick",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--config", default=str(HERE / "pipeline.toml"), help="settings file")
    ap.add_argument("--from", dest="start", choices=ORDER, default="capture", help="first step to run")
    ap.add_argument("--to", dest="stop", choices=ORDER, default="pick", help="last step to run")
    ap.add_argument("--only", choices=ORDER, help="run just this step")
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--capture-dir", help="existing capture to work on (needed without the capture step)")
    where.add_argument("--latest", action="store_true", help="work on the newest capture under captures_root")
    ap.add_argument("--name", help="paths.name: prefix of the new capture folder")
    ap.add_argument("--port", help="capture.port")
    ap.add_argument("--step", type=int, help="capture.step (degrees)")
    ap.add_argument("--speed", type=int, help="capture.speed (1-9)")
    ap.add_argument("--settle", type=float, help="capture.settle (seconds)")
    ap.add_argument("--points-file", help="paths.points_file")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override any pipeline.toml setting, repeatable")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args = ap.parse_args()

    shortcuts = [f"paths.name={args.name!r}" if args.name else None,
                 f"capture.port={args.port!r}" if args.port else None,
                 f"capture.step={args.step}" if args.step is not None else None,
                 f"capture.speed={args.speed}" if args.speed is not None else None,
                 f"capture.settle={args.settle}" if args.settle is not None else None,
                 f"paths.points_file={args.points_file!r}" if args.points_file else None]
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path, [s.replace("'", '"') for s in shortcuts if s] + args.set)
    for sec in ("paths", "python", "capture", "clean", "pick"):
        cfg.setdefault(sec, {})
    base = cfg_path.parent
    root = resolve(base, cfg["paths"].get("captures_root", "."))
    points_file = resolve(base, cfg["paths"].get("points_file", "picked_points.txt"))

    steps = [args.only] if args.only else ORDER[ORDER.index(args.start):ORDER.index(args.stop) + 1]
    if not steps:
        sys.exit("--from comes after --to: nothing to run")
    if "capture" in steps:
        if args.capture_dir or args.latest:
            sys.exit("the capture step makes a NEW capture folder; drop --capture-dir/--latest or start --from clean")
        capture = new_capture_dir(root, cfg)
    elif args.capture_dir:
        capture = Path(args.capture_dir).resolve()
        if not capture.is_dir():
            sys.exit(f"--capture-dir {capture} does not exist")
    elif args.latest:
        capture = latest_capture(root)
    else:
        sys.exit(f"steps {steps} need an existing capture: pass --capture-dir <folder> or --latest")

    def interpreter(step):
        py = cfg["python"].get(step, "")
        return str(resolve(base, py)) if py else sys.executable

    commands = {
        "capture": [interpreter("capture"), str(STEPS["capture"]), str(cfg["capture"].get("port", "")),
                    "--out", str(capture), *flags(cfg["capture"], skip=("port",))],
        "clean": [interpreter("clean"), str(STEPS["clean"]), str(capture), *flags(cfg["clean"])],
        "pick": [interpreter("pick"), str(STEPS["pick"]), str(capture), "--points-file", str(points_file),
                 *flags(cfg["pick"])],
    }
    if "capture" in steps and not cfg["capture"].get("port"):
        sys.exit("capture.port is not set (pipeline.toml [capture] port, or --port)")

    print(f"[PIPE] capture folder {capture}")
    print(f"[PIPE] steps {' -> '.join(steps)}   (settings: {cfg_path})")
    if args.dry_run:
        for s in steps:
            print(f"\n[{s}]\n  " + subprocess.list2cmdline(commands[s]))
        return

    env = dict(os.environ, PYTHONUNBUFFERED="1")          # child output appears live
    capture.mkdir(parents=True, exist_ok=True)
    log = open(capture / "pipeline.log", "a", encoding="utf-8")
    log.write(f"\n===== {datetime.now().isoformat(timespec='seconds')}  steps {steps}\n")
    t_all = time.time()
    for s in steps:
        cmd = commands[s]
        print(f"\n{'=' * 78}\n[PIPE] {s}: {subprocess.list2cmdline(cmd)}\n{'=' * 78}", flush=True)
        log.write(f"\n----- {s}: {subprocess.list2cmdline(cmd)}\n")
        t0 = time.time()
        proc = subprocess.Popen(cmd, cwd=STEPS[s].parent, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
        log.flush()
        if code != 0:
            print(f"\n[FAIL] step '{s}' exited with code {code} after {time.time() - t0:.0f} s "
                  f"(log: {capture / 'pipeline.log'}).\n[FAIL] after fixing it, continue with:\n"
                  f"       python {Path(__file__).name} --capture-dir \"{capture}\" --from {s}")
            sys.exit(code)
        print(f"[PIPE] {s} done in {time.time() - t0:.0f} s")
    log.close()

    print(f"\n[PIPE] finished in {time.time() - t_all:.0f} s -- capture {capture}")
    if "pick" in steps and points_file.is_file():
        print(f"[PIPE] picked points ({points_file}):\n" + points_file.read_text())


if __name__ == "__main__":
    main()
