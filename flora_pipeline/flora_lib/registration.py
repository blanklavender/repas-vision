"""
Sweep registration and the plant frame.

chain_sweep   adjacent-pair tag registration chained into the first shot's frame; on a
              full turn the last->first edge closes the loop and its error is spread
              evenly over the shots (shot k gets k/n of the correction).
plant_frame   turntable axis fitted to the registered rotations, pointing up; origin on
              the axis at the pot base; x horizontal toward the reference camera.
"""

import numpy as np

from .geometry import make_T, rot_deg, rotate_about, rotvec, unit
from .tags import register_pair


class RegistrationError(RuntimeError):
    pass


def is_closed_loop(degs):
    """True if the logged angles cover a full turn (last stop + one step >= 360)."""
    degs = np.asarray(degs, float)
    if len(degs) < 3:
        return False
    step = float(np.median(np.abs(np.diff(degs))))
    return bool(abs(degs[-1] - degs[0]) + step >= 360.0 - 0.5 * step)


def chain_sweep(tags, degs, names=None, loop="auto", log=print):
    """tags: per-shot tag dicts (tags.detect_tags), degs: logged turntable angles, names:
    labels for the messages (default: positions in the sweep).

    Returns (poses, info): poses[k] = T_shot_k_to_ref (X_ref = T @ X_shot), and info with
    per-edge rotation / tag ids and the loop error."""
    names = [f"{k:03d}" for k in range(len(tags))] if names is None else [str(n) for n in names]
    edges = [register_pair(a, b) for a, b in zip(tags, tags[1:])]
    info = dict(edges=[], loop=None)
    for k, (T, ids) in enumerate(edges):
        rot = None if T is None else rot_deg(T[:3, :3])
        info["edges"].append(dict(src=names[k], tgt=names[k + 1], tags=ids, rotation_deg=rot,
                                  logged_deg=degs[k + 1] - degs[k]))
        log(f"[EDGE] {names[k]}->{names[k + 1]}  tags {str(ids):<16} rotation "
            f"{'  ---' if rot is None else f'{rot:6.2f}'} deg  (logged {degs[k + 1] - degs[k]:.2f})")
    bad = [f"{e['src']}->{e['tgt']}" for e, (T, _) in zip(info["edges"], edges) if T is None]
    if bad:
        raise RegistrationError(f"no shared usable tag between shots {bad}: the "
                                f"chain is broken. Raise --max-obliquity or capture with a smaller step.")

    poses = [np.eye(4)]
    for T, _ in edges:
        poses.append(poses[-1] @ np.linalg.inv(T))
    closed = {"yes": True, "no": False, "auto": is_closed_loop(degs)}[loop]
    if closed:
        T_loop, ids = register_pair(tags[-1], tags[0])
        if T_loop is None:
            log("[WARN] the last and first shot share no usable tag -- chained only, no loop closure")
        else:
            n = len(poses)
            D = poses[-1] @ np.linalg.inv(T_loop)         # should be the identity
            rD, tD = rotvec(D[:3, :3]), D[:3, 3]
            poses = [make_T(rotate_about(rD, -k / n), -(k / n) * tD) @ P for k, P in enumerate(poses)]
            info["loop"] = dict(tags=ids, error_deg=rot_deg(D[:3, :3]), error_mm=float(np.linalg.norm(tD) * 1000))
    return poses, info


def plant_frame(poses, degs, P, base_percentile=0.5):
    """Turntable axis + plant origin/axes in the reference camera frame.

    P: the fused cloud (reference frame). Returns dict axis, origin, R (plant -> camera
    columns x y z), turn_sign (+1/-1: a turn of +d deg rotates points by turn_sign * d
    about axis), ref_deg, height_range_mm, drift_deg (registered vs logged turn per shot)."""
    rv = [rotvec(T[:3, :3]) for T in poses]
    used = [k for k, r in enumerate(rv) if np.linalg.norm(r) > np.radians(2.0)]
    if not used:
        raise RegistrationError("the shots span less than 2 deg of rotation -- cannot fit the turntable axis")
    a = np.linalg.svd(np.array([unit(rv[k]) for k in used]))[2][0]   # dominant axis, sign-free
    if a @ np.array([0.0, -1.0, 0.0]) < 0:                             # up (camera y points down)
        a = -a
    # point p on the axis:  P_perp (I - R_k) p = P_perp t_k  for every shot, + a.p = a.centroid
    Pp = np.eye(3) - np.outer(a, a)
    A = [Pp @ (np.eye(3) - poses[k][:3, :3]) for k in used] + [a[None]]
    b = [Pp @ poses[k][:3, 3] for k in used] + [np.array([a @ P.mean(0)])]
    p = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]

    O = p + a * np.percentile((P - p) @ a, base_percentile)
    v = -O                                                            # axis -> camera centre
    x = unit(v - (v @ a) * a)
    R = np.column_stack([x, np.cross(a, x), a])

    k1 = min(used, key=lambda k: abs(degs[k] - degs[0]))
    s = float(np.sign(rotvec(np.linalg.inv(poses[k1])[:3, :3]) @ a) or 1.0)
    drift = [((np.degrees(rotvec(np.linalg.inv(T)[:3, :3]) @ a) * s - (d - degs[0]) + 180.0) % 360.0) - 180.0
             for T, d in zip(poses, degs)]
    h = (P - O) @ a
    return dict(axis=a, origin=O, R=R, turn_sign=s, ref_deg=float(degs[0]),
                height_range_mm=[float(np.percentile(h, 0.1) * 1000), float(np.percentile(h, 99.9) * 1000)],
                drift_deg=np.array(drift))


def at_turn(X, F, deg):
    """Reference-camera position of X once the turntable stands at logged angle `deg`."""
    return F["origin"] + rotate_about(F["axis"], F["turn_sign"] * np.radians(deg - F["ref_deg"])) @ (X - F["origin"])


def plant_coords(X, F):
    """X (reference camera frame) -> plant-frame xyz, radius, azimuth (deg), height, and the
    turntable angle at which X faces the camera (azimuth 0)."""
    q = F["R"].T @ (X - F["origin"])
    az = float(np.degrees(np.arctan2(q[1], q[0])))
    return dict(xyz=q, radius=float(np.hypot(q[0], q[1])), azimuth_deg=az, height=float(q[2]),
                faces_camera_deg=(F["ref_deg"] - F["turn_sign"] * az) % 360.0)
