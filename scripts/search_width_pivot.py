"""Random search over the width-pivot controller parameters, scored on a grid of scenes (mass x COM height), in parallel.

Scenes: mass 0.15 ... 0.25 kg (a 0.1 kg range) and the COM on the cylinder axis, moved along it by up to +-1/8 of the cylinder length each way
(a total range of 1/4 of the cylinder: +-1.75 cm for the 14 cm cylinder). The simulator is deterministic, so a scene is one run.

    PROBE_DT=0.0005 uv run python scripts/search_width_pivot.py --n-sets 150 --seed 0 --tag a
    PROBE_DT=0.0005 uv run python scripts/search_width_pivot.py --eval '{"tilt_after_lift": 8}' --tag base      # one fixed parameter set

Results (every parameter set with its per-scene outcome) go to outputs/sweeps/search_<tag>.json (git-ignored).
"""
import argparse
import contextlib
import io
import itertools
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CYL_LEN = 0.14
WATCHDOG_S = 120  # wall-clock limit per run
MASSES = [0.15, 0.175, 0.20, 0.225, 0.25]
COM_Z = [-CYL_LEN / 8, -CYL_LEN / 16, 0.0, CYL_LEN / 16, CYL_LEN / 8]

DEFAULTS = dict(tilt_after_lift=0.0, tilt_time=0.6, ease_retract=0.0, ease_retract_speed=0.02, ease_rate=0.003, post_ease_rate=0.0,
                post_ease_below=40.0, delta_hold=0.012, delta_close=0.014, close_tilt=100.0, lead_time=0.04, lift=0.16, lift_speed=0.10,
                rate_gain=0.0, start_rate=8.0)


def to_argv(p, mass, com_z):
    argv = ["--seed", "0", "--mass=%r" % mass, "--com-z=%r" % com_z]
    for k, v in p.items():
        argv += ["--%s=%r" % (k.replace("_", "-"), v)]
    return argv


def run_scene(job):
    """One scene = one subprocess (a crash or hang of a run only fails that scene; a worker pool would block forever)."""
    import subprocess

    p, mass, com_z = job["params"], job["mass"], job["com_z"]
    argv = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_width_pivot.py")] + to_argv(p, mass, com_z)
    t0 = time.time()
    out = {"ok": False, "swing_ok": False}
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=WATCHDOG_S, env=dict(os.environ, PROBE_DT=os.environ.get("PROBE_DT", "0.0005")))
        line = next((l for l in cp.stdout.splitlines()[::-1] if l.startswith("RESULT_JSON ")), None)
        if line:
            res = json.loads(line[len("RESULT_JSON "):])
            out = {"ok": bool(res["ok"]), "swing_ok": bool(res.get("swing_ok")), "end_tilt": res.get("end_tilt")}
        else:
            out["error"] = "no result (exit %s): %s" % (cp.returncode, (cp.stderr or "").strip().splitlines()[-1:] )
    except subprocess.TimeoutExpired:
        out["error"] = "timeout %d s" % WATCHDOG_S
    out.update(set=job["set"], mass=mass, com_z=com_z, secs=round(time.time() - t0, 1))
    return out


def sample(rng):
    return dict(
        tilt_after_lift=round(rng.uniform(4.0, 16.0), 1),
        ease_retract=rng.choice([0.0, 0.02, 0.04, 0.06]),
        ease_retract_speed=rng.choice([0.01, 0.02, 0.04]),
        ease_rate=rng.choice([0.0015, 0.003, 0.006]),
        post_ease_rate=rng.choice([0.0, 0.001, 0.003]),
        delta_close=rng.choice([0.010, 0.014, 0.018, 0.024]),
        close_tilt=round(rng.uniform(60.0, 130.0), 0),
        lead_time=rng.choice([0.0, 0.04, 0.08]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sets", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--eval", default=None, help="JSON dict: evaluate this one parameter set (on top of the defaults) instead of searching")
    ap.add_argument("--init", default=None, help="JSON dict: base parameter set that the random search perturbs (default: DEFAULTS)")
    ap.add_argument("--random-scenes", type=int, default=0, help="instead of the 5x5 grid: this many random (mass, COM height) scenes drawn uniformly over the ranges (held-out check)")
    ap.add_argument("--scene-seed", type=int, default=123)
    ap.add_argument("--tag", default="a")
    a = ap.parse_args()
    os.environ.setdefault("PROBE_DT", "0.0005")

    rng = random.Random(a.seed)
    if a.eval:
        sets = [dict(DEFAULTS, **json.loads(a.eval))]
    else:
        base = dict(DEFAULTS, **(json.loads(a.init) if a.init else {}))
        sets = [dict(base)] + [dict(base, **sample(rng)) for _ in range(a.n_sets)]
    if a.random_scenes:
        srng = random.Random(a.scene_seed)
        scenes = [(round(srng.uniform(MASSES[0], MASSES[-1]), 4), round(srng.uniform(COM_Z[0], COM_Z[-1]), 5)) for _ in range(a.random_scenes)]
    else:
        scenes = list(itertools.product(MASSES, COM_Z))
    jobs = [dict(set=i, params=p, mass=m, com_z=z) for i, p in enumerate(sets) for m, z in scenes]
    print("%d parameter sets x %d scenes = %d runs on %d workers" % (len(sets), len(scenes), len(jobs), a.workers), flush=True)
    t0 = time.time()
    results = []
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, r in enumerate(pool.map(run_scene, jobs), 1):
            results.append(r)
            if i % 100 == 0 or i == len(jobs):
                print("  %d/%d (%.0f s)" % (i, len(jobs), time.time() - t0), flush=True)
    score = {}
    for r in results:
        score.setdefault(r["set"], []).append(r)
    ranked = sorted(score, key=lambda i: -sum(x["ok"] for x in score[i]))
    os.makedirs("outputs/sweeps", exist_ok=True)
    path = "outputs/sweeps/search_%s.json" % a.tag
    with open(path, "w") as f:
        json.dump([dict(params=sets[i], ok=sum(x["ok"] for x in score[i]), swing_ok=sum(x["swing_ok"] for x in score[i]), scenes=sorted(score[i], key=lambda x: (x["mass"], x["com_z"]))) for i in ranked], f, indent=1)
    print("wrote", path)
    print("\ntop parameter sets (full-pipeline successes out of %d scenes, swing ok):" % len(scenes))
    for i in ranked[:8]:
        ok, sw = sum(x["ok"] for x in score[i]), sum(x["swing_ok"] for x in score[i])
        diff = {k: v for k, v in sets[i].items() if DEFAULTS.get(k) != v}
        print("  %2d/%d (swing %2d/%d)  %s" % (ok, len(scenes), sw, len(scenes), json.dumps(diff)))
    if a.random_scenes:
        for i in ranked:
            bad = [x for x in score[i] if not x["ok"]]
            print("set %d: %d/%d, failures (mass kg, COM cm): %s" % (i, sum(x["ok"] for x in score[i]), len(scenes), [(x["mass"], round(100 * x["com_z"], 2)) for x in bad][:12]))
        return
    best = ranked[0]
    print("\nbest set, outcome per scene (rows COM height cm, columns mass kg): O = success, s = swing ok but place failed, . = no inverted hang")
    print("          " + "".join("%7.3f" % m for m in MASSES))
    for z in COM_Z:
        row = ""
        for m in MASSES:
            x = next(y for y in score[best] if y["mass"] == m and y["com_z"] == z)
            row += "      " + ("O" if x["ok"] else "s" if x["swing_ok"] else ".")
        print("  %+5.2f  " % (100 * z) + row)


if __name__ == "__main__":
    main()
