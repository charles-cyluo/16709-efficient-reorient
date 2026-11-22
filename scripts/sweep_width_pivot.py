"""Parallel sweep over the release width, closing squeeze and close trigger of the width-controlled pivot, for several masses.

Each worker process builds its own simulation (the simulator is deterministic, one run = one setting). The swing is scored on its own
(`--swing-only`): success = the cylinder is held in the pads, 160-200 deg inverted after the closing, lean < 20 deg.

    PROBE_DT=0.0005 uv run python scripts/sweep_width_pivot.py --lift 0.16 --workers 24
    PROBE_DT=0.0005 uv run python scripts/sweep_width_pivot.py --full ...   # run the whole pipeline (place + release) for every setting

Results are written to outputs/sweeps/width_pivot_<tag>.json (git-ignored).
"""
import argparse
import contextlib
import io
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_one(job):
    import run_width_pivot  # imported inside the worker (sets the headless GL variables first)

    argv = ["--seed", "0", "--mass", str(job["mass"]), "--lift", str(job["lift"]), "--ease-retract", str(job["ease_retract"]),
            "--fixed-delta", str(job["delta_release"]), "--delta-close", str(job["delta_close"]),
            "--close-tilt", str(job["close_tilt"]), "--delta-hold", str(job["delta_hold"]), "--lift-speed", str(job["lift_speed"])]
    argv += job.get("extra", [])
    if job.get("lift_only"):
        argv += ["--swing-timeout", "0.05"]
    if not job["full"]:
        argv.append("--swing-only")
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            res = run_width_pivot.main(argv)
    except Exception as e:  # keep the sweep going
        res = {"ok": False, "swing_ok": False, "error": repr(e)}
    res.update(job)
    res["secs"] = round(time.time() - t0, 1)
    t = res.get("end_tilt")
    res["swing_good"] = bool(res.get("swing_ok") and t is not None and 160 <= t <= 200 and res.get("end_lean", 90) < 20)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--masses", type=float, nargs="+", default=[0.10, 0.14, 0.18, 0.22, 0.26, 0.30])
    ap.add_argument("--delta-release", type=float, nargs="+", default=[0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045], help="m below the contact width")
    ap.add_argument("--delta-close", type=float, nargs="+", default=[0.006, 0.010])
    ap.add_argument("--close-tilt", type=float, nargs="+", default=[100.0, 125.0])
    ap.add_argument("--delta-hold", type=float, nargs="+", default=[0.005], help="squeeze while lifting [m below the contact width]")
    ap.add_argument("--lift-speed", type=float, nargs="+", default=[0.10])
    ap.add_argument("--lift", type=float, default=0.16)
    ap.add_argument("--ease-retract", type=float, default=0.0)
    ap.add_argument("--full", action="store_true", help="whole pipeline, not just the swing")
    ap.add_argument("--lift-only", action="store_true", help="only report the state after the lift (tilt / distance to the TCP) for every (mass, hold squeeze, lift speed)")
    ap.add_argument("--extra", default="", help="extra arguments passed to every run of run_width_pivot.py, e.g. \"--rate-gain 0\"")
    ap.add_argument("--tag", default="swing")
    a = ap.parse_args()
    os.environ.setdefault("PROBE_DT", "0.0005")

    jobs = [dict(mass=m, delta_release=dr, delta_close=dc, close_tilt=ct, delta_hold=dh, lift_speed=ls, lift=a.lift, ease_retract=a.ease_retract, full=a.full, lift_only=a.lift_only, extra=a.extra.split())
            for m, dr, dc, ct, dh, ls in itertools.product(a.masses, a.delta_release, a.delta_close, a.close_tilt, a.delta_hold, a.lift_speed)]
    print("%d runs on %d workers" % (len(jobs), a.workers), flush=True)
    t0 = time.time()
    results = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            results.append(r)
            if i % 20 == 0 or i == len(jobs):
                print("  %d/%d done (%.0f s)" % (i, len(jobs), time.time() - t0), flush=True)
    os.makedirs("outputs/sweeps", exist_ok=True)
    path = "outputs/sweeps/width_pivot_%s.json" % a.tag
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", path)

    if a.lift_only:
        print("\nstate after the lift: tilt[deg]/distance cup-tcp[cm]; 'ok' = tilt < 10 deg and held")
        for sp in a.lift_speed:
            print("\nlift speed %.2f m/s" % sp)
            print("mass   " + "  ".join(("hold %2.0f mm" % (1000 * h)).ljust(11) for h in a.delta_hold))
            for m in a.masses:
                row = []
                for h in a.delta_hold:
                    r = next(x for x in results if x["mass"] == m and x["delta_hold"] == h and x["lift_speed"] == sp and x["delta_release"] == a.delta_release[0]
                             and x["delta_close"] == a.delta_close[0] and x["close_tilt"] == a.close_tilt[0])
                    t, d = r.get("lift_tilt"), r.get("lift_dist")
                    row.append(("ok" if t is not None and abs(t) < 10 and d < 0.08 else ("%4.0f/%3.0f" % (t, 100 * d)) if t is not None else "err").ljust(11))
                print("%.2f   " % m + "  ".join(row))
        return
    key = "ok" if a.full else "swing_good"
    print("\nsuccessful settings per mass (%s):" % key)
    for m in a.masses:
        rs = [r for r in results if r["mass"] == m]
        good = [r for r in rs if r[key]]
        print("mass %.2f: %d/%d" % (m, len(good), len(rs)))
        for r in sorted(good, key=lambda r: abs((r.get("end_tilt") or 0) - 180))[:4]:
            print("    release %.1f mm, close %.0f mm, close tilt %.0f -> end tilt %.0f lean %.0f" % (
                1000 * r["delta_release"], 1000 * r["delta_close"], r["close_tilt"], r.get("end_tilt", float("nan")), r.get("end_lean", float("nan"))))


if __name__ == "__main__":
    main()
