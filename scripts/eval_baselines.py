"""Evaluate the scripted controllers (width pivot, naive wrist roll) on random scenes with the metrics of docs/project/metrics.md.

Scenes: cylinder mass uniform in 0.15-0.25 kg, COM on the axis uniform within +-1/8 of the cylinder length (+-1.75 cm); fixed start position.
One scene = one subprocess of scripts/run_width_pivot.py.

    PROBE_DT=0.0005 uv run python scripts/eval_baselines.py --n 200 --seed 0
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from reorient.eval.aggregate import aggregate, format_table  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def run_scene(job):
    controller, mass, com = job
    argv = [sys.executable, os.path.join(HERE, "run_width_pivot.py"), "--seed", "0", "--mass=%r" % mass, "--com-z=%r" % com, "--controller", controller]
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=180, env=dict(os.environ, PROBE_DT=os.environ.get("PROBE_DT", "0.0005")))
        line = next((l for l in cp.stdout.splitlines()[::-1] if l.startswith("RESULT_JSON ")), None)
        res = json.loads(line[len("RESULT_JSON "):]) if line else None
    except subprocess.TimeoutExpired:
        res = None
    m = (res or {}).get("metrics") or {"success": False, "drop": False}
    m.update(controller=controller, mass=mass, com_z=com)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--controllers", nargs="+", default=["pivot", "roll"])
    ap.add_argument("--tag", default="baselines")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    scenes = [(round(rng.uniform(0.15, 0.25), 4), round(rng.uniform(-0.0175, 0.0175), 5)) for _ in range(a.n)]
    jobs = [(c, m, z) for c in a.controllers for m, z in scenes]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(run_scene, jobs))
    print("%d runs in %.0f s" % (len(jobs), time.time() - t0))
    named = {c: aggregate([r for r in rows if r["controller"] == c]) for c in a.controllers}
    os.makedirs("outputs/eval", exist_ok=True)
    with open("outputs/eval/%s.json" % a.tag, "w") as f:
        json.dump({"rows": rows, "aggregate": named}, f, indent=1)
    table = format_table(named)
    with open("outputs/eval/%s.md" % a.tag, "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
