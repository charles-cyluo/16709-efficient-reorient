"""Run a queue of train_ppo.py runs (conditions x seeds from a config file), a few at a time; finished runs are skipped.

    uv run python scripts/sweep_ppo.py --config configs/ablation_v1.json --conditions task_only task_eff --seeds 0 1 2 3 4 --parallel 1

A run counts as finished when some outputs/runs/*-<condition>-<seed>/final.zip exists (any date). Logs: outputs/logs/<condition>-<seed>.log.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time


def finished(cond, seed):
    return bool(glob.glob("outputs/runs/*-%s-%d/final.zip" % (cond, seed)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--conditions", nargs="+", default=None, help="default: all conditions in the config")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--seed-major", action="store_true", help="order the queue seed by seed (all conditions of seed 0 first) instead of condition by condition")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    conds = a.conditions or list(cfg["conditions"])
    pairs = [(c, s) for s in a.seeds for c in conds] if a.seed_major else [(c, s) for c in conds for s in a.seeds]
    queue = []
    for c, s in pairs:
        if finished(c, s):
            print("skip (finished):", c, s)
            continue
        spec = cfg["conditions"][c]
        cmd = [sys.executable, "scripts/train_ppo.py", "--reward", spec["reward"], "--weights", json.dumps(spec["weights"]),
               "--seed", str(s), "--name", c] + cfg["common"]
        queue.append((c, s, cmd))
    os.makedirs("outputs/logs", exist_ok=True)
    print("%d runs queued, %d at a time" % (len(queue), a.parallel))
    if a.dry_run:
        for _, _, cmd in queue:
            print(" ".join(cmd))
        return
    running = []
    while queue or running:
        while queue and len(running) < a.parallel:
            c, s, cmd = queue.pop(0)
            log = open("outputs/logs/%s-%d.log" % (c, s), "w")
            print(time.strftime("%H:%M:%S"), "start", c, s, flush=True)
            running.append((c, s, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
        time.sleep(30)
        for item in list(running):
            c, s, p = item
            if p.poll() is not None:
                print(time.strftime("%H:%M:%S"), "end", c, s, "exit", p.returncode, flush=True)
                running.remove(item)


if __name__ == "__main__":
    main()
