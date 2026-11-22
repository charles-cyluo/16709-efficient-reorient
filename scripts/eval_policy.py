"""Evaluate trained policies on random scenes with the metrics of docs/project/metrics.md and compare them with the scripted baselines.

    uv run python scripts/eval_policy.py --runs outputs/runs/<date>-task_only-0 outputs/runs/<date>-task_eff-0 --n 200 --tag policies_v1

Each run directory needs final.zip and vecnormalize.pkl (written by train_ppo.py). Scenes are drawn from the same distribution as for the scripted baselines
(mass 0.15-0.25 kg, COM on the axis within +-1.75 cm); every environment runs ceil(n / n_envs) episodes so that short episodes are not over-represented.
If outputs/eval/<baselines>.json exists, its aggregates are added to the table.
"""
import argparse
import json
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # the A5000

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize  # noqa: E402

from reorient.envs.widowx_flip_env import WidowXFlipEnv  # noqa: E402
from reorient.eval.aggregate import aggregate, format_table  # noqa: E402


def make_env(seed):
    def _init():
        env = WidowXFlipEnv(efficiency=False)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(run, n, n_envs, seed, device="cpu", deterministic=True, ckpt_steps=None):
    venv = SubprocVecEnv([make_env(seed + i) for i in range(n_envs)])
    if ckpt_steps:  # evaluate the checkpoint saved at (about) this many steps instead of the final policy
        steps = sorted(int(f.split("_")[1]) for f in os.listdir(run) if f.startswith("ckpt_") and f.endswith("_steps.zip"))
        step = min(steps, key=lambda x: abs(x - ckpt_steps))
        vn, zp = "ckpt_vecnormalize_%d_steps.pkl" % step, "ckpt_%d_steps.zip" % step
    else:
        vn, zp = "vecnormalize.pkl", "final.zip"
    venv = VecNormalize.load(os.path.join(run, vn), venv)
    venv.training, venv.norm_reward = False, False
    model = PPO.load(os.path.join(run, zp), device=device)
    per_env = math.ceil(n / n_envs)
    done_count = np.zeros(n_envs, int)
    rows = []
    obs = venv.reset()
    while done_count.min() < per_env:
        act, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, infos = venv.step(act)
        for i, (d, info) in enumerate(zip(dones, infos)):
            if d:
                done_count[i] += 1
                if done_count[i] <= per_env and "episode_metrics" in info:
                    r = dict(info["episode_metrics"])
                    r.update(mass=info.get("mass"), com_z=info.get("com_z"))
                    rows.append(r)
    venv.close()
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-envs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--ckpt-steps", type=int, default=None, help="evaluate the checkpoint closest to this many training steps (default: final policy)")
    ap.add_argument("--groups", action="store_true", help="names are <condition>/<seed>: also print per-seed success and pooled results per condition")
    ap.add_argument("--baselines", default="baselines_v2")
    ap.add_argument("--tag", default="policies")
    a = ap.parse_args()
    names = a.names or [os.path.basename(r.rstrip("/")) for r in a.runs]
    named, allrows = {}, {}
    bpath = "outputs/eval/%s.json" % a.baselines
    if os.path.exists(bpath):
        named.update({"scripted " + k: v for k, v in json.load(open(bpath))["aggregate"].items()})
    for run, name in zip(a.runs, names):
        rows = evaluate(run, a.n, a.n_envs, a.seed, deterministic=not a.stochastic, ckpt_steps=a.ckpt_steps)
        allrows[name] = rows
        if not a.groups:
            named["RL " + name] = aggregate(rows)
    if a.groups:
        conds = {}
        for name, rows in allrows.items():
            conds.setdefault(name.split("/")[0], []).append((name, rows))
        for cond, lst in conds.items():
            named["RL " + cond + " (%d seeds pooled)" % len(lst)] = aggregate([r for _, rows in lst for r in rows])
            print(cond, "per-seed success: " + ", ".join("%s %.1f%%" % (n.split("/")[1], 100 * np.mean([r["success"] for r in rows])) for n, rows in lst))
    os.makedirs("outputs/eval", exist_ok=True)
    with open("outputs/eval/%s.json" % a.tag, "w") as f:
        json.dump({"rows": allrows, "aggregate": named}, f, indent=1)
    table = format_table(named)
    with open("outputs/eval/%s.md" % a.tag, "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
