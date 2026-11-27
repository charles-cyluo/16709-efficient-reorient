"""Roll out a trained policy for a few episodes and print, per episode, the summed reward terms and, at a few steps, the success clauses
(tilt, placement, height above the table, at rest, robot contacts) and the gripper action. Used to see why a policy never reaches success.

    uv run python scripts/probe_episode_terms.py --run outputs/runs/2026-11-25-eff_travel_only-0 --weights '{"time": 0.0, "wrist_rotation": 0.0}'
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from reorient.envs.widowx_flip_env import TABLE_Z, WidowXFlipEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", default="{}", help="efficiency weights of the run (JSON), so the printed terms match its training reward")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--at", nargs="+", type=int, default=[100, 200, 300, 499], help="steps at which the success clauses are printed")
    a = ap.parse_args()
    w = json.loads(a.weights)
    venv = VecNormalize.load(os.path.join(a.run, "vecnormalize.pkl"), DummyVecEnv([lambda: WidowXFlipEnv(efficiency=True, weights=w)]))
    venv.training, venv.norm_reward = False, False
    model = PPO.load(os.path.join(a.run, "final.zip"), device="cpu")
    env = venv.venv.envs[0]
    for ep in range(a.episodes):
        venv.seed(ep)
        obs = venv.reset()
        tot, n = collections.Counter(), 0
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(act)
            n += 1
            for k, v in info[0]["reward_terms"].items():
                tot[k] += v
            if n in a.at and not done[0]:
                mt = env.metrics
                print("  step %d: tilt %.1f deg, placement %.1f cm, lowest point %.1f mm above table, at rest %s, robot contacts %d, grip action %+.2f"
                      % (n, mt.tilt_from_inverted_deg(), 100 * mt.placement_error(), 1000 * (mt.lowest_z() - TABLE_Z), mt.at_rest(), mt.robot_contacts(), act[0][6]))
            if done[0]:
                break
        m = info[0].get("episode_metrics", {})
        print("episode %d: %d steps, success %s, drop %s, joint path %.2f rad | %s" % (
            ep, n, m.get("success"), m.get("drop"), m.get("joint_path_rad", float("nan")), " ".join("%s %+.2f" % kv for kv in tot.items())))


if __name__ == "__main__":
    main()
