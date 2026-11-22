"""Train a PPO policy on WidowXFlipEnv with task rewards only (--reward task) or task + efficiency terms (--reward efficiency).

    uv run python scripts/train_ppo.py --reward task --steps 10000000 --n-envs 30 --seed 0 --name task_only
    uv run python scripts/train_ppo.py --reward efficiency --steps 10000000 --n-envs 30 --seed 0 --name task_eff

Run directory: outputs/runs/<YYYY-MM-DD>-<name>-<seed>/ (config.json, git revision, progress.csv, checkpoints, final.zip, vecnormalize.pkl).
GPU: the A5000 only (CUDA_VISIBLE_DEVICES=0 is set unless it is already set); --device cpu for the CPU.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # the A5000; the second NVIDIA device on this machine is not usable

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize  # noqa: E402

from reorient.envs.widowx_flip_env import WidowXFlipEnv  # noqa: E402


def make_env(efficiency, weights, seed, rank, env_kwargs):
    def _init():
        env = WidowXFlipEnv(efficiency=efficiency, weights=weights, **env_kwargs)
        env.reset(seed=seed + rank)
        return env
    return _init


class MetricsCallback(BaseCallback):
    """Rolling success / drop / efficiency metrics of the training episodes, written to the logger."""

    def __init__(self, window=300):
        super().__init__()
        self.window, self.rows = window, []
        self.term_sums = {}

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "episode_metrics" in info:
                self.rows.append(info["episode_metrics"])
        if len(self.rows) >= 20 and self.num_timesteps % 20480 < self.training_env.num_envs:
            r = self.rows[-self.window:]
            ok = [x for x in r if x["success"]]
            self.logger.record("task/success_rate", np.mean([x["success"] for x in r]))
            self.logger.record("task/drop_rate", np.mean([x["drop"] for x in r]))
            self.logger.record("task/tilt_error_deg", np.mean([x["tilt_error_deg"] for x in r]))
            self.logger.record("task/placement_error_m", np.mean([x["placement_error_m"] for x in r]))
            if ok:
                for k in ("completion_time_s", "arm_link_travel_m", "joint_path_rad", "wrist_rotation_rad", "rho"):
                    self.logger.record("eff/" + k, float(np.mean([x[k] for x in ok])))
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["task", "efficiency"], required=True)
    ap.add_argument("--steps", type=int, default=10_000_000)
    ap.add_argument("--n-envs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=3840)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--sde", action="store_true", help="generalized state-dependent exploration (temporally correlated noise)")
    ap.add_argument("--sde-freq", type=int, default=8)
    ap.add_argument("--log-std-init", type=float, default=0.0)
    ap.add_argument("--weights", default="{}", help="JSON dict overriding reward weights (see WidowXFlipEnv)")
    ap.add_argument("--learn-grasp", action="store_true", help="episode starts from a ready pose: the policy learns approach, grasp and lift too")
    ap.add_argument("--privileged-obs", action="store_true", help="append mass and COM offset to the observation")
    ap.add_argument("--fixed-scene", action="store_true", help="single mass (0.2 kg) and centred COM")
    ap.add_argument("--max-time", type=float, default=10.0)
    ap.add_argument("--max-dq", type=float, default=0.06, help="max joint step per control step [rad] (action = +-1)")
    ap.add_argument("--resume", default=None, help="run directory to resume from (loads its latest checkpoint + VecNormalize statistics)")
    ap.add_argument("--gripper-log-std", type=float, default=None, help="gSDE log-std of the gripper action only (index 6)")
    ap.add_argument("--reset-gripper-head", action="store_true", help="zero the actor output weights/bias of the gripper action (index 6): its pre-clip mean had drifted to -27 while closing had no effect")
    a = ap.parse_args()
    env_kwargs = dict(learn_grasp=a.learn_grasp, privileged_obs=a.privileged_obs, max_time=a.max_time, max_dq=a.max_dq)
    if a.fixed_scene:
        env_kwargs.update(mass_range=(0.2, 0.2), com_range=(0.0, 0.0))

    name = a.name or ("task_only" if a.reward == "task" else "task_eff")
    out = "outputs/runs/%s-%s-%d" % (datetime.date.today().isoformat(), name, a.seed)
    os.makedirs(out, exist_ok=True)
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    rev = r.stdout.strip() if r.returncode == 0 else "no-commit-yet"
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    cfg = dict(vars(a), env_kwargs=env_kwargs, git_rev=rev, git_dirty=dirty, argv=sys.argv, started=datetime.datetime.now().isoformat(timespec="seconds"))
    weights = json.loads(a.weights)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    efficiency = a.reward == "efficiency"
    venv = SubprocVecEnv([make_env(efficiency, weights, a.seed * 1000, i, env_kwargs) for i in range(a.n_envs)])
    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)
    if a.resume:
        import glob, re
        ck = max(glob.glob(os.path.join(a.resume, "ckpt_*_steps.zip")), key=lambda p: int(re.search(r"ckpt_(\d+)_steps", p).group(1)))
        n = re.search(r"ckpt_(\d+)_steps", ck).group(1)
        venv = VecNormalize.load(os.path.join(a.resume, "ckpt_vecnormalize_%s_steps.pkl" % n), venv.venv)
        venv.training, venv.norm_reward = True, True
        model = PPO.load(ck, env=venv, device=a.device, custom_objects=dict(n_steps=a.n_steps, batch_size=a.batch_size))
        print("resumed from", ck)
    else:
        model = None
    if model is None:
        model = PPO("MlpPolicy", venv, n_steps=a.n_steps, batch_size=a.batch_size, n_epochs=10, learning_rate=a.lr, gamma=0.995, gae_lambda=0.95,
                clip_range=0.2, ent_coef=a.ent_coef, use_sde=a.sde, sde_sample_freq=a.sde_freq if a.sde else -1,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=a.log_std_init), seed=a.seed, device=a.device, verbose=1)
    if a.reset_gripper_head:
        import torch
        with torch.no_grad():
            model.policy.action_net.weight[6, :] = 0.0
            model.policy.action_net.bias[6] = 0.0
    if a.gripper_log_std is not None:
        import torch
        with torch.no_grad():
            model.policy.log_std[:, 6] = a.gripper_log_std
    model.set_logger(configure(out, ["stdout", "csv"]))
    ckpt = CheckpointCallback(save_freq=max(1, 1_000_000 // a.n_envs), save_path=out, name_prefix="ckpt", save_vecnormalize=True)
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=[MetricsCallback(), ckpt], progress_bar=False, reset_num_timesteps=not a.resume)
    model.save(os.path.join(out, "final"))
    venv.save(os.path.join(out, "vecnormalize.pkl"))
    print("done in %.0f s; saved to %s" % (time.time() - t0, out))


if __name__ == "__main__":
    main()
