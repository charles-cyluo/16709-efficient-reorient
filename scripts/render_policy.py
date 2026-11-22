"""Render a trained policy rollout to a video (and print per-second diagnostics).

    uv run python scripts/render_policy.py --run outputs/runs/<run> --seed 3 --video outputs/videos/rl_task_eff_seed3.mp4
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
_mesa = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"  # NVIDIA EGL can hang headless; see docs
if os.path.exists(_mesa):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _mesa)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from reorient.envs.widowx_flip_env import WidowXFlipEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", required=True)
    ap.add_argument("--extra-seconds", type=float, default=1.0, help="keep recording this long after the episode ends")
    ap.add_argument("--slow", type=int, default=1, help="repeat every frame this many times (slow motion)")
    a = ap.parse_args()
    venv = DummyVecEnv([lambda: WidowXFlipEnv(efficiency=False)])
    venv = VecNormalize.load(os.path.join(a.run, "vecnormalize.pkl"), venv)
    venv.training, venv.norm_reward = False, False
    model = PPO.load(os.path.join(a.run, "final.zip"), device="cpu")
    env = venv.venv.envs[0]
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.6, 150, -15
    venv.seed(a.seed)
    obs = venv.reset()
    frames, done, extra = [], False, 0
    print("mass %.3f kg, COM %.1f cm" % (env.mass, 100 * env.com))
    while not done or extra < int(a.extra_seconds / env.control_dt):
        if not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, d, info = venv.step(act)
            done = d[0]
        else:
            env.sim.step(env.q_cmd, env.sim.d.ctrl[6])  # let the physics run on
            extra += 1
        fr = env.render(cam)
        frames += [fr] * a.slow
    m = info[0].get("episode_metrics", {})
    print("success %s, completion %.2f s, wrist rotation %.2f rad, rho %.2f, release height %.1f mm" % (m.get("success"), m.get("completion_time_s", float("nan")), m.get("wrist_rotation_rad", float("nan")), m.get("rho", float("nan")), m.get("release_height_mm", float("nan"))))
    os.makedirs(os.path.dirname(a.video) or ".", exist_ok=True)
    imageio.mimsave(a.video, frames, fps=50 // 2)
    print("wrote", a.video, len(frames), "frames")


if __name__ == "__main__":
    main()
