"""Run the scripted rigid flip over N randomized trials.

    uv run python scripts/run_rigid_flip.py --trials 20 --video outputs/rigid_flip.mp4
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
_mesa = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"  # NVIDIA EGL can hang headless; see docs
if os.path.exists(_mesa):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _mesa)

import imageio
import numpy as np

from reorient.envs import PandaCupEnv
from reorient.policies.rigid_flip import RigidFlip, flip_success


def run_trial(env, seed, frames=None, every=2, verbose=False):
    env.reset(seed=seed)
    pol = RigidFlip(env)
    start_xy = env.cup_pose()[0][:2].copy()
    last, n = None, 0
    while not pol.done and n < 3000:
        if verbose and pol.phase != last:
            print(f"   t={n*env.dt:5.2f} phase={pol.phase} cup={env.cup_pose()[0].round(3)}")
            last = pol.phase
        env.step(pol.act())
        if frames is not None and n % every == 0:
            frames.append(env.render())
        n += 1
    return flip_success(env, start_xy) and not pol.failed, pol, env.cup_pose()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gripper", default="panda", choices=["panda", "robotiq"])
    ap.add_argument("--cup-shape", default="cup", choices=["cup", "cylinder"])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    env = PandaCupEnv(gripper=args.gripper, cup_shape=args.cup_shape)
    ok, grasps = [], []
    for i in range(args.trials):
        frames = [] if (args.video and i == 0) else None
        s, pol, cup = run_trial(env, args.seed + i, frames, verbose=args.verbose)
        ok.append(s)
        grasps.append(pol.grasped)
        print(f"trial {i:3d} seed {args.seed+i:4d} flip_success={s} grasped={pol.grasped} failed_in={pol.failed_phase or '-'} cup={cup.round(3)}")
        if frames:
            os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
            imageio.mimsave(args.video, frames, fps=25)
    print(f"grasp+lift {sum(grasps)}/{len(grasps)} = {np.mean(grasps):.0%}   flip success {sum(ok)}/{len(ok)} = {np.mean(ok):.0%}")


if __name__ == "__main__":
    main()
