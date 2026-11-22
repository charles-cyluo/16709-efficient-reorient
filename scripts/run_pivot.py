"""Run the scripted pivot + set-down baseline (stock hand, thin cylinder) over N randomized trials.

    uv run python scripts/run_pivot.py --trials 20 --video outputs/videos/pivot_setdown.mp4
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
from reorient.policies.pivot import PivotConfig, PivotSetDown, enable_free_pivot, tilt_deg
from reorient.policies.rigid_flip import flip_success


def run_trial(env, seed, cfg, frames=None, every=2, verbose=False):
    env.reset(seed=seed)
    pol = PivotSetDown(env, cfg)
    start_xy = env.cup_pose()[0][:2].copy()
    last, n = None, 0
    while not pol.done and n < 4000:
        if verbose and pol.phase != last:
            print(f"   t={n*env.dt:5.2f} stage={pol.phase} tilt={tilt_deg(env):6.1f} cup={env.cup_pose()[0].round(3)}")
            last = pol.phase
        env.step(pol.act())
        if frames is not None and n % every == 0:
            frames.append(env.render())
        n += 1
    ok = flip_success(env, start_xy) and not pol.failed
    return ok, pol, env.cup_pose()[0], start_xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--dx", type=float, default=PivotConfig.grasp_dx)
    ap.add_argument("--dz", type=float, default=PivotConfig.grasp_dz)
    ap.add_argument("--lift", type=float, default=PivotConfig.lift_dz)
    ap.add_argument("--straighten-tol", type=float, default=PivotConfig.straighten_tol, help="2 = skip the straighten step")
    ap.add_argument("--swing-force", type=float, default=PivotConfig.swing_force)
    ap.add_argument("--hold-force", type=float, default=PivotConfig.hold_force)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    cfg = PivotConfig(grasp_dx=args.dx, grasp_dz=args.dz, lift_dz=args.lift, straighten_tol=args.straighten_tol, swing_force=args.swing_force, hold_force=args.hold_force)
    env = PandaCupEnv(cup_shape="cylinder")
    enable_free_pivot(env)
    ok, hung = [], []
    for i in range(args.trials):
        frames = [] if (args.video and i == 0) else None
        s, pol, cup, xy = run_trial(env, args.seed + i, cfg, frames, verbose=args.verbose)
        ok.append(s)
        hung.append(pol.hung)
        print(f"trial {i:3d} seed {args.seed+i:4d} success={s} hung={pol.hung} failed_in={pol.failed_phase or '-'} "
              f"cup={cup.round(3)} xy_err={np.linalg.norm(cup[:2]-xy):.3f}")
        if frames:
            os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
            imageio.mimsave(args.video, frames, fps=25)
    print(f"swing reached an inverted hang {sum(hung)}/{len(hung)}   success {sum(ok)}/{len(ok)} = {np.mean(ok):.0%}")


if __name__ == "__main__":
    main()
