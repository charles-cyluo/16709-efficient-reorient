"""Exploratory pivot probe (NOT a baseline): side-pinch the object near its bottom end, lift, then reduce the grip
force and watch whether the object swings about the fingertips. Optionally record a video.

    uv run python scripts/probe_pivot.py --cup-shape cylinder --dx -0.016 --force 2.5 --seed 0 --video outputs/pivot.mp4

--dx: TCP offset along the approach axis relative to the object axis (negative = pinch behind the axis, which biases the
swing forward, away from the palm). --force: total gripper force after the lift (N, tendon force; panda: ~hold 20+).
"""
import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
_mesa = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"  # NVIDIA EGL can hang headless; see docs
if os.path.exists(_mesa):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _mesa)

import imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from reorient.envs import PandaCupEnv
from reorient.policies.rigid_flip import FlipConfig, RigidFlip


def grip_action(env, force):
    """Gripper action (1 = open) giving ~`force` N closing force at the current finger position."""
    q = 0.5 * float(env.data.qpos[env.finger_qpos_adr].sum())
    if env.gripper == "panda":  # force = 0.1569 * ctrl - 1000 * q, ctrl = 255 a
        return float(np.clip((1000 * q - force) / 40.0, 0, 1))
    return float(np.clip(1 - (force + 100 * q) / 0.3137255 / 255, 0, 1))  # robotiq


def set_condim(env, condim):
    """3 = no torsional friction (lets the object swing about the pinch), 4 = torsional friction (holds it)."""
    m = env.model
    for g in range(m.ngeom):
        b = m.body(m.geom_bodyid[g]).name
        if b in ("left_finger", "right_finger") or b.startswith("rq_") or m.geom_bodyid[g] == env.cup_body:
            m.geom_condim[g] = condim


def no_torsion(env):
    set_condim(env, 3)


def tilt_deg(env):
    ax = R.from_quat(env.cup_pose()[1][[1, 2, 3, 0]]).as_matrix()[:, 2]
    return float(np.degrees(np.arctan2(ax[0], ax[2])))  # 0 = upright, +90 = pointing forward (+x), +-180 = inverted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gripper", default="panda", choices=["panda", "robotiq"])
    ap.add_argument("--cup-shape", default="cylinder", choices=["cup", "cylinder"])
    ap.add_argument("--dx", type=float, default=-0.016)
    ap.add_argument("--dz", type=float, default=-0.045, help="TCP height relative to object center (negative = lower)")
    ap.add_argument("--force", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reclamp-tilt", type=float, default=150.0, help="re-tighten the grip once |tilt| exceeds this [deg]; 0 = never")
    ap.add_argument("--hold-force", type=float, default=20.0)
    ap.add_argument("--time", type=float, default=3.5, help="seconds to watch after the lift")
    ap.add_argument("--torsion", action="store_true", help="keep torsional friction (default: removed)")
    ap.add_argument("--torsion-coef", type=float, default=0.0, help="torsional friction coefficient [m] of fingers + object (condim 4); 0 = condim 3 (removed)")
    ap.add_argument("--pinch-force", type=float, default=0.0, help="grip force during the lift [N]; 0 = fully closed")
    ap.add_argument("--sphere-pad", type=float, default=0.0, help="replace the Panda pad boxes with one sphere of this radius per finger [m]")
    ap.add_argument("--noslip", type=int, default=0, help="noslip_iterations (post-solve friction anti-creep)")
    ap.add_argument("--impratio", type=float, default=None)
    ap.add_argument("--cone", default=None, choices=["pyramidal", "elliptic"])
    ap.add_argument("--solref", type=float, nargs=2, default=None, help="contact solref of the fingers and object (timeconst dampratio)")
    ap.add_argument("--solimp", type=float, nargs=5, default=None, help="contact solimp of the fingers and object")
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--pad-mu", type=float, default=None, help="finger friction")
    ap.add_argument("--summary", action="store_true", help="one-line result")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    env = PandaCupEnv(gripper=args.gripper, cup_shape=args.cup_shape)
    if not args.torsion:
        no_torsion(env)
    m = env.model
    m.opt.noslip_iterations = args.noslip
    if args.impratio is not None:
        m.opt.impratio = args.impratio
    if args.cone:
        m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL if args.cone == "pyramidal" else mujoco.mjtCone.mjCONE_ELLIPTIC
    for g in range(m.ngeom):
        if m.body(m.geom_bodyid[g]).name in ("left_finger", "right_finger") or m.geom_bodyid[g] == env.cup_body:
            if args.solref:
                m.geom_solref[g] = args.solref
            if args.solimp:
                m.geom_solimp[g] = args.solimp
            if args.margin is not None:
                m.geom_margin[g] = args.margin
            if args.pad_mu is not None and m.geom_bodyid[g] != env.cup_body:
                m.geom_friction[g, 0] = args.pad_mu
    if args.torsion_coef:
        for g in range(m.ngeom):
            if m.body(m.geom_bodyid[g]).name in ("left_finger", "right_finger") or m.geom_bodyid[g] == env.cup_body:
                m.geom_condim[g] = 4
                m.geom_friction[g, 1] = args.torsion_coef
    if args.sphere_pad:
        r = args.sphere_pad
        for gl, gp, others in ((70, 70, (69, 71, 72, 73, 74)), (78, 78, (77, 79, 80, 81, 82))):
            for o in others:
                m.geom_contype[o] = m.geom_conaffinity[o] = 0
            m.geom_type[gp] = mujoco.mjtGeom.mjGEOM_SPHERE
            m.geom_size[gp] = [r, 0, 0]
            m.geom_pos[gp] = [0.0, 0.0015 + r, 0.0445]
            m.geom_rbound[gp] = r
            m.geom_aabb[gp] = [0, 0, 0, r, r, r]
    env.reset(seed=args.seed)
    pol = RigidFlip(env, FlipConfig(grasp_dz=args.dz, grasp_dx=args.dx, lift_dz=0.10))
    frames = []
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.7, 150, -15

    def snap():
        cam.lookat[:] = env.tcp_pose()[0] + np.array([0.0, 0.0, 0.03])
        frames.append(env.render(cam))

    n = 0
    while pol.phase != "roll" and not pol.done and n < 3000:  # pregrasp, approach, close, lift
        act = pol.act()
        if args.pinch_force and pol.phase in ("lift", "roll"):
            act[7] = grip_action(env, args.pinch_force)
        env.step(act)
        n += 1
        if args.video and n % 2 == 0:
            snap()
    if pol.failed:
        print("grasp/lift failed in", pol.failed_phase)
        return
    q = pol.ctrl.q_cmd.copy()
    trace, force, reclamped = [], args.force, False
    for k in range(int(args.time / env.dt)):
        if args.reclamp_tilt and not reclamped and abs(tilt_deg(env)) > args.reclamp_tilt:
            force, reclamped = args.hold_force, True
            set_condim(env, 4)  # close the grip on the inverted object: torsional friction back on
        env.step(np.concatenate([q, [grip_action(env, force)]]))
        if k % int(0.1 / env.dt) == 0:
            trace.append(round(tilt_deg(env)))
        if args.video and k % 2 == 0:
            snap()
    dist = float(np.linalg.norm(env.cup_pose()[0] - env.tcp_pose()[0]))
    if args.summary:
        mx = max(abs(t) for t in trace)
        print("seed %d: max|tilt| %3d final %4d, %s (dist %.3f)" % (args.seed, mx, trace[-1], "IN fingers" if dist < 0.08 else "left", dist))
        return
    print("tilt every 0.1 s (deg; 0 up, 180 inverted):", trace)
    print("re-clamped:", reclamped)
    print("object %s the fingers at the end (distance %.3f m)" % ("still in" if dist < 0.08 else "LEFT", dist))
    if args.video:
        os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
        imageio.mimsave(args.video, frames, fps=25)
        print("wrote", args.video, len(frames), "frames")


if __name__ == "__main__":
    main()
