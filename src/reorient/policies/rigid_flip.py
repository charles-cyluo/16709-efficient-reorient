"""Scripted rigid flip: side-grasp the cup, lift, roll the wrist 180 deg (joint 7), set the cup down
upside-down at its start position, release and retract. The cup is held rigidly (no in-hand slip).

Hand frame: approach (z) = world +x (horizontal), finger-opening axis (y) = world y, hand x = down.
Rolling about the approach axis (= joint 7's axis) flips the cup about a horizontal axis. Phases:
pregrasp, approach, close, lift, roll (pure joint-7 motion, min-jerk), lower (servoing the *cup* to its
start xy / table), release, retract (straight back along the approach axis), settle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from reorient import TABLE_Z
from reorient.policies.cartesian import R_SIDE, CartesianController


def roll(rot, angle):
    """Rotate a TCP orientation by `angle` about its own z (approach) axis."""
    return rot @ R.from_euler("z", angle).as_matrix()


@dataclass
class FlipConfig:
    standoff: float = 0.08       # TCP distance short of the cup center along -x when approaching [m]
    grasp_dx: float = -0.01       # TCP offset along the approach axis relative to cup center (<0: cup deeper toward fingertips) [m]
    grasp_dz: float = 0.0        # TCP height relative to cup center when grasping [m]
    lift_dz: float = 0.15        # lift height before rolling [m]
    roll_angle: float = np.pi    # wrist roll [rad]
    roll_time: float = 2.0       # [s]
    speed: float = 0.25          # [m/s]
    pos_tol: float = 0.01
    rot_tol: float = 0.10
    close_time: float = 0.8
    open_time: float = 0.6
    settle_time: float = 0.8
    retract_dist: float = 0.12   # [m]
    place_clearance: float = 0.001  # cup center above its resting height when released [m]
    max_phase_time: float = 8.0


class RigidFlip:
    """Phase list; each phase is (name, target_pos_fn, roll_angle_fn, gripper)."""

    def __init__(self, env, cfg: FlipConfig | None = None):
        self.env, self.cfg = env, cfg or FlipConfig()
        self.ctrl = CartesianController(env, gain=0.5)
        self.reset()

    def reset(self):
        self.ctrl.reset()
        c = self.cup0 = self.env.cup_pose()[0].copy()
        g = self.cfg
        grasp = np.array([c[0] + g.grasp_dx, c[1], c[2] + g.grasp_dz])
        pre = grasp - np.array([g.standoff, 0, 0])
        lifted = grasp + np.array([0, 0, g.lift_dz])
        # (name, position, roll, gripper, kind) kind: "move" waits for pose, "wait" waits for time
        self.phases = [
            ("pregrasp", pre, 0.0, 1.0, "move"),
            ("approach", grasp, 0.0, 1.0, "move"),
            ("close", grasp, 0.0, 0.0, "wait", g.close_time),
            ("lift", lifted, 0.0, 0.0, "move"),
            ("roll", lifted, g.roll_angle, 0.0, "move"),
            ("lower", grasp, g.roll_angle, 0.0, "move"),
            ("release", grasp, g.roll_angle, 1.0, "wait", g.open_time),
            ("retract", pre, g.roll_angle, 1.0, "move"),
            ("settle", pre, g.roll_angle, 1.0, "wait", g.settle_time),
        ]
        self.i = 0
        self.t = 0.0
        self.failed, self.failed_phase = False, None
        self.grasped = False  # set once the cup has been lifted clear of the table while held
        self.carrot = self.env.tcp_pose()[0].copy()
        self.roll_now = self.roll_from = 0.0
        self.q_hold = None
        self._retract_goal = None

    @property
    def phase(self):
        return "done" if self.i >= len(self.phases) else self.phases[self.i][0]

    @property
    def done(self):
        return self.i >= len(self.phases)

    def act(self):
        env, g = self.env, self.cfg
        if self.done:
            return np.concatenate([self.ctrl.q_cmd, [1.0]])
        name, pos, roll_t, grip, kind, *rest = self.phases[self.i]
        if name == "lower":
            # servo the *cup* (not the TCP) to its start xy, resting on the table
            cup = env.cup_pose()[0]
            goal_cup = np.array([self.cup0[0], self.cup0[1], self.cup0[2] + g.place_clearance])
            pos = env.tcp_pose()[0] + (goal_cup - cup)
            self.carrot = np.where(np.arange(3) < 2, pos, self.carrot)  # xy: follow measured error
        if name == "retract":
            # back straight out along the approach axis: the open fingers only clear the cup by ~9 mm
            if self._retract_goal is None:
                self._retract_goal = env.tcp_pose()[0] + np.array([-g.retract_dist, 0.0, 0.0])
            pos = self._retract_goal
        # carrot: move linearly toward goal at cfg.speed
        d = pos - self.carrot
        dist = np.linalg.norm(d)
        step = g.speed * env.dt
        self.carrot = pos.copy() if dist <= step else self.carrot + d / dist * step
        # min-jerk blend of the wrist roll over roll_time (smooth start/stop avoids exciting the grasp)
        u = min(1.0, self.t / g.roll_time)
        if name == "roll":
            # The roll is a pure joint-7 motion (joint 7's axis is the hand's approach axis); doing it
            # through IK wobbles the other joints and drags the cup out of the grasp.
            if self.q_hold is None:
                self.q_hold, self.q7_start = self.ctrl.q_cmd.copy(), env.arm_qpos()[6]
            self.roll_now = self.roll_from + (roll_t - self.roll_from) * (10 * u**3 - 15 * u**4 + 6 * u**5)
            q = self.q_hold.copy()
            q[6] = self.q7_start + (self.roll_now - self.roll_from)
            self.ctrl.q_cmd = q.copy()
        else:
            self.q_hold = None
            self.roll_now = roll_t
            q = self.ctrl.step(self.carrot, roll(R_SIDE, self.roll_now))
        tcp, rot = env.tcp_pose()
        self.t += env.dt
        if kind == "move":
            rot_err = np.linalg.norm(R.from_matrix(rot.T @ roll(R_SIDE, roll_t)).as_rotvec())
            pos_ok = np.linalg.norm(pos - tcp) < g.pos_tol or name == "roll"  # roll: arm sags a bit under load
            if pos_ok and rot_err < g.rot_tol and (name != "roll" or u >= 1.0):
                self._next()
            elif self.t > g.max_phase_time:
                self.failed, self.failed_phase = True, name
                self.i = len(self.phases)
        elif self.t >= rest[0]:
            self._next()
        return np.concatenate([q, [grip]])

    def _next(self):
        if self.phase == "lift":
            cup = self.env.cup_pose()[0]
            self.grasped = bool(cup[2] > self.cup0[2] + 0.5 * self.cfg.lift_dz
                                and np.linalg.norm(cup - self.env.tcp_pose()[0]) < 0.03)
        self.i += 1
        self.t = 0.0
        self.roll_from = self.roll_now
        self._retract_goal = None


def flip_success(env, start_xy, tilt_tol_deg=15.0, place_tol=0.03) -> bool:
    """Cup axis points down within tilt_tol, resting within place_tol of its start xy, and
    sitting on the table (center near half height, not falling)."""
    p, q = env.cup_pose()
    axis = R.from_quat(q[[1, 2, 3, 0]]).as_matrix()[:, 2]
    tilt = np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))
    return bool(tilt < tilt_tol_deg and np.linalg.norm(p[:2] - start_xy) < place_tol and p[2] < TABLE_Z + env.cup_half_height + 0.015)
