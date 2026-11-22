"""Scripted pivot baseline (stock Panda hand, thin cylinder): pinch the object near its bottom end, lift, loosen the
grip so that it swings about the fingertips under gravity with no wrist motion, then set it down inverted.

Phases: the grasp / approach / close / lift phases of `RigidFlip` are reused. Then
  swing      arm still, light grip force; wait until the object hangs (|tilt| > hang_tilt and nearly still)
  straighten the hand pulls the object's now-upper end back over its lower end, which rests on the table
             (proportional on the axis' horizontal component), grip force kept
  release    open the gripper
  retract    straight back along the approach axis
  settle

Tilt convention: 0 = upright, +-180 = inverted. Torsional friction of the pads and the object is removed
(`enable_free_pivot`), otherwise the object cannot swing about the pinch axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from reorient.policies.cartesian import R_SIDE
from reorient.policies.rigid_flip import FlipConfig, RigidFlip


def enable_free_pivot(env):
    """Point-like contacts without torsional friction (condim 3) for the fingers and the object."""
    m = env.model
    for g in range(m.ngeom):
        b = m.body(m.geom_bodyid[g]).name
        if b in ("left_finger", "right_finger") or m.geom_bodyid[g] == env.cup_body:
            m.geom_condim[g] = 3


def grip_for_force(env, force):
    """Panda gripper action (1 = open) giving ~`force` N closing force at the current finger position."""
    q = 0.5 * float(env.data.qpos[env.finger_qpos_adr].sum())
    return float(np.clip((1000 * q - force) / 40.0, 0, 1))  # force = 0.1569 * ctrl - 1000 * q, ctrl = 255 a


def object_axis(env):
    return R.from_quat(env.cup_pose()[1][[1, 2, 3, 0]]).as_matrix()[:, 2]


def tilt_deg(env):
    ax = object_axis(env)
    return float(np.degrees(np.arctan2(ax[0], ax[2])))


@dataclass
class PivotConfig:
    grasp_dx: float = -0.016     # TCP behind the object axis: biases the swing forward, away from the palm [m]
    grasp_dz: float = -0.045     # TCP height relative to the object centre [m]
    lift_dz: float = 0.10
    swing_force: float = 2.5     # grip force while the object swings [N]
    hold_force: float = 2.5      # grip force while straightening [N]
    hang_tilt: float = 145.0     # [deg]
    hang_rate: float = 60.0      # |d tilt/dt| below this counts as hanging [deg/s]
    max_swing_time: float = 3.0
    straighten_gain: float = 0.8
    straighten_tol: float = 0.05  # |axis x| below this = upright
    straighten_speed: float = 0.08  # [m/s]
    max_straighten_time: float = 4.0
    open_time: float = 0.6
    retract_dist: float = 0.12
    retract_speed: float = 0.15
    settle_time: float = 0.8


class PivotSetDown:
    def __init__(self, env, cfg: PivotConfig | None = None):
        self.env, self.cfg = env, cfg or PivotConfig()
        c = self.cfg
        self.grasp = RigidFlip(env, FlipConfig(grasp_dx=c.grasp_dx, grasp_dz=c.grasp_dz, lift_dz=c.lift_dz))
        self.ctrl = self.grasp.ctrl
        self.cup0 = self.grasp.cup0.copy()
        self.stage, self.t = "grasp", 0.0
        self.failed, self.failed_phase = False, None
        self.grasped = False
        self.hung = False
        self.q_hold = None
        self.carrot = None
        self._prev_tilt = 0.0
        self._goal = None
        self._upright_t = 0.0
        self.t_swing_end = None

    @property
    def phase(self):
        return self.stage

    @property
    def done(self):
        return self.stage == "done"

    def _fail(self, where):
        self.failed, self.failed_phase, self.stage = True, where, "done"

    def act(self):
        env, c = self.env, self.cfg
        dt = env.dt
        if self.stage == "grasp":
            if self.grasp.phase == "roll" or self.grasp.done:
                if self.grasp.failed or self.grasp.done:
                    self._fail(self.grasp.failed_phase or "grasp")
                    return np.concatenate([self.ctrl.q_cmd, [1.0]])
                self.grasped = self.grasp.grasped
                self.q_hold = self.ctrl.q_cmd.copy()
                self.stage, self.t = "swing", 0.0
                self._prev_tilt = tilt_deg(env)
            else:
                return self.grasp.act()
        self.t += dt
        tcp = env.tcp_pose()[0]
        if self.stage == "swing":
            tl = tilt_deg(env)
            rate = abs(tl - self._prev_tilt) / dt if abs(tl) < 180 else 0.0
            self._prev_tilt = tl
            if abs(tl) > c.hang_tilt and rate < c.hang_rate:
                self.hung, self.stage, self.t = True, "straighten", 0.0
                self.t_swing_end = None
                self.carrot = tcp.copy()
            elif self.t > c.max_swing_time:
                self._fail("swing")
            return np.concatenate([self.q_hold, [grip_for_force(env, c.swing_force)]])
        if self.stage == "straighten":
            ax = object_axis(env)
            # object's lower end is at centre + axis*half_length; pull the hand so the axis becomes vertical
            step = np.clip(c.straighten_gain * ax[0] * 2 * env.cup_half_height, -c.straighten_speed * dt, c.straighten_speed * dt)
            self.carrot = self.carrot + np.array([step, 0.0, 0.0])
            q = self.ctrl.step(self.carrot, R_SIDE)
            self._upright_t = self._upright_t + dt if abs(ax[0]) < c.straighten_tol else 0.0
            if self._upright_t > 0.3:
                self.stage, self.t = "release", 0.0
            elif self.t > c.max_straighten_time:
                self._fail("straighten")
            return np.concatenate([q, [grip_for_force(env, c.hold_force)]])
        if self.stage == "release":
            q = self.ctrl.step(self.carrot, R_SIDE)
            if self.t >= c.open_time:
                self.stage, self.t = "retract", 0.0
                self._goal = tcp + np.array([-c.retract_dist, 0.0, 0.0])
            return np.concatenate([q, [1.0]])
        if self.stage == "retract":
            d = self._goal - self.carrot
            n = np.linalg.norm(d)
            s = c.retract_speed * dt
            self.carrot = self._goal.copy() if n <= s else self.carrot + d / n * s
            q = self.ctrl.step(self.carrot, R_SIDE)
            if np.linalg.norm(self._goal - tcp) < 0.01:
                self.stage, self.t = "settle", 0.0
            elif self.t > 6.0:
                self._fail("retract")
            return np.concatenate([q, [1.0]])
        # settle
        q = self.ctrl.step(self.carrot, R_SIDE)
        if self.t >= c.settle_time:
            self.stage = "done"
        return np.concatenate([q, [1.0]])
