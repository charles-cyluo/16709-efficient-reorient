"""Gymnasium environment: invert a thin cylinder with the WidowX 250 S and set it down inverted at its start position.

The episode starts at the moment of **pickup**: the cylinder is gripped near its bottom end and lifted 16 cm (a snapshot produced once per process by a
scripted approach / close / lift, see `pickup_snapshot`). Each reset randomizes the cylinder's mass and the height of its centre of mass along the axis, and
starts from that snapshot. The policy then has to invert the cylinder, put it down on the table at its start position, release it and move the hand away.
The efficiency metrics (docs/project/metrics.md) count from the start of the episode, i.e. "after pickup".

Action (7, in [-1, 1]): 6 joint position increments (|a| = 1: `max_dq` rad per control step, 20 ms) and the gripper: the finger-position target is
`q_contact - (0.009 + 0.015 a)`, i.e. a = 0 is a firm grip (9 mm below the contact width, about 9 N), a = -1 opens by 6 mm, a = +1 squeezes by 24 mm.
Observation (38): arm q (6), arm dq (6), finger q (2), finger dq (2), cylinder position (3) / quaternion (4) / linear velocity (3) / angular velocity (3),
time fraction (1), previous action (7), robot-contact flag (1).
Reward = task terms (+ efficiency terms if `efficiency=True`); every term is reported in `info["reward_terms"]`.
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from reorient import TABLE_Z
from reorient.envs.widowx_sim import CLOSED, CUP_XY, CYL_H, OPEN, Sim
from reorient.eval.metrics import EpisodeMetrics
from reorient.policies.cartesian import R_SIDE

LIFT = 0.16
GRASP_DZ = -0.035  # learn_grasp: assigned grasp point below the cylinder centre; at -4.5 cm (scripted pickup) the level hand's wrist link rests on the table (TCP 2.5 cm above it)
HOLD_SQUEEZE = 0.012
_SNAPSHOTS: dict = {}


def pickup_snapshot(dt=0.0005):
    """Scripted pickup (approach, close, lift 16 cm with a firm grip) -> state snapshot of the pickup moment. Cached per process."""
    if dt in _SNAPSHOTS:
        return _SNAPSHOTS[dt]
    sim = Sim(0, 0.2, 1.5, pad_mu=1.5, sphere_r=0.010, pad_x=0.09, dt=dt, ctrl_dt=0.02)
    sim.apply_contact_model()
    sim.set_cylinder(0.2, 0.0)
    cx, cy = CUP_XY
    grasp = np.array([cx - 0.008, cy, TABLE_Z + CYL_H - 0.045])
    pre = grasp - np.array([0.04, 0, 0])
    q_gr, _ = sim.solve_ik(grasp, R_SIDE, [0, 0.7, 0.6, 0, -1.3, 0])
    q_pre, _ = sim.solve_ik(pre, R_SIDE, q_gr)
    sim.reset(q_pre)
    q, carrot, width = q_pre.copy(), pre.copy(), [OPEN]

    def run(secs, goal=None, speed=0.12):
        nonlocal q, carrot
        for _ in range(int(secs / 0.02)):
            if goal is not None:
                d = goal - carrot
                n = float(np.linalg.norm(d))
                carrot = carrot + d / max(n, 1e-9) * min(n, speed * 0.02)
            q = sim.servo(q, carrot, R_SIDE)
            sim.step(q, width[0])

    run(0.5)
    run(1.5, goal=grasp)
    run(0.3)
    for _ in range(300):  # close at 3 cm/s until the fingers stop
        width[0] = max(CLOSED, width[0] - 0.03 * 0.02)
        run(0.02)
        if abs(sim.d.qvel[sim.m.joint("left_finger").dofadr[0]]) < 5e-3 and width[0] < sim.d.qpos[sim.lf] - 0.001:
            break
    q_contact = float(sim.d.qpos[sim.lf])
    width[0] = q_contact - HOLD_SQUEEZE
    run(0.3)
    run(3.0, goal=grasp + np.array([0, 0, LIFT]), speed=0.10)
    run(0.5)
    snap = dict(qpos=sim.d.qpos.copy(), qvel=sim.d.qvel.copy(), ctrl=sim.d.ctrl.copy(), q_cmd=q.copy(), q_contact=q_contact, target_xy=np.array(CUP_XY))
    _SNAPSHOTS[dt] = snap
    return snap


class WidowXFlipEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, efficiency=False, weights=None, mass_range=(0.15, 0.25), com_range=(-0.0175, 0.0175), max_time=10.0, dt=0.0005,
                 control_dt=0.02, max_dq=0.06, render_size=(480, 640), learn_grasp=False, privileged_obs=False):
        self.efficiency = efficiency
        self.learn_grasp, self.privileged_obs = learn_grasp, privileged_obs
        self.w = dict(table_contact=0.02, joint_limit=0.05, approach=6.0, orient=4.0, close=3.0, grasp=3.0, lift=4.0, inversion=3.0, position=1.0, height=2.0, release=1.0, retreat=1.0, success=20.0, drop=10.0,
                      time=0.02, arm_travel=0.5, wrist_rotation=2.0)
        if weights:
            self.w.update(weights)
        self.mass_range, self.com_range, self.max_time = mass_range, com_range, max_time
        self.control_dt, self.max_dq = control_dt, max_dq
        self.snap = pickup_snapshot(dt)
        self.sim = Sim(0, 0.2, 1.5, pad_mu=1.5, sphere_r=0.010, pad_x=0.09, dt=dt, ctrl_dt=control_dt)
        self.sim.apply_contact_model()
        self.sim.set_cylinder(0.2, 0.0)
        self.target_xy = self.snap["target_xy"]
        self.q_contact = self.snap["q_contact"]
        self.metrics = EpisodeMetrics(self.sim, self.target_xy, period=control_dt)
        m = self.sim.m
        self.finger_dof = [m.joint("left_finger").dofadr[0], m.joint("right_finger").dofadr[0]]
        self.finger_qadr = [m.joint("left_finger").qposadr[0], m.joint("right_finger").qposadr[0]]
        self.cup_dof = m.joint("cup_joint").dofadr[0]
        self.wrist_adr = m.joint("wrist_rotate").qposadr[0]
        if learn_grasp:  # episode starts from a ready pose with the gripper open: the policy also learns approach, grasp and lift
            grasp = np.array([CUP_XY[0] - 0.008, CUP_XY[1], TABLE_Z + CYL_H + GRASP_DZ])
            self.q_ready, _ = self.sim.solve_ik(grasp + np.array([-0.12, 0, 0.07]), R_SIDE, [0, 0.7, 0.6, 0, -1.3, 0])
            self.tcp_site = m.site("tcp").id
            self.pad_ids = [m.geom("left_pad").id, m.geom("right_pad").id]
            self.cup_z0 = TABLE_Z + CYL_H
        self.max_steps = int(max_time / control_dt)
        self.render_size = render_size
        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (38 + (7 if learn_grasp else 0) + (2 if privileged_obs else 0),), np.float32)

    # ---- helpers --------------------------------------------------------------------------------------
    def _obs(self):
        sim, d = self.sim, self.sim.d
        p, quat = sim.cup()
        if quat[0] < 0:
            quat = -quat
        contacts = 1.0 if self.metrics.robot_contacts() > 0 else 0.0
        extra = []
        if self.learn_grasp:
            tcp = sim.tcp()
            extra.append(np.concatenate([tcp, p - tcp, [(self.grip_cmd - CLOSED) / (OPEN - CLOSED)]]))
        if self.privileged_obs:
            extra.append([(self.mass - 0.2) / 0.05, self.com / 0.0175])
        return np.concatenate(extra + [
            d.qpos[sim.adr], d.qvel[sim.dof], d.qpos[self.finger_qadr], d.qvel[self.finger_dof], p, quat,
            d.qvel[self.cup_dof : self.cup_dof + 3], d.qvel[self.cup_dof + 3 : self.cup_dof + 6],
            [self.t / self.max_time], self.prev_action, [contacts]]).astype(np.float32)

    def _potentials(self):
        mt = self.metrics
        tilt_err = np.radians(mt.tilt_from_inverted_deg())
        xy = mt.placement_error()
        h = max(mt.lowest_z() - TABLE_Z, 0.0)
        placed = float(tilt_err < np.radians(15) and h < 0.01)
        opening = (self.sim.d.qpos[self.finger_qadr[0]] - (self.q_contact - 0.012)) / (OPEN - (self.q_contact - 0.012))
        dist = float(np.linalg.norm(self.sim.tcp() - mt.lower_end_centre()))
        # the height term only pays when the cylinder is (nearly) inverted: lowering an upright cylinder to the table earns nothing (it was a plateau of the first runs)
        gate = float(np.clip(1.0 - tilt_err / (np.pi / 2), 0.0, 1.0))
        extra = {}
        if self.learn_grasp:
            # Pre-grasp potentials (approach, orient, close, grasp) use exponential kernels: steep near the target, bounded, and potential-based so that
            # hovering or retreating earns nothing. Once the cup has been lifted they are frozen at their maximum (no penalty for the pivot or for easing the grip).
            c = self.sim.cup()[0]
            tcp = self.sim.tcp()
            lift = float(np.clip((c[2] - self.cup_z0) / 0.16, 0.0, 1.0))
            touch = self._pads_touch()
            if self.lifted:
                pos_pot = ori_pot = close_pot = grasp_pot = 1.0
            else:
                d3 = self._grasp_dist()  # assigned grasp position: cup centre + (-8 mm, 0, GRASP_DZ)
                pos_pot = 0.3 * (1.0 - min(d3, 0.6) / 0.6) + 0.35 * np.exp(-d3 / 0.10) + 0.35 * np.exp(-(d3 / 0.03) ** 2)  # far-field slope + two exponential scales
                rot_err = float(R.from_matrix(self.sim.d.site_xmat[self.tcp_site].reshape(3, 3).T @ R_SIDE).magnitude())  # assigned orientation: side-grasp frame
                ori_pot = float(np.exp(-rot_err / 0.3))
                closed = 1.0 - (self.grip_cmd - CLOSED) / (OPEN - CLOSED)
                close_pot = float(np.exp(-(d3 / 0.015) ** 2)) * closed  # closing pays in proportion to how close the hand is to the grasp pose
                grasp_pot = float(touch)
            if lift > 0.1:
                self.lifted = True
            extra = dict(approach=float(pos_pot), orient=ori_pot, grasp=grasp_pot, close=close_pot,
                         lift=lift * (1.0 - gate) * float(touch or self.lifted))
        return dict(**extra, inversion=-tilt_err / np.pi, position=-min(xy, 0.25) / 0.25, height=gate * (1.0 - min(h, 0.2) / 0.2),
                    release=placed * float(np.clip(opening, 0, 1)), retreat=placed * min(dist, 0.2) / 0.2)

    def _robot_table_contact(self):
        """True if any robot geom touches the table (the cup does not count)."""
        d, m = self.sim.d, self.sim.m
        tb = self.metrics.table
        for c in d.contact[: d.ncon]:
            if tb in (c.geom1, c.geom2):
                o = c.geom2 if c.geom1 == tb else c.geom1
                b = m.geom_bodyid[o]
                if b != self.metrics.cb and b != 0:
                    return True
        return False

    def _grasp_dist(self):
        # anisotropic: the open gripper clears the cup by only ~5.5 mm per side in y (pad spheres at +-38 mm, radius 10 mm, cup radius 22.5 mm), so lateral (y) error
        # is weighted x4 and height (z) x2 against the approach direction (x); a policy 1.6 cm off in y rightly refuses to advance (a pad would topple the cup)
        return float(np.linalg.norm((self.sim.tcp() - (self.sim.cup()[0] + np.array([-0.008, 0.0, GRASP_DZ]))) * np.array([1.0, 4.0, 2.0])))

    def _pads_touch(self):
        d, m = self.sim.d, self.sim.m
        hit = set()
        for c in d.contact[: d.ncon]:
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g in self.pad_ids and m.geom_bodyid[o] == self.metrics.cb:
                    hit.add(g)
        return len(hit) == 2

    # ---- gym API --------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        sim, d, m = self.sim, self.sim.d, self.sim.m
        rng = self.np_random
        self.mass = float(rng.uniform(*self.mass_range))
        self.com = float(rng.uniform(*self.com_range))
        sim.set_cylinder(self.mass, self.com)
        mujoco.mj_resetData(m, d)
        if self.learn_grasp:
            q0 = self.q_ready + rng.uniform(-0.03, 0.03, 6)
            sim.reset(q0)
            self.q_cmd = q0.copy()
            self.lifted = False
            self.grip_cmd = OPEN
        else:
            d.qpos[:], d.qvel[:], d.ctrl[:] = self.snap["qpos"], self.snap["qvel"], self.snap["ctrl"]
            self.q_cmd = self.snap["q_cmd"].copy()
        d.time = 0.0
        mujoco.mj_forward(m, d)
        for _ in range(int(0.2 / sim.dt)):  # let the contacts settle with the new mass
            mujoco.mj_step(m, d)
        self.t, self.steps = 0.0, 0
        self.prev_action = np.zeros(7)
        if self.learn_grasp:
            self.prev_action[6] = -1.0
        self.metrics.start()
        self.pot = self._potentials()
        self.prev_q = d.qpos[sim.adr].copy()
        return self._obs(), {"mass": self.mass, "com_z": self.com}

    def step(self, action):
        sim, d = self.sim, self.sim.d
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_now = d.qpos[sim.adr]
        lo, hi = sim.lo.copy(), sim.hi.copy()
        if self.learn_grasp:  # workspace limit: the cup is straight ahead, the base yaw stays within +-1 rad (wrist_rotate keeps its full range; forearm roll +-1 rad)
            lo[0], hi[0] = -1.0, 1.0
            lo[3], hi[3] = -1.0, 1.0  # forearm roll: no solution needs it (the roll baseline uses wrist_rotate); at +-pi it allowed a flipped arm configuration
        self.q_cmd = np.clip(self.q_cmd + a[:6] * self.max_dq, np.maximum(lo, q_now - 0.35), np.minimum(hi, q_now + 0.35))
        if self.learn_grasp:  # finger target moves at a * 1.5 mm per control step (a = -1 opens, +1 closes; full range in ~15 steps = 0.3 s)
            # closing is only possible at the grasp position (anisotropic distance <= 2 cm: the pads span +-2.45 cm around the TCP, closer than that they straddle the cup), or once the cup is touched / lifted; opening is always possible
            # (without this PPO closed the gripper early and stalled with closed fingertips against the cup)
            close_ok = self.lifted or self._grasp_dist() < 0.02 or self._pads_touch()
            self.grip_cmd = float(np.clip(self.grip_cmd - (a[6] if (a[6] < 0 or close_ok) else 0.0) * 0.0015, CLOSED, OPEN))
            grip = self.grip_cmd
        else:
            grip = float(np.clip(self.q_contact - (0.009 + 0.015 * a[6]), CLOSED, OPEN))
        sim.step(self.q_cmd, grip)  # one control period; the metrics hook below samples it
        self.metrics.update(force=True)
        self.t += self.control_dt
        self.steps += 1
        # ---- reward ----
        pot = self._potentials()
        keys = ("inversion", "position", "height", "release", "retreat") + (("approach", "orient", "close", "grasp", "lift") if self.learn_grasp else ())
        terms = {k: self.w[k] * (pot[k] - self.pot[k]) for k in keys}
        self.pot = pot
        q = d.qpos[sim.adr].copy()
        dq = np.abs(q - self.prev_q)
        self.prev_q = q
        if self.efficiency:
            terms["time"] = -self.w["time"]
            terms["arm_travel"] = -self.w["arm_travel"] * float(dq.sum())
            terms["wrist_rotation"] = -self.w["wrist_rotation"] * float(dq[5])
        if self.learn_grasp:
            if self._robot_table_contact():
                terms["table_contact"] = -self.w["table_contact"]
            mid, half = 0.5 * (sim.hi + sim.lo), 0.5 * (sim.hi - sim.lo)  # joints 1, 2, 4 (shoulder, elbow, wrist pitch): within 10% of a limit; the roll joints are exempt
            near = np.clip((np.abs(q[[1, 2, 4]] - mid[[1, 2, 4]]) / half[[1, 2, 4]] - 0.9) / 0.1, 0.0, 1.0)
            terms["joint_limit"] = -self.w["joint_limit"] * float(near.sum())
        terminated = False
        if self.metrics.t_success is not None and self.metrics.success_condition():
            terms["success"] = self.w["success"]
            terminated = True
        elif self.metrics.drop:
            terms["drop"] = -self.w["drop"]
            terminated = True
        truncated = (not terminated) and self.steps >= self.max_steps
        self.prev_action = a
        reward = float(sum(terms.values()))
        info = {"reward_terms": terms}
        if terminated or truncated:
            info["episode_metrics"] = self.metrics.summary()
            info["mass"], info["com_z"] = self.mass, self.com
        return self._obs(), reward, terminated, truncated, info

    def render(self, camera=None):
        if camera is None:
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 0.6, 150, -15
        return self.sim.render(camera)
