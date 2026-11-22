"""Task and efficiency metrics of one inversion episode (WidowX + thin cylinder). Definitions: docs/project/metrics.md.

Usage: create one `EpisodeMetrics(sim, target_xy)` per episode, call `start()` at the moment of pickup (the cylinder is gripped and lifted) and `update()` after
every control step (it samples at most every `period` s of simulated time, so scripted controllers that step at 2 ms and the RL environment at 20 ms are
scored identically); `summary()` returns the dictionary.
"""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from reorient import TABLE_Z

# success tolerances (AGENTS.md): inverted within 15 deg, placement error < 3 cm, no drop; plus: standing on the table, at rest, robot clear
TILT_TOL_DEG = 15.0
PLACE_TOL = 0.03
REST_LIN, REST_ANG = 0.01, 0.1  # [m/s], [rad/s]
SETTLE_S = 0.5  # the success condition must hold this long
AIRBORNE_Z = 0.005  # [m] lowest point above the table with no robot contact = dropped (released in the air: tightened from 3 cm after the first PPO policy placed by throwing the cylinder down from ~1.5 cm)
THROWN_SPEED = 0.2  # [m/s] downward speed of the cylinder at the moment it is released: faster = thrown / dropped
TOPPLED_LEAN_DEG = 60.0  # lying on the table (axis this far from the vertical) = dropped / toppled
LINKS = ("wx250s/upper_arm_link", "wx250s/upper_forearm_link", "wx250s/lower_forearm_link", "wx250s/wrist_link", "wx250s/gripper_link")
ROLL_JOINTS = ("forearm_roll", "wrist_rotate")


def _quat_to_R(q):
    return R.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()


def angle_between(Ra, Rb):
    """Rotation angle [rad] between two orientations."""
    return float(R.from_matrix(Ra.T @ Rb).magnitude())


class EpisodeMetrics:
    def __init__(self, sim, target_xy, period=0.02):
        self.sim, self.target_xy, self.period = sim, np.asarray(target_xy, float), period
        m = sim.m
        self.cb = m.body("cup").id
        self.table = m.geom("table").id
        self.link_ids = [m.body(n).id for n in LINKS]
        self.roll_adr = {n: m.joint(n).qposadr[0] for n in ROLL_JOINTS}
        self.hand = m.body("wx250s/gripper_link").id
        self.started = False

    # ---- state helpers -------------------------------------------------------------------------------
    def cup_axis(self):
        return _quat_to_R(self.sim.cup()[1])[:, 2]

    def lower_end_centre(self):
        c, ax = self.sim.cup()[0], self.cup_axis()
        h = self.sim.cyl_h
        return c + ax * h if (c + ax * h)[2] < (c - ax * h)[2] else c - ax * h

    def lowest_z(self):
        ax = self.cup_axis()
        d = -np.array([0, 0, 1.0]) + ax[2] * ax
        n = float(np.linalg.norm(d))
        return float(self.lower_end_centre()[2] + (self.sim.cyl_r * d[2] / n if n > 1e-6 else 0.0))

    def robot_contacts(self):
        d, m = self.sim.d, self.sim.m
        n = 0
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if self.cb in (b1, b2) and self.table not in (c.geom1, c.geom2):
                n += 1
        return n

    def tilt_from_inverted_deg(self):
        return float(np.degrees(np.arccos(np.clip(-self.cup_axis()[2], -1, 1))))

    def placement_error(self):
        return float(np.linalg.norm(self.lower_end_centre()[:2] - self.target_xy))

    def at_rest(self):
        sim = self.sim
        dof = sim.m.joint("cup_joint").dofadr[0]
        v = sim.d.qvel[dof : dof + 6]
        return bool(np.linalg.norm(v[:3]) < REST_LIN and np.linalg.norm(v[3:]) < REST_ANG)

    def success_condition(self):
        """Standing inverted within tolerance at the target, resting on the table, at rest, and the robot is not touching it."""
        return bool(self.tilt_from_inverted_deg() < TILT_TOL_DEG and self.placement_error() < PLACE_TOL
                    and self.lowest_z() < TABLE_Z + 0.01 and self.at_rest() and self.robot_contacts() == 0)

    def lean_deg(self):
        """Angle of the cylinder axis from the vertical (0 = standing either way up, 90 = lying)."""
        return float(np.degrees(np.arccos(np.clip(abs(self.cup_axis()[2]), 0, 1))))

    def dropped_now(self):
        """Failure by dropping: airborne (lowest point well above the table) with no robot contact, out of the workspace, or toppled onto its side
        (lying on the table: axis more than 60 deg from the vertical with the lowest point on the table)."""
        c = self.sim.cup()[0]
        airborne = self.lowest_z() > TABLE_Z + AIRBORNE_Z and self.robot_contacts() == 0
        toppled = self.lowest_z() < TABLE_Z + 0.012 and self.lean_deg() > TOPPLED_LEAN_DEG
        return bool(airborne or toppled or c[2] < TABLE_Z - 0.05)

    # ---- accumulation ----------------------------------------------------------------------------------
    def start(self):
        sim, d = self.sim, self.sim.d
        self.started = True
        self.t0 = d.time
        self.t_last = d.time
        self.prev_tcp = sim.tcp()
        self.prev_links = d.xpos[self.link_ids].copy()
        self.prev_q = d.qpos[sim.adr].copy()
        self.prev_roll = {n: d.qpos[a] for n, a in self.roll_adr.items()}
        self.tcp_path = 0.0
        self.link_path = 0.0
        self.joint_path = 0.0
        self.roll_rot = {n: 0.0 for n in ROLL_JOINTS}
        self.drop = False
        self.t_hold = None  # start of the current stretch in which the success condition holds
        self.t_success = None
        self.R_cup0 = _quat_to_R(sim.cup()[1])
        self.R_rel0 = d.xmat[self.hand].reshape(3, 3).T @ self.R_cup0
        self.R_cup_grip = self.R_cup0.copy()  # orientations at the last control step in which both pads still touch the cylinder
        self.R_rel_grip = self.R_rel0.copy()
        self.max_tilt_rate = 0.0
        self.final_check = None
        self.in_contact = True
        self.release_height = float(self.lowest_z() - TABLE_Z)
        self.release_speed = 0.0

    def update(self, force=False):
        if not self.started:
            return
        sim, d = self.sim, self.sim.d
        if not force and d.time - self.t_last < self.period - 1e-9:
            return
        self.t_last = d.time
        tcp = sim.tcp()
        links = d.xpos[self.link_ids].copy()
        q = d.qpos[sim.adr].copy()
        self.tcp_path += float(np.linalg.norm(tcp - self.prev_tcp))
        self.link_path += float(np.linalg.norm(links - self.prev_links, axis=1).sum())
        self.joint_path += float(np.abs(q - self.prev_q).sum())
        for n, a in self.roll_adr.items():
            self.roll_rot[n] += abs(d.qpos[a] - self.prev_roll[n])
            self.prev_roll[n] = d.qpos[a]
        self.prev_tcp, self.prev_links, self.prev_q = tcp, links, q
        # orientation bookkeeping for rho: remember the orientations while the cylinder is still gripped (>= 2 contacts with the robot)
        if self.robot_contacts() >= 2:
            self.R_cup_grip = _quat_to_R(sim.cup()[1])
            self.R_rel_grip = d.xmat[self.hand].reshape(3, 3).T @ self.R_cup_grip
        contact_now = self.robot_contacts() > 0
        if self.in_contact and not contact_now:  # (final) release: height of the lowest point and downward speed of the cylinder at that moment
            dof = sim.m.joint("cup_joint").dofadr[0]
            self.release_height = float(self.lowest_z() - TABLE_Z)
            self.release_speed = float(max(-d.qvel[dof + 2], 0.0))
            if self.release_speed > THROWN_SPEED:
                self.drop = True
        self.in_contact = contact_now
        if self.dropped_now():
            self.drop = True
        ok = self.success_condition()
        if ok and self.t_hold is None:
            self.t_hold = d.time
        elif not ok:
            self.t_hold = None
        if self.t_hold is not None and self.t_success is None and d.time - self.t_hold >= SETTLE_S:
            self.t_success = self.t_hold  # completion time = when the stable success state was reached

    # ---- result -----------------------------------------------------------------------------------------
    def summary(self):
        """Dictionary of the metrics at the current moment (call at the end of the episode)."""
        sim, d = self.sim, self.sim.d
        success = self.t_success is not None and self.success_condition() and not self.drop
        # total rotation: net rotation of the cylinder in the world between pickup and the last gripped control step;
        # in-hand rotation: net rotation of the cylinder relative to the hand frame over the same interval
        total = angle_between(self.R_cup0, self.R_cup_grip)
        inhand = angle_between(self.R_rel0, self.R_rel_grip)
        rho = float(np.clip(inhand / total, 0.0, 1.0)) if total > 1e-3 else 0.0
        out = dict(
            success=bool(success),
            drop=bool(self.drop),
            tilt_error_deg=self.tilt_from_inverted_deg(),
            placement_error_m=self.placement_error(),
            completion_time_s=(self.t_success - self.t0) if self.t_success is not None else float("nan"),
            episode_time_s=d.time - self.t0,
            tcp_path_m=self.tcp_path,
            arm_link_travel_m=self.link_path,
            joint_path_rad=self.joint_path,
            wrist_rotation_rad=self.roll_rot["wrist_rotate"],
            roll_joint_rotation_rad=sum(self.roll_rot.values()),
            cup_rotation_deg=float(np.degrees(total)),
            inhand_rotation_deg=float(np.degrees(inhand)),
            rho=rho,
            release_height_mm=1000 * self.release_height,
            release_speed_mps=self.release_speed,
        )
        return out
