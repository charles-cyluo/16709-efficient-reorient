"""Scripted wrist-roll expert that acts through the action interface of `WidowXFlipEnv` (joint increments + gripper). Used to check that the episode is
solvable within the horizon and as a demonstration source. (The full-featured scripted controllers, including the gravity pivot, are in
scripts/run_width_pivot.py.)"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from reorient import TABLE_Z
from reorient.policies.cartesian import R_SIDE


class RollExpert:
    def __init__(self, env, roll_time=2.0):
        self.env, self.roll_time = env, roll_time
        self.phase, self.t_phase = "roll", 0.0
        sim = env.sim
        self.q0_wrist = float(env.q_cmd[5])
        self.angle = min(np.pi, sim.m.jnt_range[sim.m.joint("wrist_rotate").id, 1] - 1e-3)
        self.rot = R_SIDE
        self.carrot = None

    def _to_action(self, q_target, grip):
        env = self.env
        a = np.zeros(7)
        a[:6] = np.clip((q_target - env.q_cmd) / env.max_dq, -1, 1)
        a[6] = grip
        return a

    def act(self):
        env, sim, mt = self.env, self.env.sim, self.env.metrics
        dt = env.control_dt
        self.t_phase += dt
        if self.phase == "roll":
            u = min(1.0, self.t_phase / self.roll_time)
            q_t = env.q_cmd.copy()
            q_t[5] = self.q0_wrist + self.angle * (10 * u**3 - 15 * u**4 + 6 * u**5)
            if u >= 1.0 and self.t_phase > self.roll_time + 0.5:
                self.phase, self.t_phase = "align", 0.0
                self.rot = R_SIDE @ R.from_euler("z", self.angle).as_matrix()
                self.carrot = sim.tcp()
            return self._to_action(q_t, 0.0)
        if self.phase == "align":  # move the lower end face over the target, then descend
            err = env.target_xy - mt.lower_end_centre()[:2]
            n = float(np.linalg.norm(err))
            if n > 0.003:
                self.carrot[:2] += err / n * min(n, 0.06 * dt)
            elif mt.lowest_z() > TABLE_Z + 0.001:
                self.carrot[2] -= 0.04 * dt
            else:
                self.phase, self.t_phase = "release", 0.0
            q_t = sim.servo(env.q_cmd, self.carrot, self.rot)
            return self._to_action(q_t, 0.0)
        if self.phase == "release":
            q_t = sim.servo(env.q_cmd, self.carrot, self.rot)
            if self.t_phase > 0.9:
                self.phase, self.t_phase = "up", 0.0
                self.goal_z = self.carrot[2] + 0.06
            return self._to_action(q_t, -1.0)
        if self.phase == "up":
            self.carrot[2] = min(self.carrot[2] + 0.05 * dt, self.goal_z)
            if self.carrot[2] >= self.goal_z - 1e-6:
                self.phase, self.t_phase = "back", 0.0
                self.goal_x = self.carrot[0] - 0.10
            q_t = sim.servo(env.q_cmd, self.carrot, self.rot)
            return self._to_action(q_t, -1.0)
        # back
        self.carrot[0] = max(self.carrot[0] - 0.10 * dt, self.goal_x)
        q_t = sim.servo(env.q_cmd, self.carrot, self.rot)
        return self._to_action(q_t, -1.0)
