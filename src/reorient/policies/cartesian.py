"""Damped-least-squares Cartesian controller for the Panda TCP (joint-position commands)."""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


# Side-grasp TCP orientation: hand approach (z) = world +x, finger-opening axis (y) = world y, hand x = down.
# Columns are the TCP x, y, z axes in world coordinates.
R_SIDE = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def top_down_rotation(yaw: float) -> np.ndarray:
    """TCP orientation with the approach axis (z) pointing down and the finger-opening
    axis (y) along world direction (cos yaw, sin yaw, 0)."""
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


class CartesianController:
    """Integrating differential IK. Each call nudges the joint command toward the target pose
    using the *measured* TCP error, so steady gravity sag of the position servos is removed."""

    def __init__(self, env, gain=0.4, damping=0.05, posture_gain=0.1, max_dq=0.08):
        self.env, self.gain, self.damping = env, gain, damping
        self.posture_gain, self.max_dq = posture_gain, max_dq
        self.scratch = mujoco.MjData(env.model)
        self.q_cmd = env.arm_qpos()
        self.q_rest = env.ready_qpos[env.arm_qpos_adr].copy()
        self.jacp = np.zeros((3, env.model.nv))
        self.jacr = np.zeros((3, env.model.nv))

    def reset(self):
        self.q_cmd = self.env.arm_qpos()

    def step(self, target_pos, target_rot) -> np.ndarray:
        env = self.env
        s = self.scratch
        s.qpos[:] = env.data.qpos
        mujoco.mj_kinematics(env.model, s)
        mujoco.mj_comPos(env.model, s)
        mujoco.mj_jacSite(env.model, s, self.jacp, self.jacr, env.tcp_site)
        J = np.vstack([self.jacp, self.jacr])[:, env.arm_dof_adr]
        pos = s.site_xpos[env.tcp_site]
        rot = s.site_xmat[env.tcp_site].reshape(3, 3)
        err = np.concatenate([target_pos - pos, (R.from_matrix(target_rot) * R.from_matrix(rot).inv()).as_rotvec()])
        JJt = J @ J.T + self.damping**2 * np.eye(6)
        Jp = J.T @ np.linalg.inv(JJt)
        dq = Jp @ (self.gain * err)
        # null-space posture pull toward the ready configuration
        q = env.arm_qpos()
        dq += (np.eye(7) - Jp @ J) @ (self.posture_gain * (self.q_rest - q))
        dq = np.clip(dq, -self.max_dq, self.max_dq)
        self.q_cmd = np.clip(self.q_cmd + dq, env.ctrl_lo, env.ctrl_hi)
        # don't let the command run away from the measured state (anti-windup)
        self.q_cmd = np.clip(self.q_cmd, q - 0.3, q + 0.3)
        return self.q_cmd.copy()


def solve_ik(env, pos, rot, q0, iters=300, damping=0.02, step=0.5):
    """Kinematic (no dynamics) IK to a TCP pose with joint-limit clipping. Returns (q, err)."""
    m = env.model
    d = mujoco.MjData(m)
    d.qpos[:] = env.data.qpos
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    lo, hi = env.ctrl_lo, env.ctrl_hi
    q = np.array(q0, dtype=float)
    for _ in range(iters):
        d.qpos[env.arm_qpos_adr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, jacp, jacr, env.tcp_site)
        J = np.vstack([jacp, jacr])[:, env.arm_dof_adr]
        p = d.site_xpos[env.tcp_site]
        Rc = d.site_xmat[env.tcp_site].reshape(3, 3)
        err = np.concatenate([pos - p, (R.from_matrix(rot) * R.from_matrix(Rc).inv()).as_rotvec()])
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
        q = np.clip(q + step * dq, lo + 0.02, hi - 0.02)
    return q, float(np.linalg.norm(err))
