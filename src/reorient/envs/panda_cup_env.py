"""Franka Panda + cup on a table (MuJoCo, Gymnasium-style).

Milestone-1 version: no reward/termination yet. The env exposes the pieces the scripted
baselines need (joint-position targets + gripper command, ground-truth cup pose, randomized
cup mass/friction). Reward terms and episode logic are added with the RL milestone.

Action (provisional): 8 floats = 7 absolute joint-position targets [rad] + gripper command
in [0, 1] (0 = closed, 1 = open).
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from reorient import TABLE_Z
from reorient.envs.model import GRIPPERS, build_model
from reorient.policies.cartesian import R_SIDE, solve_ik

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
READY_KEY = "ready"


class PandaCupEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        cup_shape: str = "cup",  # "cup" (hollow, 9 cm) or "cylinder" (thin solid, 4.5 x 14 cm, like the reference video)
        gripper: str = "panda",  # "panda" (stock hand) or "robotiq" (2F-85)
        n_substeps: int = 40,  # 40 * 0.5 ms = 50 Hz control
        mass_range=(0.10, 0.30),
        friction_range=(0.6, 1.2),
        cup_xy_jitter: float = 0.02,
        render_size=(480, 640),
    ):
        assert gripper in GRIPPERS
        self.gripper = gripper
        self.model = build_model(gripper)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = n_substeps
        self.dt = n_substeps * self.model.opt.timestep
        self.mass_range, self.friction_range = mass_range, friction_range
        self.cup_xy_jitter = cup_xy_jitter
        self.render_size = render_size
        self._renderer = None

        m = self.model
        self.arm_qpos_adr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_dof_adr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        fj = ["finger_joint1", "finger_joint2"] if gripper == "panda" else ["rq_right_driver_joint", "rq_left_driver_joint"]
        self.finger_qpos_adr = np.array([m.joint(j).qposadr[0] for j in fj])
        self.finger_dof_adr = np.array([m.joint(j).dofadr[0] for j in fj])
        self.cup_qpos_adr = m.joint("cup_joint").qposadr[0]
        self.cup_body = m.body("cup").id
        assert cup_shape in ("cup", "cylinder")
        self.cup_shape = cup_shape
        self.cup_geoms = np.where(m.geom_bodyid == self.cup_body)[0]
        cyl = m.geom("cyl").id
        for g in self.cup_geoms:  # enable exactly one shape (collision + visibility)
            on = (g == cyl) == (cup_shape == "cylinder")
            m.geom_contype[g] = m.geom_conaffinity[g] = 1 if on else 0
            m.geom_rgba[g, 3] = 1.0 if on else 0.0
        m.geom_rgba[cyl, :3] = [0.15, 0.3, 0.9]
        if cup_shape == "cylinder":
            r, h = (float(x) for x in m.geom_size[cyl][:2])
            self.cup_half_height = h
            m.body_ipos[self.cup_body] = 0.0
            m.body_iquat[self.cup_body] = [1, 0, 0, 0]
            self.cup_inertia_unit = np.array([(3 * r**2 + (2 * h) ** 2) / 12] * 2 + [r**2 / 2])  # per unit mass
            m.body_mass[self.cup_body] = 0.15
        else:
            self.cup_half_height = 0.045
            self.cup_inertia_unit = m.body_inertia[self.cup_body] / m.body_mass[self.cup_body]
        self.tcp_site = m.site("tcp").id
        self.ctrl_lo, self.ctrl_hi = m.actuator_ctrlrange[:7].T
        if gripper == "panda":
            self.ready_qpos = m.key(READY_KEY).qpos.copy()
            self.ready_ctrl = m.key(READY_KEY).ctrl.copy()
        else:  # no keyframe: solve the side pre-grasp pose by IK (TCP 9 cm in front of the nominal cup)
            cup0 = m.body_pos[self.cup_body]
            self.ready_qpos = m.qpos0.copy()
            self.ready_qpos[self.arm_qpos_adr] = self._solve_ready(cup0 - np.array([0.09, 0.0, 0.0]))
            self.ready_ctrl = np.zeros(m.nu)
            self.ready_ctrl[:7] = self.ready_qpos[self.arm_qpos_adr]

        self.ready_qpos[self.cup_qpos_adr + 2] = TABLE_Z + self.cup_half_height + 0.001
        self.action_space = spaces.Box(-np.inf, np.inf, (8,), np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, (9 + 9 + 7,), np.float64)

    def _solve_ready(self, tcp_pos):
        """Joint configuration with the hand horizontal (R_SIDE) at tcp_pos. The arm is redundant; pick the
        solution whose joint 7 is closest to -1.5 rad so that a +pi wrist roll (rigid-flip baseline) stays in range."""
        best = None
        for q3 in (-0.6, 0.0, 0.6):
            for q5 in (-1.2, 0.0, 1.2):
                q, e = solve_ik(self, tcp_pos, R_SIDE, [0.0, -0.5, q3, -2.2, q5, 3.2, -1.2], iters=500)
                if e < 1e-3 and (best is None or abs(q[6] + 1.5) < abs(best[6] + 1.5)):
                    best = q
        assert best is not None, "no IK solution for the ready pose"
        return best

    # ---- state accessors -------------------------------------------------
    def cup_pose(self):
        a = self.cup_qpos_adr
        return self.data.qpos[a : a + 3].copy(), self.data.qpos[a + 3 : a + 7].copy()

    def tcp_pose(self):
        return self.data.site_xpos[self.tcp_site].copy(), self.data.site_xmat[self.tcp_site].reshape(3, 3).copy()

    def arm_qpos(self):
        return self.data.qpos[self.arm_qpos_adr].copy()

    def gripper_width(self):
        """Finger opening: sum of the finger joint positions (panda: m, robotiq: driver angles, 0 = open)."""
        return float(self.data.qpos[self.finger_qpos_adr].sum())

    def _obs(self):
        p, q = self.cup_pose()
        d = self.data
        return np.concatenate([d.qpos[self.arm_qpos_adr], d.qpos[self.finger_qpos_adr],
                               d.qvel[self.arm_dof_adr], d.qvel[self.finger_dof_adr], p, q])

    # ---- gym API ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m, d, rng = self.model, self.data, self.np_random
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.ready_qpos
        d.ctrl[:] = self.ready_ctrl
        # domain randomization
        mass = rng.uniform(*self.mass_range)
        m.body_mass[self.cup_body] = mass
        m.body_inertia[self.cup_body] = self.cup_inertia_unit * mass
        m.geom_friction[self.cup_geoms, 0] = rng.uniform(*self.friction_range)
        # cup starts upright; xy jitter around the nominal position
        a = self.cup_qpos_adr
        d.qpos[a : a + 2] += rng.uniform(-self.cup_xy_jitter, self.cup_xy_jitter, 2)
        mujoco.mj_forward(m, d)
        for _ in range(100):  # let the cup settle
            mujoco.mj_step(m, d)
        return self._obs(), {"cup_mass": float(m.body_mass[self.cup_body]),
                             "cup_friction": float(m.geom_friction[self.cup_geoms[0], 0])}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.data.ctrl[:7] = np.clip(action[:7], self.ctrl_lo, self.ctrl_hi)
        g = float(np.clip(action[7], 0.0, 1.0))  # 1 = open
        self.data.ctrl[7] = 255.0 * (g if self.gripper == "panda" else 1.0 - g)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        return self._obs(), 0.0, False, False, {}

    def render(self, camera=None):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.render_size)
        if camera is None:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = (0.45, 0.0, 0.85)
            camera.distance, camera.azimuth, camera.elevation = 1.5, 140, -20
        self._renderer.update_scene(self.data, camera)
        return self._renderer.render()
