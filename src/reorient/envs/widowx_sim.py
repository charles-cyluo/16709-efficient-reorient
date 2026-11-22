"""WidowX 250 S + thin cylinder simulation helper (shared by the scripted controllers, the metrics and the RL environment)."""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from reorient import TABLE_Z
from reorient.envs.model import WIDOWX_PAD_HALF, WIDOWX_PALM_BOX, WIDOWX_TCP_X, build_widowx_model

ARM = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
CYL_R, CYL_H, CUP_XY = 0.0225, 0.07, (0.6, 0.0)
OPEN, CLOSED = 0.037, 0.015  # left finger position target range [m]


class Sim:
    """WidowX 250 S + thin cylinder on the table: model construction, IK helpers, state accessors, stepping.

    `dt` is the physics step, `ctrl_dt` the default control period of `step`."""

    def __init__(self, seed, mass, mu, pad_mu=1.0, kp=1000.0, pad_half=None, sphere_r=None, pad_x=None, palm_half=None, cyl_r=CYL_R, cyl_h=CYL_H,
                 dt=0.0005, ctrl_dt=0.02):
        self.cyl_r, self.cyl_h = cyl_r, cyl_h
        self.dt, self.ctrl_dt = dt, ctrl_dt
        self.substeps = round(ctrl_dt / dt)
        self.m = m = build_widowx_model(pad_half or WIDOWX_PAD_HALF, pad_mu, sphere_r, pad_x, palm_half or WIDOWX_PALM_BOX[1])
        self.tcp_x = WIDOWX_TCP_X if pad_x is None else pad_x
        m.opt.timestep = dt
        self.d = d = mujoco.MjData(m)
        self.adr = [m.joint(n).qposadr[0] for n in ARM]
        self.dof = [m.joint(n).dofadr[0] for n in ARM]
        ids = [m.joint(n).id for n in ARM]
        self.lo, self.hi = m.jnt_range[ids, 0], m.jnt_range[ids, 1]
        self.site = m.site("tcp").id
        self.lf = m.joint("left_finger").qposadr[0]
        self.cup_adr = m.joint("cup_joint").qposadr[0]
        a = m.actuator("gripper").id
        m.actuator_gainprm[a, 0], m.actuator_biasprm[a, 1] = kp, -kp
        self.kp = kp
        body = m.body("cup").id
        for g in np.where(m.geom_bodyid == body)[0]:  # only the thin cylinder collides / shows
            on = g == m.geom("cyl").id
            m.geom_contype[g] = m.geom_conaffinity[g] = 1 if on else 0
            m.geom_rgba[g, 3] = 1.0 if on else 0.0
            m.geom_friction[g, 0] = mu
        m.geom_rgba[m.geom("cyl").id, :3] = [0.15, 0.3, 0.9]
        m.geom_size[m.geom("cyl").id, :2] = [cyl_r, cyl_h]
        m.body_ipos[body] = 0.0
        m.body_iquat[body] = [1, 0, 0, 0]
        m.body_mass[body] = mass
        m.body_inertia[body] = mass * np.array([(3 * cyl_r**2 + (2 * cyl_h) ** 2) / 12] * 2 + [cyl_r**2 / 2])
        self.jp, self.jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.scratch = mujoco.MjData(m)
        self.rng = np.random.default_rng(seed)
        self.renderer = None
        self.hook, self.hook_every_substep = None, False
        self.start_offset = None
        self.start_jitter = 0.0  # [m]; the cylinder starts at a fixed position unless --start-jitter is given


    # --- scene configuration (shared by the scripted controllers and the RL environment) ------------------
    def apply_contact_model(self, torsion=0.004, noslip=10):
        """Torsional friction on the pads and the cylinder (the brake of the swing) and noslip iterations."""
        m = self.m
        m.opt.noslip_iterations = noslip
        for g in (m.geom("left_pad").id, m.geom("right_pad").id, m.geom("cyl").id):
            m.geom_condim[g] = 4
            m.geom_friction[g, 1] = torsion

    def set_cylinder(self, mass, com_z=0.0, data=None):
        """Set mass, inertia (about the centre) and a COM offset along the axis; refreshes MuJoCo's derived constants (mj_setConst), which set
        the contact softness. Without it the contact behaviour depends on the cup the model was compiled with."""
        m = self.m
        body = m.body("cup").id
        m.body_mass[body] = mass
        m.body_inertia[body] = mass * np.array([(3 * self.cyl_r**2 + (2 * self.cyl_h) ** 2) / 12] * 2 + [self.cyl_r**2 / 2])
        m.body_ipos[body] = [0.0, 0.0, com_z]
        mujoco.mj_setConst(m, self.scratch if data is None and self.scratch is not None else (data if data is not None else self.d))

    # --- kinematics -------------------------------------------------------
    def _err_jac(self, data, pos, rot):
        mujoco.mj_kinematics(self.m, data)
        mujoco.mj_comPos(self.m, data)
        mujoco.mj_jacSite(self.m, data, self.jp, self.jr, self.site)
        J = np.vstack([self.jp, self.jr])[:, self.dof]
        p, Rc = data.site_xpos[self.site], data.site_xmat[self.site].reshape(3, 3)
        return np.concatenate([pos - p, (R.from_matrix(rot) * R.from_matrix(Rc).inv()).as_rotvec()]), J

    def solve_ik(self, pos, rot, q0, iters=500):
        s, q = self.scratch, np.array(q0, float)
        for _ in range(iters):
            s.qpos[self.adr] = q
            e, J = self._err_jac(s, pos, rot)
            q = np.clip(q + 0.5 * J.T @ np.linalg.solve(J @ J.T + 4e-4 * np.eye(6), e), self.lo + 0.01, self.hi - 0.01)
        return q, float(np.linalg.norm(e))

    def servo(self, q_cmd, pos, rot, gain=0.2, max_dq=0.08):
        """Integrating differential IK: nudges the command with the measured TCP error (removes gravity sag)."""
        s = self.scratch
        s.qpos[:] = self.d.qpos
        e, J = self._err_jac(s, pos, rot)
        dq = np.clip(J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(6), gain * e), -max_dq, max_dq)
        q = self.d.qpos[self.adr]
        return np.clip(np.clip(q_cmd + dq, self.lo, self.hi), q - 0.3, q + 0.3)

    # --- state ------------------------------------------------------------
    def tcp(self):
        return self.d.site_xpos[self.site].copy()

    def cup(self):
        a = self.cup_adr
        return self.d.qpos[a : a + 3].copy(), self.d.qpos[a + 3 : a + 7].copy()

    def tilt(self):
        ax = R.from_quat(self.cup()[1][[1, 2, 3, 0]]).as_matrix()[:, 2]
        return float(np.degrees(np.arctan2(ax[0], ax[2])))  # 0 up, +90 forward, +-180 inverted

    def grip_ctrl(self, force):
        """Finger target giving ~force N closing force at the current finger position (open = 0.037)."""
        return float(np.clip(self.d.qpos[self.lf] - force / self.kp, CLOSED, OPEN))

    def step(self, q_cmd, grip, n=None):
        self.d.ctrl[:6] = q_cmd
        self.d.ctrl[6] = grip
        for _ in range(self.substeps if n is None else n):
            mujoco.mj_step(self.m, self.d)
            if self.hook is not None and self.hook_every_substep:
                self.hook()
        if self.hook is not None and not self.hook_every_substep:
            self.hook()

    def reset(self, q_ready):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[self.adr] = q_ready
        d.qpos[self.lf], d.qpos[self.lf + 1] = OPEN, -OPEN
        d.qpos[self.cup_adr : self.cup_adr + 3] = [*CUP_XY, TABLE_Z + self.cyl_h + 0.001]
        d.qpos[self.cup_adr : self.cup_adr + 2] += self.rng.uniform(-self.start_jitter, self.start_jitter, 2) if self.start_offset is None else self.start_offset
        d.ctrl[:6], d.ctrl[6] = q_ready, OPEN
        mujoco.mj_forward(m, d)
        for _ in range(200):
            mujoco.mj_step(m, d)

    def pad_forces(self):
        """Normal force [N] on the cylinder from the left and right pad (0 if not touching)."""
        out = {"left_pad": 0.0, "right_pad": 0.0}
        f = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            names = {self.m.geom(c.geom1).name, self.m.geom(c.geom2).name}
            for n in out:
                if n in names and "cyl" in names:
                    mujoco.mj_contactForce(self.m, self.d, i, f)
                    out[n] += f[0]
        return out

    def render(self, cam):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, 480, 640)
        cam.lookat[:] = self.tcp() + np.array([0, 0, 0.03])
        self.renderer.update_scene(self.d, cam)
        return self.renderer.render()
