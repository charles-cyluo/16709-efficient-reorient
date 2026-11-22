"""Exploratory pivot probe with the WidowX 250 S (NOT a baseline): side-pinch the thin cylinder near its bottom end,
lift, loosen the grip so it swings about the fingertips, and close the grip again right after it has flipped.

    uv run python scripts/probe_widowx.py --seed 0 --force 0.3 --video outputs/videos/pivot_probe_widowx.mp4
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

from reorient import TABLE_Z
from reorient.envs.model import WIDOWX_PAD_HALF, WIDOWX_PALM_BOX, build_widowx_model
from reorient.policies.cartesian import R_SIDE

from reorient.envs.widowx_sim import ARM, CLOSED, CUP_XY, CYL_H, CYL_R, OPEN  # noqa: E402
DT = float(os.environ.get("PROBE_DT", "0.0005"))  # physics step; 20 ms control period
CTRL = float(os.environ.get("PROBE_CTRL", "0.02"))  # grip-control period during the swing (the arm is still servoed at 20 ms)
SUBSTEPS = round(CTRL / DT)
ARM_EVERY = max(1, round(0.02 / CTRL))


from reorient.envs.widowx_sim import Sim as _Sim  # noqa: E402


class Sim(_Sim):
    """Package Sim with the physics step / control period of this script (PROBE_DT, PROBE_CTRL)."""

    def __init__(self, *args, **kw):
        kw.setdefault("dt", DT)
        kw.setdefault("ctrl_dt", CTRL)
        super().__init__(*args, **kw)


class GripFeedback:
    """Grip force from feedback on the swing. The pads' torsional friction brakes the rotation (torque cap ~ force), so the force is
    raised when the object is ahead of a reference tilt trajectory / swinging too fast and lowered when it lags.
    Unwrapped tilt: 0 upright .. ~160 deg hanging inverted."""

    def __init__(self, sim, a):
        self.sim, self.a = sim, a
        self.t0 = None
        self.prev = None
        self.last = a.pinch_force

    def unwrapped(self):
        tl = self.sim.tilt()
        return tl if tl > -90 else tl + 360.0

    def ref(self, t):
        u = min(1.0, max(0.0, t / self.a.fb_time))
        return self.a.fb_end * (10 * u**3 - 15 * u**4 + 6 * u**5), self.a.fb_end * (30 * u**2 - 60 * u**3 + 30 * u**4) / self.a.fb_time

    def force(self):
        sim, a = self.sim, self.a
        now = sim.d.time
        th = self.unwrapped()
        if self.t0 is None:
            self.t0, self.prev = now, (now, th)
        dt = max(now - self.prev[0], 1e-6)
        om = (th - self.prev[1]) / dt
        self.prev = (now, th)
        th_ref, om_ref = self.ref(now - self.t0)
        if a.fb_brake_only:  # only squeeze harder when ahead of / faster than the reference; never loosen below f0
            f = a.fb_f0 + a.fb_kp * max(0.0, th - th_ref) + a.fb_kd * max(0.0, om - om_ref)
        else:
            f = a.fb_f0 + a.fb_kp * (th - th_ref) + a.fb_kd * (om - om_ref)
        self.last = float(np.clip(f, a.fb_fmin, a.fb_fmax))
        if a.verbose:
            print("  fb t %.2f tilt %6.1f ref %6.1f rate %7.1f (ref %6.1f) force %.2f cup-tcp %.3f" % (now - self.t0, th, th_ref, om, om_ref, self.last, float(np.linalg.norm(sim.cup()[0] - sim.tcp()))))
        return self.last


def axis_x(sim):
    return float(R.from_quat(sim.cup()[1][[1, 2, 3, 0]]).as_matrix()[:, 2][0])


def place_and_score(sim, a, q, start_xy, go):
    """Hang -> (straighten only if leaning) -> (carry/lower only if needed) -> release -> lift straight up -> retract -> score.
    If the object has fallen or slipped out at any point the hand stops pushing: it opens and goes straight up."""
    force = a.hold_force
    grip = lambda: sim.grip_ctrl(force)
    dt = 0.02
    sub = round(dt / DT)
    rest_z = TABLE_Z + a.cyl_h
    cb = sim.m.body("cup").id
    table = sim.m.geom("table").id

    def lean_deg():  # angle of the object's axis from vertical
        az = R.from_quat(sim.cup()[1][[1, 2, 3, 0]]).as_matrix()[:, 2][2]
        return float(np.degrees(np.arccos(np.clip(abs(az), 0, 1))))

    def fallen():
        return lean_deg() > 45.0 or float(np.linalg.norm(sim.cup()[0] - sim.tcp())) > 0.12

    def trace(tag, k):
        if a.verbose and k % 5 == 0:
            print("  %s t %.1f tilt %4.0f lean %4.1f cup %s tcp %s" % (tag, k * dt, sim.tilt(), lean_deg(), np.round(sim.cup()[0], 3), np.round(sim.tcp(), 3)))

    def move(carrot, q, grip_cmd, steps=1):
        for _ in range(steps):
            q = sim.servo(q, carrot, R_SIDE)
            sim.step(q, grip_cmd(), sub)
        return q

    vel = np.zeros(3)

    def goto(goal, q, grip_cmd, vmax=0.10, amax=0.4):
        """Move the carrot to `goal` with an acceleration-limited trapezoid velocity profile (no jerk at start/stop)."""
        nonlocal carrot, vel
        for _ in range(int(10.0 / dt)):
            d = goal - carrot
            dist = float(np.linalg.norm(d))
            if dist < 1e-4 and float(np.linalg.norm(vel)) < 1e-3:
                break
            v_des = d / max(dist, 1e-9) * min(vmax, np.sqrt(2 * amax * dist))
            dv = v_des - vel
            n = float(np.linalg.norm(dv))
            vel = vel + dv / max(n, 1e-9) * min(n, amax * dt)
            carrot = carrot + vel * dt
            q = move(carrot, q, grip_cmd)
        vel = np.zeros(3)
        return q

    def abort(why):
        """Open the gripper in place and go straight up: nothing is pushed along the table."""
        nonlocal q, carrot
        print("ABORT: %s -> open, lift straight up" % why)
        for _ in range(int(0.5 / dt)):
            q = move(carrot, q, lambda: OPEN)
        q = goto(carrot + np.array([0, 0, 0.08]), q, lambda: OPEN, vmax=0.10)
        for _ in range(int(0.5 / dt)):
            q = move(carrot, q, lambda: OPEN)
        print("RESULT fail: %s (cup z %.3f, lean %.0f deg)" % (why, sim.cup()[0][2], lean_deg()))
        return False

    # 1. wait until the object hangs inverted and is nearly still
    prev, t = sim.tilt(), 0.0
    carrot = sim.tcp()
    while t < 3.0:
        sim.step(q, grip(), sub)
        t += dt
        tl = sim.tilt()
        rate = abs(tl - prev) / dt if abs(tl) < 180 else 0.0
        prev = tl
        if abs(tl) > 125 and rate < 20:
            break
    print("hang: t %.2f s, tilt %.0f, lean %.1f, cup z %.3f, held=%s" % (t, sim.tilt(), lean_deg(), sim.cup()[0][2], np.linalg.norm(sim.cup()[0] - sim.tcp()) < 0.1))
    if abs(sim.tilt()) < 125:
        print("RESULT fail: no inverted hang")
        return False
    if fallen():
        return abort("object lost right after the swing")

    # 2. straighten only if it leans more than --straighten-lean deg: hand forward over the lower end; stop when upright, stalled or 6 cm travelled
    carrot = sim.tcp()
    x_start = carrot[0]
    if lean_deg() > a.straighten_lean:
        best, t_best = lean_deg(), 0.0
        for k in range(int(4.0 / dt)):
            ax = axis_x(sim)
            carrot = carrot + np.array([np.clip(0.8 * ax * 0.095, -0.03 * dt, 0.03 * dt), 0.0, 0.0])
            q = move(carrot, q, lambda: sim.grip_ctrl(min(force, a.straighten_force)))
            trace("straighten", k)
            if fallen():
                return abort("object fell while straightening")
            ln = lean_deg()
            if ln < best - 0.5:
                best, t_best = ln, k * dt
            if ln < 6.0 or k * dt - t_best > 0.6 or abs(carrot[0] - x_start) > 0.06:
                break
        print("straightened: lean %.1f deg, hand moved %.1f cm" % (lean_deg(), 100 * (carrot[0] - x_start)))
    else:
        print("straighten skipped: lean %.1f deg is already within tolerance" % lean_deg())

    # 3. carry / lower only if the object is far from its start xy or hangs above the table
    if np.linalg.norm(start_xy - sim.cup()[0][:2]) > 0.03:
        z0 = carrot[2]
        for k in range(int(0.6 / dt)):
            carrot[2] = min(carrot[2] + 0.04 * dt, z0 + 0.02)
            q = move(carrot, q, grip)
        for k in range(int(4.0 / dt)):
            err = start_xy - sim.cup()[0][:2]
            n = np.linalg.norm(err)
            carrot[:2] = carrot[:2] + err / max(n, 1e-9) * min(n, 0.06 * dt)
            q = move(carrot, q, grip)
            trace("carry", k)
            if fallen():
                return abort("object fell while carrying")
            if n < 0.004:
                break
    if sim.cup()[0][2] > rest_z + 0.003:
        for k in range(int(3.0 / dt)):
            carrot[2] -= 0.03 * dt
            q = move(carrot, q, grip)
            trace("lower", k)
            if fallen():
                return abort("object fell while lowering")
            if sim.cup()[0][2] < rest_z + 0.0015:
                break
    print("placed: lean %.1f deg, cup %s, xy err %.3f" % (lean_deg(), np.round(sim.cup()[0], 3), np.linalg.norm(sim.cup()[0][:2] - start_xy)))

    # 4. release, lift the hand straight up (nothing dragged along the cup), then retract back
    def touching():
        nm = lambda g: sim.m.geom(g).name or "mesh:" + sim.m.body(sim.m.geom_bodyid[g]).name
        return sorted({(nm(c.geom1) if sim.m.geom_bodyid[c.geom1] != cb else nm(c.geom2)) for c in sim.d.contact[: sim.d.ncon]
                       if cb in (sim.m.geom_bodyid[c.geom1], sim.m.geom_bodyid[c.geom2]) and table not in (c.geom1, c.geom2)})

    def robot_contacts():
        return len(touching())

    # gentle release: ease the grip off until the object stands by itself, then open the fingers slowly
    for i in range(int(a.release_ease / dt)):
        u = (i + 1) * dt / a.release_ease
        q = move(carrot, q, lambda: sim.grip_ctrl(max(0.25, force * (1 - u))))
    for i in range(int(2.5 / dt)):  # the hand only moves once nothing of it touches the object any more
        u = min(1.0, (i + 1) * dt / a.release_open_time)
        q = move(carrot, q, lambda: sim.d.qpos[sim.lf] + (OPEN - sim.d.qpos[sim.lf]) * u if u < 1.0 else OPEN)
        if i * dt >= a.release_open_time and robot_contacts() == 0:
            break
        if a.verbose and i % 5 == 0:
            print("  release t %.2f cup %s lean %.1f fingers q L %.4f R %.4f touching %s" % (i * dt, np.round(sim.cup()[0], 3), lean_deg(), sim.d.qpos[sim.lf], sim.d.qpos[sim.lf + 1], touching()))
    if fallen():
        return abort("object fell on release")
    def contact_force():  # largest normal force between the object and any robot geom [N]
        f, best = np.zeros(6), 0.0
        for i, c in enumerate(sim.d.contact[: sim.d.ncon]):
            b1, b2 = sim.m.geom_bodyid[c.geom1], sim.m.geom_bodyid[c.geom2]
            if cb in (b1, b2) and table not in (c.geom1, c.geom2):
                mujoco.mj_contactForce(sim.m, sim.d, i, f)
                best = max(best, float(f[0]))
        return best

    if robot_contacts() > 0 and contact_force() < 0.5:  # resting lightly against the hand (the palm sits behind it): back off a little
        q = goto(carrot + np.array([-0.015, 0, 0]), q, lambda: OPEN, vmax=0.03)
        for _ in range(int(0.3 / dt)):
            q = move(carrot, q, lambda: OPEN)
    if robot_contacts() > 0:
        print("RESULT fail: object still touches the hand after opening (%s, %.2f N); hand not moved further (lean %.0f deg, cup z %.3f)" % (touching(), contact_force(), lean_deg(), sim.cup()[0][2]))
        return False
    q = goto(carrot + np.array([0, 0, a.release_lift]), q, lambda: OPEN, vmax=0.08)
    q = goto(carrot + np.array([-0.10, 0, 0]), q, lambda: OPEN, vmax=0.12)
    for _ in range(int(0.5 / dt)):  # let the arm come to rest, still servoed
        q = move(carrot, q, lambda: OPEN)
    if a.verbose:
        print("  retract done: tcp %s |qvel arm| %.4f" % (np.round(sim.tcp(), 4), float(np.linalg.norm(sim.d.qvel[sim.dof]))))

    # 5. score only once nothing of the robot touches the object and it has stood at rest on its own
    dof = sim.m.joint("cup_joint").dofadr[0]
    rest_t, waited = 0.0, 0.0
    while waited < 6.0 and rest_t < 1.0:
        sim.step(q, OPEN, sub)
        waited += dt
        v = sim.d.qvel[dof : dof + 6]
        still = np.linalg.norm(v[:3]) < 0.01 and np.linalg.norm(v[3:]) < 0.1
        rest_t = rest_t + dt if (robot_contacts() == 0 and still) else 0.0
    print("settled: robot contacts %d, waited %.1f s, at rest for %.1f s" % (robot_contacts(), waited, rest_t))
    p = sim.cup()[0]
    axis = R.from_quat(sim.cup()[1][[1, 2, 3, 0]]).as_matrix()[:, 2]
    tilt_inv = float(np.degrees(np.arccos(np.clip(-axis[2], -1, 1))))
    err = float(np.linalg.norm(p[:2] - start_xy))
    ok = rest_t >= 1.0 and tilt_inv < 15 and err < 0.03 and p[2] < TABLE_Z + a.cyl_h + 0.015
    print("RESULT %s: tilt from inverted %.1f deg, placement error %.3f m, cup z %.3f" % ("SUCCESS" if ok else "fail", tilt_inv, err, p[2]))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=-0.016, help="TCP offset along approach vs the cylinder axis")
    ap.add_argument("--dz", type=float, default=-0.045, help="TCP height rel. to cylinder centre")
    ap.add_argument("--force", type=float, default=0.3, help="grip force during the swing [N]")
    ap.add_argument("--hold-force", type=float, default=4.0, help="grip force after the flip [N]")
    ap.add_argument("--pinch-force", type=float, default=4.0)
    ap.add_argument("--reclamp-tilt", type=float, default=150.0, help="close the grip when |tilt| exceeds this; 0 = never")
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--mass", type=float, default=0.15)
    ap.add_argument("--mu", type=float, default=1.0, help="cylinder friction")
    ap.add_argument("--pad-mu", type=float, default=1.0, help="finger contact-sphere friction")
    ap.add_argument("--pad-half", type=float, nargs=3, default=None, help="pad box half-sizes x y z [m]")
    ap.add_argument("--sphere-r", type=float, default=None, help="use a round pad of this radius instead of the box")
    ap.add_argument("--pad-x", type=float, default=None, help="pad / TCP position along the gripper x axis [m]")
    ap.add_argument("--palm-half", type=float, nargs=3, default=None, help="palm box half-sizes x y z [m] (front face at 2*x)")
    ap.add_argument("--cyl-r", type=float, default=CYL_R, help="cylinder radius [m]")
    ap.add_argument("--cyl-h", type=float, default=CYL_H, help="cylinder half height [m]")
    ap.add_argument("--contacts", type=float, default=0, help="print what the cylinder touches once |tilt| exceeds this [deg]")
    ap.add_argument("--rate-target", type=float, default=0, help="swing speed to hold [deg/s]; 0 = constant force")
    ap.add_argument("--rate-gain", type=float, default=0.02, help="N per (deg/s) per second of rate error")
    ap.add_argument("--fmin", type=float, default=1.0)
    ap.add_argument("--fmax", type=float, default=8.0)
    ap.add_argument("--torsion", type=float, default=0.0, help="torsional friction coefficient [m] of pads + cylinder (condim 4); 0 = condim 3")
    ap.add_argument("--randomize", action="store_true", help="randomize mass (0.10-0.30 kg), cylinder friction (0.6-1.2) and centre of mass per seed")
    ap.add_argument("--rand-only", default="", help="comma list of mass,mu,com: randomize only these (with --randomize)")
    ap.add_argument("--com-axial", type=float, default=0.015, help="COM offset range along the cylinder axis [m] (+-)")
    ap.add_argument("--com-lateral", type=float, default=0.005, help="COM offset range across the axis [m] (+-)")
    ap.add_argument("--start-jitter", type=float, default=0.0, help="random cylinder start offset, uniform +- this [mm] (default 0: fixed start)")
    ap.add_argument("--start-offset", type=float, nargs=2, default=None, help="fixed cylinder start offset x y [mm] instead of the random +-10 mm")
    ap.add_argument("--feedback", action="store_true", help="grip force from feedback on the swing (tilt reference tracking)")
    ap.add_argument("--fb-f0", type=float, default=3.0)
    ap.add_argument("--fb-kp", type=float, default=0.03, help="N per deg of tilt ahead of the reference")
    ap.add_argument("--fb-kd", type=float, default=0.004, help="N per deg/s of rate ahead of the reference")
    ap.add_argument("--fb-brake-only", action="store_true")
    ap.add_argument("--fb-fmin", type=float, default=1.0)
    ap.add_argument("--fb-fmax", type=float, default=8.0)
    ap.add_argument("--fb-time", type=float, default=1.0, help="time to reach the final tilt [s], from the start of the lift")
    ap.add_argument("--fb-end", type=float, default=160.0, help="final reference tilt [deg]")
    ap.add_argument("--straighten-lean", type=float, default=3.0, help="straighten only if the object leans more than this [deg]")
    ap.add_argument("--straighten-force", type=float, default=1.6, help="grip force while straightening (the object rests on the table) [N]")
    ap.add_argument("--release-ease", type=float, default=0.4, help="time to ease the grip force off before opening [s]")
    ap.add_argument("--release-open-time", type=float, default=0.8, help="time to open the fingers [s]")
    ap.add_argument("--release-lift", type=float, default=0.06, help="straight-up lift after releasing, before retracting [m]")
    ap.add_argument("--place", action="store_true", help="after the swing: straighten, set down at the start xy, release, retract, score")
    ap.add_argument("--noslip", type=int, default=0)
    ap.add_argument("--impratio", type=float, default=None)
    ap.add_argument("--cone", default=None, choices=["pyramidal", "elliptic"])
    ap.add_argument("--solref", type=float, nargs=2, default=None, help="pad + cylinder contact solref")
    ap.add_argument("--solimp", type=float, nargs=5, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", type=float, default=3.0)
    ap.add_argument("--video", default=None)
    a = ap.parse_args()

    sim = Sim(a.seed, a.mass, a.mu, a.pad_mu, pad_half=a.pad_half, sphere_r=a.sphere_r, pad_x=a.pad_x, palm_half=a.palm_half, cyl_r=a.cyl_r, cyl_h=a.cyl_h)
    m = sim.m
    sim.start_jitter = a.start_jitter / 1000.0
    if a.start_offset is not None:
        sim.start_offset = np.array(a.start_offset) / 1000.0
    if a.randomize:
        rng = np.random.default_rng(1000 + a.seed)
        body = m.body("cup").id
        what = set(a.rand_only.split(",")) if a.rand_only else {"mass", "mu", "com"}
        mass = rng.uniform(0.10, 0.30) if "mass" in what else a.mass
        m.body_mass[body] = mass
        m.body_inertia[body] = mass * np.array([(3 * a.cyl_r**2 + (2 * a.cyl_h) ** 2) / 12] * 2 + [a.cyl_r**2 / 2])
        mu_r = rng.uniform(0.6, 1.2)
        if "mu" in what:
            m.geom_friction[m.geom("cyl").id, 0] = mu_r
        com = np.array([rng.uniform(-a.com_lateral, a.com_lateral), rng.uniform(-a.com_lateral, a.com_lateral),
                        rng.uniform(-a.com_axial, a.com_axial)])
        if "com" not in what:
            com = com * 0
        m.body_ipos[body] = com
        print("randomized: mass %.3f kg, cylinder mu %.2f, COM offset (x y axial) %s mm" % (mass, m.geom_friction[m.geom('cyl').id, 0], np.round(com * 1000, 1)))
    m.opt.noslip_iterations = a.noslip
    if a.impratio is not None:
        m.opt.impratio = a.impratio
    if a.cone:
        m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL if a.cone == "pyramidal" else mujoco.mjtCone.mjCONE_ELLIPTIC
    for g in [m.geom("left_pad").id, m.geom("right_pad").id, m.geom("cyl").id]:
        if a.torsion:
            m.geom_condim[g] = 4
            m.geom_friction[g, 1] = a.torsion
        if a.solref:
            m.geom_solref[g] = a.solref
        if a.solimp:
            m.geom_solimp[g] = a.solimp
        if a.margin is not None:
            m.geom_margin[g] = a.margin
    sim.reset([0, 0.7, 0.6, 0, -1.3, 0])  # places the cylinder (with jitter); the arm pose is redone below
    cx, cy = sim.cup()[0][:2]
    start_xy = np.array([cx, cy])
    grasp = np.array([cx + a.dx, cy, TABLE_Z + a.cyl_h + a.dz])
    pre = grasp - np.array([0.04, 0, 0])
    q_gr, e2 = sim.solve_ik(grasp, R_SIDE, [0, 0.7, 0.6, 0, -1.3, 0])
    q_pre, e = sim.solve_ik(pre, R_SIDE, q_gr)
    assert e < 1e-3 and e2 < 1e-3, (e, e2)
    cup_qpos = sim.d.qpos[sim.cup_adr : sim.cup_adr + 7].copy()
    sim.reset(q_pre)
    sim.d.qpos[sim.cup_adr : sim.cup_adr + 7] = cup_qpos
    mujoco.mj_forward(sim.m, sim.d)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.6, 150, -15
    frames, n = [], 0
    last_t = [-1.0]

    def snap():  # called after every physics call: one frame per 40 ms of simulated time, whatever the phase
        if a.video and sim.d.time - last_t[0] >= 0.04 - 1e-9:
            last_t[0] = sim.d.time
            frames.append(sim.render(cam))

    sim.hook = snap

    def go(target, grip_fn, secs, tol=None, rot=R_SIDE, fast_grip=False):
        nonlocal q, n
        carrot = sim.tcp()
        for _ in range(int(secs / 0.02)):
            d = target - carrot
            carrot = carrot + d * min(1.0, 0.25 * 0.02 / max(np.linalg.norm(d), 1e-9))
            q = sim.servo(q, carrot, rot)
            if fast_grip:  # grip loop at CTRL, arm servo still at 20 ms
                for _ in range(max(1, round(0.02 / CTRL))):
                    sim.step(q, grip_fn(), SUBSTEPS)
            else:
                sim.step(q, grip_fn(), round(0.02 / DT))
            n += 1

    q = sim.d.qpos[sim.adr].copy()
    go(pre, lambda: OPEN, 0.5)
    go(grasp, lambda: OPEN, 2.0)
    go(grasp, lambda: sim.grip_ctrl(a.pinch_force), 0.6)
    p0 = sim.cup()[0]
    print("pinch: finger gap %.4f, cup tilt %.1f, cup-tcp %s" % (2 * sim.d.qpos[sim.lf], sim.tilt(), np.round(p0 - sim.tcp(), 3)))
    fb = GripFeedback(sim, a) if a.feedback else None
    go(grasp + [0, 0, a.lift], (lambda: sim.grip_ctrl(fb.force())) if fb else (lambda: sim.grip_ctrl(a.pinch_force)), 1.0, fast_grip=bool(fb))
    if fb:
        print("feedback: force at end of lift %.2f N, tilt %.0f" % (fb.last, sim.tilt()))
    print("after lift: tcp z %.3f, cup z %.3f, finger q %.4f, tilt %.1f" % (sim.tcp()[2], sim.cup()[0][2], sim.d.qpos[sim.lf], sim.tilt()))
    if a.place:
        if fb:
            a.hold_force = fb.last
        place_and_score(sim, a, q, start_xy, go)
        if a.video:
            os.makedirs(os.path.dirname(a.video) or ".", exist_ok=True)
            imageio.mimsave(a.video, frames, fps=25)
            print("wrote", a.video, len(frames), "frames")
        return
    # swing: loose grip, arm still
    force, reclamped, trace = a.force, False, []
    prev_tilt, omega_f = sim.tilt(), 0.0
    carrot = sim.tcp()
    for k in range(int(a.time / CTRL)):
        if a.reclamp_tilt and not reclamped and abs(sim.tilt()) > a.reclamp_tilt:
            force, reclamped = a.hold_force, True
        if a.rate_target and not reclamped:  # feedback on the swing speed: squeeze harder when it swings too fast
            tl = sim.tilt()
            omega = (tl - prev_tilt) / CTRL
            prev_tilt = tl
            omega_f = 0.7 * omega_f + 0.3 * omega
            force = float(np.clip(force + a.rate_gain * (omega_f - a.rate_target) * CTRL, a.fmin, a.fmax))
        if k % ARM_EVERY == 0:
            q = sim.servo(q, carrot, R_SIDE)
        sim.step(q, sim.grip_ctrl(force))
        if a.contacts and abs(sim.tilt()) > a.contacts:
            nm = lambda g: sim.m.geom(g).name or "mesh:" + sim.m.body(sim.m.geom_bodyid[g]).name
            cb = sim.m.body("cup").id
            hits = sorted({(nm(c.geom1) if sim.m.geom_bodyid[c.geom1] != cb else nm(c.geom2)) for c in sim.d.contact[: sim.d.ncon]
                           if cb in (sim.m.geom_bodyid[c.geom1], sim.m.geom_bodyid[c.geom2])})
            print("  t %.2f tilt %4.0f cup z %.3f contacts with: %s" % (k * CTRL, sim.tilt(), sim.cup()[0][2], hits))
        if k % round(0.1 / CTRL) == 0:
            trace.append(round(sim.tilt()))
            if a.verbose:
                pf = sim.pad_forces()
                print("t %.1f tilt %4.0f cup z %.3f cmd %.1f N pad N: L %.2f R %.2f finger q %.4f" % (k * CTRL, sim.tilt(), sim.cup()[0][2], force, pf["left_pad"], pf["right_pad"], sim.d.qpos[sim.lf]))
    dist = float(np.linalg.norm(sim.cup()[0] - sim.tcp()))
    print("tilt every 0.1 s:", trace)
    print("re-clamped:", reclamped, "| object %s the fingers (distance %.3f m, cup z %.3f)" % ("in" if dist < 0.08 else "LEFT", dist, sim.cup()[0][2]))
    if a.video:
        os.makedirs(os.path.dirname(a.video) or ".", exist_ok=True)
        imageio.mimsave(a.video, frames, fps=25)
        print("wrote", a.video, len(frames), "frames")


if __name__ == "__main__":
    main()
