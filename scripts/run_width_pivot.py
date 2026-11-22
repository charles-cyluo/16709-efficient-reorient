"""Width-controlled pivot on the WidowX (experiment, stock Panda hand not used): IK for the arm, finger *width* position control for the grip.

Sequence (all hand motions are IK targets with an acceleration-limited profile):
  1. approach and close on the bottom part of the cylinder; the closing width is measured (finger position at contact).
  2. lift with a firm grip (width target a little tighter than the contact width -> force = kp * delta).
  3. ease the width off (smaller delta, less friction): the cylinder rotates about the pads under gravity.
  4. when the tilt gets close to 180 deg (predicted with the measured rate), close the width again to stop the rotation.
  5. from the pose of the now static cylinder find its lowest point; move the hand so that this point is above the contact point of the
     final pose (start xy on the table), lower until the lowest point touches the table.
  6. open the width while retracting straight back (the retract starts once the fingers have cleared the cylinder), then wait until the
     cylinder stands at rest on its own with no robot contact, and score it.

    PROBE_DT=0.0005 uv run python scripts/run_width_pivot.py --seed 0 --mass 0.2 --video outputs/videos/width_pivot_mass200.mp4

The grip is the only thing that depends on the object: nothing here uses the mass.  Contact model (as in the other WidowX probes):
round pads (10 mm), torsional friction 0.004 m on pads and cylinder, noslip 10.
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
_mesa = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"  # NVIDIA EGL can hang headless; see docs
if os.path.exists(_mesa):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _mesa)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from probe_widowx import CLOSED, DT, OPEN, Sim  # noqa: E402  (sets the headless GL env vars)
from reorient import TABLE_Z  # noqa: E402
from reorient.eval.metrics import EpisodeMetrics  # noqa: E402
from reorient.policies.cartesian import R_SIDE  # noqa: E402

CTRL = 0.002  # grip / monitoring loop [s]; the arm servo runs every 20 ms
ARM_DT = 0.02


class WidthPivot:
    def __init__(self, sim, a, start_xy):
        self.sim, self.a, self.start_xy = sim, a, np.asarray(start_xy)
        self.q = sim.d.qpos[sim.adr].copy()
        self.carrot = sim.tcp()
        self.vel = np.zeros(3)
        self.metrics = None  # optional EpisodeMetrics; started at pickup (end of the lift)
        self.rot = R_SIDE  # hand orientation target (pitched a few degrees about the finger axis after the lift, see tilt_hand)
        self.width = OPEN  # finger position target [m] (the left finger; the right one follows through the equality constraint)
        self.q_contact = None
        self.log = []

    # ---- low level ---------------------------------------------------------
    def tick(self, n_grip=None):
        """One arm period (20 ms) with the grip loop at CTRL: arm IK servo once, gripper target `self.width` applied throughout."""
        sim = self.sim
        self.q = sim.servo(self.q, self.carrot, self.rot)
        for _ in range(round(ARM_DT / CTRL)):
            w = self.width() if callable(self.width) else self.width
            sim.step(self.q, w, round(CTRL / DT))

    def goto(self, goal, vmax=0.10, amax=0.4, stop=None):
        """Move the carrot to `goal` with a trapezoid velocity profile; `stop()` can end the move early (returns True)."""
        dt = ARM_DT
        for _ in range(int(12.0 / dt)):
            d = goal - self.carrot
            dist = float(np.linalg.norm(d))
            if dist < 2e-4 or (dist <= float(np.linalg.norm(self.vel)) * dt and float(np.linalg.norm(self.vel)) < 0.02):
                self.carrot = goal.copy()  # arrived (snap the last fraction of a step)
                self.tick()
                break
            v_des = d / max(dist, 1e-9) * min(vmax, np.sqrt(2 * amax * dist))
            dv = v_des - self.vel
            n = float(np.linalg.norm(dv))
            self.vel = self.vel + dv / max(n, 1e-9) * min(n, amax * dt)
            self.carrot = self.carrot + self.vel * dt
            self.tick()
            if stop is not None and stop():
                self.vel = np.zeros(3)
                return True
        self.vel = np.zeros(3)
        return False

    def tilt_hand(self, deg, secs=0.6):
        """Pitch the hand about the finger-opening axis (world y) by `deg` (positive: the cup's top moves forward), about the TCP, smoothly.
        The grip is firm, so the cylinder tilts with the hand; releasing the width a little then starts a forward rotation."""
        target = R.from_euler("y", np.radians(deg)).as_matrix() @ R_SIDE
        n = max(1, int(secs / ARM_DT))
        for i in range(1, n + 1):
            u = i / n
            u = 10 * u**3 - 15 * u**4 + 6 * u**5
            self.rot = R.from_euler("y", np.radians(deg * u)).as_matrix() @ R_SIDE
            self.tick()
        self.rot = target
        self.hold(self.a.settle_time)

    def hold(self, secs):
        for _ in range(int(secs / ARM_DT)):
            self.tick()

    # ---- object state ------------------------------------------------------
    def tilt(self):
        tl = self.sim.tilt()
        return tl if tl > -90 else tl + 360.0  # unwrapped: 0 upright ... 180 inverted

    def axis(self):
        return R.from_quat(self.sim.cup()[1][[1, 2, 3, 0]]).as_matrix()[:, 2]

    def lean(self):
        return float(np.degrees(np.arccos(np.clip(abs(self.axis()[2]), 0, 1))))

    def _lower_end(self):
        c, ax = self.sim.cup()[0], self.axis()
        h = self.a.cyl_h
        return c + ax * h if (c + ax * h)[2] < (c - ax * h)[2] else c - ax * h

    def lowest_point(self):
        """Centre of the lower end face: the point that is aligned with the contact point of the final pose (they coincide in the
        upright pose; for a tilted cylinder the rim point that touches the table is 2.25 cm away, see lowest_z)."""
        return self._lower_end()

    def lowest_z(self):
        """Height of the true lowest point (a rim point of the lower end face if the axis is tilted)."""
        ax = self.axis()
        d = -np.array([0, 0, 1.0]) + ax[2] * ax
        n = float(np.linalg.norm(d))
        return float(self._lower_end()[2] + (self.a.cyl_r * d[2] / n if n > 1e-6 else 0.0))

    def out_of_hand(self):
        return float(np.linalg.norm(self.sim.cup()[0] - self.sim.tcp())) > 0.12

    def lost(self):  # after the swing: fallen over (more than 45 deg from vertical) or out of the hand
        return self.lean() > 45.0 or self.out_of_hand()

    def robot_contacts(self):
        sim = self.sim
        cb, table = sim.m.body("cup").id, sim.m.geom("table").id
        return sum(1 for c in sim.d.contact[: sim.d.ncon]
                   if cb in (sim.m.geom_bodyid[c.geom1], sim.m.geom_bodyid[c.geom2]) and table not in (c.geom1, c.geom2))

    # ---- phases ------------------------------------------------------------
    def approach_and_close(self, grasp, pre):
        a, sim = self.a, self.sim
        self.width = OPEN
        self.goto(pre, vmax=0.25)
        self.goto(grasp, vmax=0.15)
        self.hold(0.3)
        # close at a constant speed until the fingers stop (contact), then record the contact width
        w = [OPEN]

        def closing():
            w[0] = max(CLOSED, w[0] - a.close_speed * CTRL)
            return w[0]

        self.width = closing
        still = 0
        for _ in range(int(3.0 / ARM_DT)):
            self.tick()
            still = still + 1 if abs(sim.d.qvel[sim.dof_lf]) < 5e-3 and w[0] < sim.d.qpos[sim.lf] - 0.001 else 0
            if still >= 3:
                break
        self.q_contact = float(sim.d.qpos[sim.lf])
        self.width = self.q_contact - a.delta_hold
        self.hold(0.3)
        print("closed: contact finger position %.4f m (pad gap %.1f mm), tcp-cup %s" % (
            self.q_contact, 1000 * 2 * (self.q_contact - 0.009 + 0.0), np.round(sim.cup()[0] - sim.tcp(), 3)))

    def lift_and_swing(self, grasp):
        a, sim = self.a, self.sim
        self.goto(grasp + np.array([0, 0, a.lift]), vmax=a.lift_speed)  # lift with the firm grip
        if self.metrics is not None:
            self.metrics.start()  # pickup done: the efficiency metrics count from here
        self.hold(a.settle_time)
        if a.tilt_after_lift:
            self.tilt_hand(a.tilt_after_lift, a.tilt_time)
        lift_tilt, lift_dist = self.tilt(), float(np.linalg.norm(sim.cup()[0] - sim.tcp()))
        print("lifted: tilt %.1f, cup z %.3f" % (self.tilt(), sim.cup()[0][2]))
        # ease the width off slowly until the cylinder starts to rotate, then regulate the width to keep a gentle rotation speed
        firm = self.q_contact - a.delta_hold
        delta = [a.delta_hold]
        self.width = lambda: self.q_contact - delta[0]
        prev, t_close, started = self.tilt(), None, False
        floor = [a.delta_min]
        x_start = self.carrot[0]
        self.info = {"lift_tilt": lift_tilt, "lift_dist": lift_dist, "delta_start": None, "close_t": None, "close_tilt": None, "close_rate": None, "lost": False}
        if a.fixed_delta > 0:  # fixed release width (the original design): no ease-off search, no speed regulation
            delta[0], started = a.fixed_delta, True
            floor[0] = delta[0]
            self.info["delta_start"] = 1000 * delta[0]
        for k in range(int(a.swing_timeout / CTRL)):
            if not started and x_start - self.carrot[0] < a.ease_retract:  # pull the hand back a little while the width eases off
                self.carrot = self.carrot + np.array([-a.ease_retract_speed * CTRL, 0.0, 0.0])
            self.tick_fast(k)
            th = self.tilt()
            om = (th - prev) / CTRL
            om_f = om if k == 0 else 0.9 * om_f + 0.1 * om
            prev = th
            if a.verbose and k % 10 == 0 and started:
                print("  swing t %.3f tilt %6.1f rate %7.1f delta %.2f mm finger q %.4f cup-tcp %.3f" % (k * CTRL, th, om_f, 1000 * delta[0], sim.d.qpos[sim.lf], float(np.linalg.norm(sim.cup()[0] - sim.tcp()))))
            if t_close is None:
                if not started and a.fixed_delta <= 0:
                    delta[0] = max(a.delta_min, delta[0] - a.ease_rate * CTRL)  # ease off
                    if om_f > a.start_rate:
                        started = True
                        floor[0] = max(a.delta_min, a.floor_frac * delta[0])  # never grip looser than where the rotation started
                        self.info["delta_start"] = 1000 * delta[0]
                        print("rotation started at t=%.2f s: width delta %.2f mm, tilt %.0f" % (k * CTRL, 1000 * delta[0], th))
                elif a.fixed_delta <= 0:  # P on the swing speed: squeeze when faster than the target, ease off when slower
                    if om_f < a.post_ease_below:  # still creeping: keep easing off slowly so that it does not stall
                        delta[0] = max(floor[0] * 0 + a.delta_min, delta[0] - a.post_ease_rate * CTRL)
                    delta[0] = float(np.clip(delta[0] + a.rate_gain * (om_f - a.rate_target) * CTRL, floor[0], max(a.delta_hold, a.reg_max)))
                if started and th + a.lead_time * om_f >= a.close_tilt and om_f > 5.0:
                    t_close = k * CTRL
                    self.info.update(close_t=t_close, close_tilt=th, close_rate=om_f)
                    print("close at t=%.2f s: tilt %.0f, rate %.0f deg/s, delta was %.2f mm" % (t_close, th, om_f, 1000 * delta[0]))
                    delta[0] = a.delta_close
            elif k * CTRL > t_close + 0.25 and abs(om_f) < 15.0:
                break
            if self.out_of_hand():
                print("object lost during the swing (tilt %.0f, delta %.2f mm)" % (th, 1000 * delta[0]))
                self.info["lost"] = True
                return False
        self.width = firm
        th = self.tilt()
        self.info.update(end_tilt=th, end_lean=self.lean())
        ok = th > a.min_hang_tilt and not self.lost()
        print("swing end: tilt %.0f, lean %.1f, cup z %.3f -> %s" % (th, self.lean(), self.sim.cup()[0][2], "inverted and held" if ok else "no inverted hang"))
        return ok

    def lift_and_roll(self, grasp):
        """Naive baseline: lift with the firm grip, roll the wrist joint by 180 deg (pure joint motion, min-jerk), keep the grip firm."""
        a, sim = self.a, self.sim
        self.goto(grasp + np.array([0, 0, a.lift]), vmax=a.lift_speed)
        if self.metrics is not None:
            self.metrics.start()
        self.hold(a.settle_time)
        angle = min(np.pi, sim.m.jnt_range[sim.m.joint("wrist_rotate").id, 1] - 1e-3)
        q0 = self.q.copy()
        n = max(1, int(a.roll_time / ARM_DT))
        for i in range(1, n + 1):
            u = i / n
            self.q = q0.copy()
            self.q[5] = q0[5] + angle * (10 * u**3 - 15 * u**4 + 6 * u**5)
            for _ in range(round(ARM_DT / CTRL)):
                w = self.width() if callable(self.width) else self.width
                sim.step(self.q, w, round(CTRL / DT))
            if self.out_of_hand():
                print("object lost during the roll")
                return False
        self.rot = self.rot @ R.from_euler("z", angle).as_matrix()  # the roll is about the approach axis (TCP z)
        self.carrot = sim.tcp()
        self.hold(a.settle_time)
        th = self.tilt()
        self.info = {"lift_tilt": None, "lift_dist": None, "delta_start": None, "close_t": None, "close_tilt": None, "close_rate": None, "lost": False,
                     "end_tilt": th, "end_lean": self.lean()}
        ok = abs(th) > a.min_hang_tilt and not self.lost()
        print("roll end: tilt %.0f, lean %.1f, cup z %.3f -> %s" % (th, self.lean(), sim.cup()[0][2], "inverted and held" if ok else "failed"))
        return ok

    def tick_fast(self, k):
        """Grip loop step: arm servo every ARM_DT, physics at CTRL."""
        sim = self.sim
        if k % round(ARM_DT / CTRL) == 0:
            self.q = sim.servo(self.q, self.carrot, self.rot)
        w = self.width() if callable(self.width) else self.width
        sim.step(self.q, w, round(CTRL / DT))

    def align_and_lower(self, vmax=0.06):
        """Raise 1.5 cm, move the lowest point of the cylinder over the target contact point, lower until it touches the table."""
        a = self.a
        target = np.array([self.start_xy[0], self.start_xy[1], TABLE_Z])
        d_xy = target[:2] - self.lowest_point()[:2]
        if a.verbose:
            print("  align: end-face centre %s -> %s; tcp %s cup-tcp %s width finger q %.4f (contact %.4f)" % (np.round(self.lowest_point(), 3), np.round(target, 3), np.round(self.sim.tcp(), 3), np.round(self.sim.cup()[0] - self.sim.tcp(), 3), self.sim.d.qpos[self.sim.lf], self.q_contact))
        if np.linalg.norm(d_xy) > 0.003:
            self.goto(self.carrot + np.array([0, 0, 0.015]), vmax=0.05)
            if self.lost():
                return self.abort("object lost while lifting off the table")
            self.goto(self.carrot + np.array([d_xy[0], d_xy[1], 0.0]), vmax=vmax)
            if a.verbose:
                print("  after carrying: end-face centre %s (error %.1f mm), tcp %s" % (np.round(self.lowest_point(), 3), 1000 * np.linalg.norm(self.lowest_point()[:2] - target[:2]), np.round(self.sim.tcp(), 3)))
            if self.lost():
                return self.abort("object lost while carrying")
        stop = lambda: self.lowest_z() <= TABLE_Z + a.touch_tol
        if not stop():
            self.goto(self.carrot + np.array([0, 0, -0.15]), vmax=0.04, amax=0.3, stop=lambda: stop() or self.lost())
            if self.lost():
                return self.abort("object lost while lowering")
        return True

    def place(self):
        a = self.a
        p_low = self.lowest_point()
        target = np.array([self.start_xy[0], self.start_xy[1], TABLE_Z])
        print("lower end-face centre %s -> target %s (move %.1f cm), lowest point %.1f mm above the table, lean %.1f" % (
            np.round(p_low, 3), np.round(target, 3), 100 * np.linalg.norm(target[:2] - p_low[:2]), 1000 * (self.lowest_z() - TABLE_Z), self.lean()))
        # the cylinder hangs free in the grip, tilted a few degrees from the vertical: pitch the (rigidly gripping) hand by the error first
        if self.lean() > a.pitch_fix_lean:
            self.pitch_fix()
        if not self.align_and_lower():
            return False
        if a.use_level and self.lean() > a.level_lean:
            self.level()
            if not self.align_and_lower(vmax=0.04):
                return False
        print("lowered: lowest point %.1f mm above the table, lean %.1f deg, end-face centre xy error %.1f mm" % (
            1000 * (self.lowest_z() - TABLE_Z), self.lean(), 1000 * np.linalg.norm(self.lowest_point()[:2] - target[:2])))
        return True

    def pitch_fix(self):
        """Fallback when the stand-up by translation does not remove the lean (the cylinder is rigid in the grip and just slides along the table):
        lift it slightly and pitch the hand about the finger axis by the remaining error so that the cylinder's axis becomes vertical."""
        th = self.tilt()
        if th < 90:  # not inverted: nothing sensible to do
            return
        delta = 180.0 - th
        print("pitch fix: hand pitched by %.1f deg (cylinder tilt %.1f -> 180)" % (delta, th))
        if self.lowest_z() < TABLE_Z + 0.02:
            self.goto(self.carrot + np.array([0, 0, 0.02]), vmax=0.04)
        cur = self.rot
        n = max(1, int(1.0 / ARM_DT))
        for i in range(1, n + 1):
            u = i / n
            u = 10 * u**3 - 15 * u**4 + 6 * u**5
            self.rot = R.from_euler("y", np.radians(delta * u)).as_matrix() @ cur
            self.tick()
        self.hold(0.2)

    def level(self):
        """Stand the cylinder up: the lowest point stays on the table, the hand moves until the axis is vertical (stops when upright,
        when the lean stops improving, or after a travel limit). Lighter grip: the cylinder rests on the table."""
        a, sim = self.a, self.sim
        if self.lean() <= a.level_lean:
            return
        self.width = self.q_contact - a.delta_level
        x0, best, k_best = self.carrot[0], self.lean(), 0
        for k in range(int(4.0 / ARM_DT)):
            ax = self.axis()[0]
            # horizontal motion only: pushing the hand down onto the resting cylinder made the pads slide up its side
            self.carrot = self.carrot + np.array([np.clip(0.8 * ax * 0.095, -0.03 * ARM_DT, 0.03 * ARM_DT), 0.0, 0.0])
            self.tick()
            if a.verbose and k % 5 == 0:
                print("  level t %.1f lean %5.1f axis_x %+.3f carrot x %.3f tcp x %.3f cup %s fingers L %.4f (contact %.4f)" % (k * ARM_DT, self.lean(), self.axis()[0], self.carrot[0], self.sim.tcp()[0], np.round(self.sim.cup()[0], 3), self.sim.d.qpos[self.sim.lf], self.q_contact))
            if self.lost():
                return
            ln = self.lean()
            if ln < best - 0.5:
                best, k_best = ln, k
            if ln < a.level_lean or (k - k_best) * ARM_DT > 0.6 or abs(self.carrot[0] - x0) > 0.06:
                break
        self.width = self.q_contact - a.delta_hold

    def release_and_retract(self):
        a, sim = self.a, self.sim
        q_open = [self.q_contact - a.delta_hold]
        t0 = sim.d.time

        def opening():
            u = min(1.0, (sim.d.time - t0) / a.open_time)
            return (self.q_contact - a.delta_hold) + (OPEN - (self.q_contact - a.delta_hold)) * u

        self.width = opening
        self.watch = lambda tag: print("  %s t %.2f lean %5.1f cup %s fingers L %.4f R %.4f tcp x %.3f contacts %d" % (tag, sim.d.time - t0, self.lean(), np.round(sim.cup()[0], 3), sim.d.qpos[sim.lf], sim.d.qpos[sim.lf + 1], sim.tcp()[0], self.robot_contacts())) if a.verbose else None
        cleared = lambda: sim.d.qpos[sim.lf] > self.q_contact + a.clear_margin
        # wait until the fingers have opened by the margin (no pressure on the object), then retract while they keep opening
        for i in range(int(2.0 / ARM_DT)):
            self.tick()
            if i % 4 == 0:
                self.watch("opening")
            if cleared():
                break
        if not cleared():
            print("fingers did not open")
        # the palm can rest against the back of the cylinder (it slides along the pads during the swing): go straight up first so that nothing is
        # pushed backwards, then retract
        self.goto(self.carrot + np.array([0.0, 0.0, a.release_lift]), vmax=0.05)
        self.hold(0.2)
        self.watch("before retract")
        self.goto(self.carrot + np.array([-a.retract_dist, 0, 0]), vmax=0.12, stop=lambda: (self.watch("retracting") if int(sim.d.time * 50) % 5 == 0 else None) or False)
        self.width = OPEN
        self.watch("after retract")
        self.hold(0.5)
        self.watch("settled")

    def abort(self, why):
        print("ABORT: %s -> open in place, straight up" % why)
        self.width = OPEN
        self.hold(0.5)
        self.goto(self.carrot + np.array([0, 0, 0.08]), vmax=0.10)
        self.hold(0.5)
        print("RESULT fail: %s" % why)
        return False

    def score(self):
        sim, a = self.sim, self.a
        dof = sim.m.joint("cup_joint").dofadr[0]
        rest_t, waited = 0.0, 0.0
        self.width = OPEN
        while waited < 6.0 and rest_t < 1.0:
            self.tick()
            waited += ARM_DT
            v = sim.d.qvel[dof : dof + 6]
            still = np.linalg.norm(v[:3]) < 0.01 and np.linalg.norm(v[3:]) < 0.1
            rest_t = rest_t + ARM_DT if (self.robot_contacts() == 0 and still) else 0.0
        p = sim.cup()[0]
        tilt_inv = float(np.degrees(np.arccos(np.clip(-self.axis()[2], -1, 1))))
        err = float(np.linalg.norm(p[:2] - self.start_xy))
        ok = rest_t >= 1.0 and tilt_inv < 15 and err < 0.03 and p[2] < TABLE_Z + a.cyl_h + 0.015
        print("settled: robot contacts %d, at rest for %.1f s" % (self.robot_contacts(), rest_t))
        print("RESULT %s: tilt from inverted %.1f deg, placement error %.3f m, cup z %.3f" % ("SUCCESS" if ok else "fail", tilt_inv, err, p[2]))
        return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mass", type=float, default=0.15)
    ap.add_argument("--randomize", action="store_true", help="mass 0.10-0.30, COM +-5 mm lateral / +-15 mm axial (per seed)")
    ap.add_argument("--com-only", action="store_true", help="with --randomize: keep --mass fixed, randomize only the COM")
    ap.add_argument("--com-axial", type=float, default=0.015)
    ap.add_argument("--com-lateral", type=float, default=0.005)
    ap.add_argument("--cyl-r", type=float, default=0.0225)
    ap.add_argument("--cyl-h", type=float, default=0.07)
    ap.add_argument("--dx", type=float, default=-0.008, help="TCP offset along the approach axis vs the cylinder axis [m]")
    ap.add_argument("--dz", type=float, default=-0.045, help="TCP height rel. to the cylinder centre [m]")
    ap.add_argument("--lift", type=float, default=0.16, help="lift height before the swing [m]; the inverted cylinder hangs ~11.5 cm below the pads")
    ap.add_argument("--lift-speed", type=float, default=0.10, help="top speed of the lift [m/s]")
    ap.add_argument("--settle-time", type=float, default=0.5, help="wait after the lift and after the tilt so that the arm and the cylinder come to rest [s]")
    ap.add_argument("--tilt-after-lift", type=float, default=6.1, help="pitch the hand (and the gripped cup) forward by this many degrees after the lift, before easing the width off")
    ap.add_argument("--tilt-time", type=float, default=0.6)
    ap.add_argument("--post-ease-rate", type=float, default=0.0, help="keep easing the width off at this speed [m/s] while the swing is slower than --post-ease-below")
    ap.add_argument("--post-ease-below", type=float, default=40.0, help="deg/s")
    ap.add_argument("--com-z", type=float, default=None, help="COM offset along the cylinder axis [m], on the axis (no lateral offset); positive = towards the end away from the pads")
    ap.add_argument("--ease-retract", type=float, default=0.04, help="pull the hand back this far while the width eases off [m]")
    ap.add_argument("--ease-retract-speed", type=float, default=0.01, help="speed of that pull-back [m/s]")
    ap.add_argument("--close-speed", type=float, default=0.03, help="finger speed when closing [m/s]")
    ap.add_argument("--delta-hold", type=float, default=0.012, help="width target below the contact width when gripping firmly [m] (force ~ 1000 N/m * delta)")
    ap.add_argument("--delta-loose", type=float, default=0.0015, help="same while the cylinder is allowed to rotate [m]")
    ap.add_argument("--ease-rate", type=float, default=0.006, help="how fast the width target is eased off before the rotation starts [m/s]")
    ap.add_argument("--delta-min", type=float, default=0.0003, help="loosest allowed width target (below the contact width) [m]")
    ap.add_argument("--start-rate", type=float, default=8.0, help="rotation counts as started above this rate [deg/s]")
    ap.add_argument("--floor-frac", type=float, default=1.0, help="loosest width during the swing as a fraction of the delta at which the rotation started")
    ap.add_argument("--rate-target", type=float, default=60.0, help="swing speed to regulate to [deg/s]")
    ap.add_argument("--reg-max", type=float, default=0.024, help="largest squeeze the swing-speed regulator may use [m below the contact width]")
    ap.add_argument("--rate-gain", type=float, default=1e-3, help="width change per (deg/s) of rate error, per second [m]")
    ap.add_argument("--level-lean", type=float, default=2.0, help="stand the cylinder up if it leans more than this [deg]")
    ap.add_argument("--pitch-fix-lean", type=float, default=1.5, help="if the hanging cylinder leans more than this, pitch the hand by the error before placing it [deg]")
    ap.add_argument("--use-level", action="store_true", help="also try the translation stand-up on the table after lowering (off by default)")
    ap.add_argument("--delta-level", type=float, default=0.006, help="width target below the contact width while standing it up [m]")
    ap.add_argument("--fixed-delta", type=float, default=0.0, help="fixed release width during the swing [m below the contact width]; 0 = ease-off search + speed regulation")
    ap.add_argument("--delta-close", type=float, default=0.024, help="width target (below the contact width) when closing to stop the rotation [m]")
    ap.add_argument("--swing-only", action="store_true", help="stop after the swing (for sweeps)")
    ap.add_argument("--close-tilt", type=float, default=64.0, help="close the grip when the (predicted) tilt reaches this [deg]")
    ap.add_argument("--lead-time", type=float, default=0.08, help="prediction horizon for the closing trigger [s]")
    ap.add_argument("--min-hang-tilt", type=float, default=125.0)
    ap.add_argument("--swing-timeout", type=float, default=3.0)
    ap.add_argument("--touch-tol", type=float, default=0.001, help="lowering stops when the lowest point is this close to the table [m]")
    ap.add_argument("--open-time", type=float, default=0.8)
    ap.add_argument("--clear-margin", type=float, default=0.004, help="start retracting once the fingers opened this much beyond the contact width [m]")
    ap.add_argument("--release-lift", type=float, default=0.06, help="straight-up lift of the open hand before retracting [m]")
    ap.add_argument("--retract-dist", type=float, default=0.10)
    ap.add_argument("--contact-solref", type=float, default=0.0, help="if > 0: solref timeconst [s] for table and cylinder (default model: 0.02)")
    ap.add_argument("--stiff-table", type=int, default=0, help="1: stiff table contacts (priority over the cylinder); 0: model defaults")
    ap.add_argument("--pad-mu", type=float, default=1.5, help="sliding friction of the finger pads")
    ap.add_argument("--cyl-mu", type=float, default=1.5, help="sliding friction of the cylinder (MuJoCo uses the larger of the two geoms)")
    ap.add_argument("--torsion", type=float, default=0.004)
    ap.add_argument("--noslip", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--controller", default="pivot", choices=["pivot", "roll"], help="pivot: width-controlled gravity pivot; roll: naive 180 deg wrist roll")
    ap.add_argument("--roll-time", type=float, default=2.0, help="duration of the wrist roll [s]")
    ap.add_argument("--video", default=None)
    a = ap.parse_args(argv)

    sim = Sim(a.seed, a.mass, a.cyl_mu, pad_mu=a.pad_mu, sphere_r=0.010, pad_x=0.09, cyl_r=a.cyl_r, cyl_h=a.cyl_h)
    sim.dof_lf = sim.m.joint("left_finger").dofadr[0]
    m = sim.m
    m.opt.noslip_iterations = a.noslip
    if a.stiff_table:  # the table's contact parameters win over the cylinder's (priority): the cylinder no longer sinks into the table when its COM is off-centre
        tb = m.geom("table").id
        m.geom_priority[tb] = 1
        m.geom_solref[tb] = [0.002, 1.0]
        m.geom_solimp[tb] = [0.99, 0.999, 0.0001, 0.5, 2.0]
    if a.contact_solref > 0:  # same softness for table and cylinder (pads keep their own mixed with the cylinder)
        for g in (m.geom("table").id, m.geom("cyl").id):
            m.geom_solref[g] = [a.contact_solref, 1.0]
            m.geom_solimp[g] = [0.95, 0.99, 0.001, 0.5, 2.0]
    for g in [m.geom("left_pad").id, m.geom("right_pad").id, m.geom("cyl").id]:
        m.geom_condim[g] = 4
        m.geom_friction[g, 1] = a.torsion
    body = m.body("cup").id
    if a.randomize:
        rng = np.random.default_rng(1000 + a.seed)
        mass = rng.uniform(0.10, 0.30) if not a.com_only else a.mass
        m.body_mass[body] = mass
        m.body_inertia[body] = mass * np.array([(3 * a.cyl_r**2 + (2 * a.cyl_h) ** 2) / 12] * 2 + [a.cyl_r**2 / 2])
        com = np.array([rng.uniform(-a.com_lateral, a.com_lateral), rng.uniform(-a.com_lateral, a.com_lateral), rng.uniform(-a.com_axial, a.com_axial)])
        m.body_ipos[body] = com
        print("randomized: mass %.3f kg, COM offset (x y axial) %s mm" % (mass, np.round(com * 1000, 1)))
    else:
        print("mass %.3f kg" % m.body_mass[body])

    if a.com_z is not None:
        m.body_ipos[body] = [0.0, 0.0, a.com_z]
    mujoco.mj_setConst(m, sim.d)  # mass / inertia / COM were edited at runtime: refresh the derived constants (they set the contact softness)
    from probe_widowx import CUP_XY  # noqa: E402
    cx, cy = CUP_XY  # nominal start position (no jitter): the arm pose is solved before the scene is reset, so nothing disturbs the cylinder
    grasp = np.array([cx + a.dx, cy, TABLE_Z + a.cyl_h + a.dz])
    pre = grasp - np.array([0.04, 0, 0])
    q_gr, e2 = sim.solve_ik(grasp, R_SIDE, [0, 0.7, 0.6, 0, -1.3, 0])
    q_pre, e1 = sim.solve_ik(pre, R_SIDE, q_gr)
    assert e1 < 1e-3 and e2 < 1e-3
    sim.reset(q_pre)
    start_xy = np.array(sim.cup()[0][:2])
    print("start: cup %s tilt %.1f" % (np.round(sim.cup()[0], 4), sim.tilt()))

    frames, last_t = [], [-1.0]
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.6, 150, -15

    def snap():
        if a.video and sim.d.time - last_t[0] >= 0.04 - 1e-9:
            last_t[0] = sim.d.time
            frames.append(sim.render(cam))

    metrics = EpisodeMetrics(sim, start_xy)

    def hook():
        snap()
        metrics.update()

    sim.hook = hook
    ctl = WidthPivot(sim, a, start_xy)
    ctl.metrics = metrics
    ok = False
    result = {"ok": False, "swing_ok": False, "mass": float(m.body_mass[body])}
    try:
        ctl.approach_and_close(grasp, pre)
        print("[sim time %.1f s] grasped" % sim.d.time)
        swing_ok = ctl.lift_and_swing(grasp) if a.controller == "pivot" else ctl.lift_and_roll(grasp)
        print("[sim time %.1f s] swing done" % sim.d.time)
        result.update(swing_ok=bool(swing_ok), **{k: v for k, v in ctl.info.items()})
        if swing_ok and not a.swing_only:
            if ctl.place():
                print("[sim time %.1f s] placed" % sim.d.time)
                ctl.release_and_retract()
                print("[sim time %.1f s] released and retracted" % sim.d.time)
                ok = ctl.score()
                print("[sim time %.1f s] scored" % sim.d.time)
        elif not swing_ok:
            print("RESULT fail: no inverted hang")
    finally:
        if a.video and frames:
            os.makedirs(os.path.dirname(a.video) or ".", exist_ok=True)
            imageio.mimsave(a.video, frames, fps=25)
            print("wrote", a.video, len(frames), "frames")
    result["ok"] = bool(ok)
    metrics.update(force=True)
    result["metrics"] = {k: (float(v) if isinstance(v, (float, np.floating)) else bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in metrics.summary().items()} if metrics.started else None
    import json
    print("RESULT_JSON " + json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in result.items()}))
    return result


if __name__ == "__main__":
    main()
