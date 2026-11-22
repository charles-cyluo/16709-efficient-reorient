# Pivot feasibility probe (stock Panda hand, then Robotiq 2F-85): partial / negative result

**Question.** Can the stock Panda hand pinch the cup near its bottom, lift, and let it swing about the
fingertips under gravity with no wrist rotation (the target behavior in AGENTS.md)?

**Setup.** Same scene as [rigid flip](2026-11-20-rigid-flip.md). Side grasp as in `RigidFlip`, but the pinch is
2.5 cm above the cup base (`grasp_dz=-0.020`; at 1.3 cm the palm flange catches the table's near edge). After the lift
the arm holds still and the gripper force is reduced to F (tendon force, ~F/2 per finger), optionally with a short
hand translation ("nudge") to start the swing. Throwaway probe scripts, not committed. Single seed (0) unless noted;
this is exploratory, not an evaluation.

**Findings.**
1. A cup pinched on its own axis is in unstable equilibrium: zero gravity torque, so it never starts to tip. It needs
   an off-axis pinch or a hand nudge.
2. With the pinch on the cup diameter (`dx=0`) the cup tips to ~30 deg and then **sticks**, for F from 8 down to 1.5 N.
   Cause (contact dump): at ~30 deg the cup's lower rim rests on the **palm** (hand-frame 3.8 cm behind the pinch
   point). Removing torsional friction (condim 3) and replacing the pad boxes with point-like spheres changed nothing.
   For a 6.2 cm cup the Panda fingers are too short: the base sweeps into the palm.
3. F below ~1.5 N: the cup slides down the pads (weight ~2 N vs friction limit) and drops; there is no swing window at
   this cup size.
4. Pinching off-axis (`dx=-0.02`, cup hanging beyond the fingertips) swings the cup a full 180 deg, but the pinch
   normals are tilted so the cup is **squeezed out forward** and falls ~10 cm; it happened to land inverted on the
   table. That is a throw, not a controlled slip.
5. A narrower cup (outer diameter ~5.6 cm) clears the palm and swings freely (90-100 deg in 0.1 s) but is thrown out of
   the grip and lands on its side; a 4.6 cm cup made the simulation blow up.

**Conclusion.** The pivot baseline is not yet working. The finger-depth concern is confirmed for this cup size with the
stock hand. Options for the team (gripper/cup design is a team decision, see `AGENTS.md`): (a) extend the Panda
fingertips with a longer collision pad (custom gripper), (b) Robotiq 2F-85 from Menagerie (longer fingers; adaptive), (c)
smaller/lighter cup plus damped (controlled) release of grip during the swing, (d) ignore cup-palm collision (not
recommended: physically wrong).

## Part 2: Robotiq 2F-85 (same day)
Setup: `PandaCupEnv(gripper="robotiq")`, pinch 3 cm below the cup centre (1.5 cm above its base), arm held still after
the lift, grip force reduced afterwards. Throwaway probe scripts again; single runs / <= 8 seeds, exploratory.

Findings:
1. **Finger depth is solved**: the Robotiq has 3.7 cm pads and its body clears the cup, so the cup can swing about the
   fingertips without touching the hand (seen swinging to ~156 deg and 113 deg in some runs). The palm collision
   of the stock hand is gone.
2. **The blocker is now friction control**, not geometry. The cup is in unstable equilibrium upright, but the pads' static
   friction holds it: measured breakaway torque about the pinch axis is 0.057 Nm (flat pads, F = 0.5), 0.027 Nm
   (6 mm sphere pads), vs a gravity torque of m g h sin(tilt) ~ 0.067 sin(tilt) Nm (h ~ 3 cm, m = 0.23 kg). So the cup
   tips until ~25 deg and then sticks, matching observations. Breakaway scales with grip force, but the force must still
   hold the cup's ~2.2 N weight, leaving a narrow window.
3. Commanded grip force is not the real normal force: with flat pads the cup is still held at zero commanded force
   (the closed linkage just stays in position).
4. Gentle or hard hand-translation nudges (up to 20 cm in 0.1 s) hardly help (cup tilt <= ~25 deg): the position-controlled
   arm cannot deliver the multi-g accelerations needed.
5. With point-like sphere pads, a slowly decreasing grip force makes the cup start rotating (to 28-78 deg in 3 of 5
   seeds), but my first slip-feedback controller re-tightened the grip as soon as it detected slip and locked the cup
   mid-swing, and one seed threw the cup. A controller for the pivot is plausible but not found yet.

Takeaway: with the Robotiq the pivot is a *grip-force/friction control* problem (a narrow hold-vs-slip window) that
depends on contact modelling choices (pad shape, friction). That is a reasonable thing for the RL policy to have to learn,
but a hand-scripted baseline needs a better controller or modelling choices the team should agree on.

## Part 3: thin solid cylinder (like the reference video), stock Panda hand
The reference video (frames inspected, not the full timeline) shows a thin solid cylinder, the robot and object on the
floor (no table), and the stock hand. The object is now selectable: `PandaCupEnv(cup_shape="cylinder")`, 4.5 x 14 cm
(assumed size). Probe: `scripts/probe_pivot.py` (see its docstring); torsional friction removed on pads and object.

Findings (seeds 0-5 for the main configuration, fewer elsewhere; exploratory):
1. With the cylinder the palm no longer blocks the swing *if it swings forward* (away from the palm). Which way it falls is
   decided by tiny initial biases (unstable equilibrium): left alone it tips backward, its long top section hits the palm,
   and it sticks at ~45 deg.
2. Pinching slightly behind the object's axis biases the swing forward. With `--dx -0.016 --force 2.5` (seed 0) the cylinder
   swings upright -> horizontal -> hanging inverted below the fingers (tilt about 160-200 deg for ~0.5 s) with no wrist
   motion, matching the reference behaviour qualitatively. Video: `outputs/videos/pivot_probe_cylinder_seed0.mp4`
   (git-ignored; regenerate with the command in the script docstring, `--reclamp-tilt 0`).
3. It is **not reliable**: of seeds 0-5 only seed 0 reached an inverted hang; others stuck at ~60 deg, tipped to horizontal
   and fell, or were thrown out. After the swing the loose grip cannot hold the cylinder's weight and it slips out
   (seed 0 falls ~0.5 s later). Pinch offsets of 2 cm or more squeeze the object out forward; 1.2 cm tips it backward.
4. Re-clamping when |tilt| exceeds 150 deg is too late (the object is already moving fast and the gripper needs time to
   close). Needed next: a controller that keeps a hold force sufficient for the weight throughout the swing (or time-varying
   force tied to the angle/angular velocity), then lower and place inverted.
5. Rigid flip on the cylinder (same policy): grasp+lift 30/30, flip 22/30 (73%); failures are simulation blow-ups in the
   lowering phase (thin object, small time step already 0.5 ms). Not investigated further.

Videos for review: `outputs/videos/pivot_probe_cylinder_seed0.mp4`, `outputs/videos/rigid_flip_cylinder_seed0.mp4`,
`outputs/videos/rigid_flip_cup_seed0.mp4`.

## Part 4: closing the grip after the flip (stock hand and Robotiq)

Goal: time a grip close right after the thin cylinder has swung inverted, then set it down.
`scripts/probe_pivot.py` now switches torsional friction (condim 3 to 4) together with the grip force at the re-clamp.

- **Stock hand, close in place (re-clamp at 100-163 deg, seed 0):** the cylinder always ends horizontal and out of the hand.
  Reason: after the swing it is not between the pads. Its centre is about 6 cm ahead of and 6 cm below the TCP, hooked over the
  fingertips, with its lower end on the table. The fingers close to ~0 width on nothing.
- **Stock hand, open and regrasp:** opening drops the hook; the approach knocks the cylinder off the table.
- **Robotiq 2F-85, seed 0:** the grip does not release at any commanded force (0.3-5 N) or pad friction (0.2-0.7). Its box pads
  give a line contact that resists the swing, and the linkage preload dominates. Replacing the pad boxes with spheres at runtime
  gave chaotic swings and solver blow-ups. Not pursued further.
- **Open:** a free hang (lift above the object length) plus a controlled forward move of the hand, or a set-down release as the
  object lands inverted. Team decision pending.

## Part 5: WidowX 250 S (Menagerie `trossen_wx250s`) on the table

Setup: `scripts/probe_widowx.py` (standalone sim, `build_widowx_model()`), arm base on the table at x = 0.2 m, thin cylinder
(4.5 x 14 cm, 0.15 kg) at x = 0.6 m, side pinch near the bottom, lift 10 cm, loosen the grip, arm still, optional re-clamp.
Example: `uv run python scripts/probe_widowx.py --dx -0.008 --pinch-force 3 --force 2.8 --sphere-r 0.010 --pad-x 0.09 --reclamp-tilt 20 --hold-force 7`.

Reachability: a horizontal hand at pinch height needs the base at least ~0.36 m from the object (wrist-angle limit).

**5a. Menagerie contacts (two tiny tip spheres per finger).** The finger mesh is ~5.5 cm long (not longer than the Panda's)
and the spheres sit at the very tip, 2.4 cm apart vertically; the palm mesh hull fills the gap between the fingers.
The pinch must be within ~5 mm of the axis; two spheres per finger resist tilting as strongly as sliding. Release forces
0.05-2.2 N: the cylinder slides straight down, or tips and is ejected at about 90 deg. Never swings past horizontal.

**5b. Own primitive pads (Menagerie finger/palm collision switched off).** Flat box pad on each finger's inner face plus a
palm box (front face at gripper x = 0.058 instead of the hull). Findings:
- The inner face is at finger-joint position - 9 mm (ray cast on the mesh). The palm hull front is only ~3 cm behind the
  fingertips, so a 4.5 cm cylinder sits at the very front of the original fingers.
- A flat pad touches the cylinder along the line through its axis, so there is no gravity torque: the grip holds upright
  (>= 1.5 N) or slips straight down (<= 0.9 N); no tipping. Pinching on the pad's front edge (dx -0.030) tips it, but the
  cylinder falls out within 0.2 s (contact lost at ~10 deg tilt).
- Round pads (sphere r 8-15 mm, centre 1-2 cm behind the tip): at pinch dx -0.008 and release 2.8 N the cylinder tilts
  continuously and slowly (2, 4, 6, 5, 5, 6, 8, 10, 13, 21, 40, 66, 91 deg over 1.3 s) while held; below ~2.4 N it drops
  out at ~10 deg. With a 7 N re-clamp at 20 deg it stays at ~32 deg for 0.8 s and then rotates on to ~90 deg and falls out.
  The release happens at about 60-90 deg for every force tried (also with the re-clamp), which fits the lower end of the
  cylinder swinging back into the palm as it tips forward (the same problem as the Panda).
- Pinching closer to the bottom end (dz -0.055, -0.062) to clear the palm was blocked here because the palm box then hits
  the table; not resolved.

**Conclusion so far:** with the WidowX (own box or sphere pads) the cylinder can be held, lifted and made to tilt slowly, but
not carried past ~90 deg, so there is no inverted hang to close the grip on. Not better than the stock hand.

**5c. Palm clearance and smaller cylinder (WidowX, own pads).** What the cylinder touches was logged (`--contacts`):
- The release at 60-90 deg in 5b is not the palm: the cylinder has no contact at all (free fall) from ~30 deg on; the pads let go
  at ~10-20 deg. Higher friction (1.5-2.5) keeps it held while tilting slowly (9 -> 17 deg over 0.5 s at 3.6 N) but it still
  escapes at 50-110 deg for every constant force, and also with a tilt-rate feedback on the grip force (`--rate-target`).
  The lowest usable pinch height is limited by the wrist hull touching the table (about 2.5 cm above the table).
- Smaller cylinder (3 x 10 cm, 0.05 kg; `--cyl-r 0.015 --cyl-h 0.05 --mass 0.05 --dz -0.025`): needs a 0.25 ms step
  (`PROBE_DT=0.00025`), else the solver blows up. It is held rigidly at dx -0.006 (force irrelevant) and slips out at dx -0.010;
  in between (dx -0.007 to -0.009, 0.4-1.0 N) the outcomes are chaotic (tilt traces jumping across +-180 deg, several solver
  blow-ups, one run sitting at 120-130 deg for ~1 s on the hand/table). Not a controlled pivot.
- Decision (team, this session): fall back to a set-down baseline on the stock Panda hand.

## Part 6: pivot + set-down on the stock Panda hand (`scripts/run_pivot.py`, `src/reorient/policies/pivot.py`)

Policy: grasp/approach/close/lift as in the rigid flip (thin cylinder, TCP 1.6 cm behind the axis, 4.5 cm below the centre,
lift 10 cm), then loosen the grip to 2.5 N with the arm still (swing), detect the inverted hang (|tilt| > 145 deg, nearly still),
then pull the hand back so the object's axis becomes vertical (straighten), open, retract straight back.
Commands: `uv run python scripts/run_pivot.py --trials 6 -v`; the sweep below used `--straighten-tol 2` (skip straighten).

- Seed 0: swing reaches the inverted hang at t = 2.64 s (tilt -154 deg), but the gripper width is already ~0 there: the object
  is not between the fingers. It stands on the table leaning ~25 deg with its upper end forward, nothing in contact, and topples
  forward within ~0.5 s, so there is nothing for the hand to straighten (straighten phase fails, the object ends on the floor).
- Sweep (6 seeds each, no straighten; lift 0.06 / 0.10 / 0.14 m x dx -0.012 / -0.016 / -0.020 m, 2.5 N): inverted hang reached in
  0-2 of 6 runs, success 0/6 in 8 of 9 settings and 1/6 in one (a lucky landing); most runs end with the object lying on the
  table (z 0.742) or knocked off; a few solver blow-ups at lift 0.14.
- Conclusion: the set-down baseline as scripted does not work on the stock hand either: the object is not in the grip after
  the swing, and it lands leaning ~25 deg and falls. Open options: a catch with the hand from the lean side, a lighter / softer
  object, or active grip modulation during the swing (needs a model of the contact, not just a force schedule).

## Part 7: contact-model changes; a working pivot on the WidowX (this is the first one)

Panda, 6 seeds each (`probe_pivot.py --summary`): noslip iterations 5, soft contacts (solref 0.05-0.1), stiff contacts (solref
0.004), pyramidal cone and impratio 100 did not keep the object in the fingers after the swing (0/6 in every case; some grasp
failures with the soft/noslip settings).

WidowX (own sphere pads, see Part 5): `noslip_iterations` and a high `impratio` only make the grip stick rigidly (no creep) until
a friction cliff; a faster grip loop (2 ms) stretched the controlled swing (to ~116 deg) but it was still flung out at ~60-90 deg
because point contacts without torsional friction cannot brake the rotation. **Giving the pads and the cylinder a small torsional
friction (condim 4, coefficient 0.004 m) fixed it:** during the 10 cm lift the cylinder tips over the fingertips (arm moves only
up, no wrist motion), swings to an inverted hang of 136-168 deg and stays held in the pads (6/6 seeds held for 3 s), with its
lower end resting on the table. With 0.008 m and above it stays upright (no swing); with 0 it is flung out.

Place sequence (`--place`): wait for the hang, move the hand forward until the axis is vertical (lower end on the table), raise
2 cm / carry to the start xy only if more than 3 cm off, lower, release, retract 10 cm straight back.

Command (seeds 0-19, 20 trials, success = tilt from inverted < 15 deg, placement error < 3 cm, standing on the table):

    PROBE_DT=0.0005 uv run python scripts/probe_widowx.py --seed N --sphere-r 0.010 --pad-x 0.09 --dx -0.008 --mu 1.5 --pad-mu 1.5 \
      --pinch-force 3 --reclamp-tilt 0 --noslip 10 --hold-force 3.0 --torsion 0.004 --place

Result: **14/20 success (70%)**; failures: 4 trials never reach the inverted hang (cylinder drops or stays tipped over), 2 topple
during the straighten step. Successful trials end with tilt 0.0 deg from inverted and placement error 2.0-2.7 cm (so the 3 cm
limit is only just met; the hang lands ~5 cm in front of the start). Video (git-ignored): `outputs/videos/pivot_widowx_seed0.mp4`.

Caveats: parameters were tuned on seeds 0-5 and the cylinder mass (0.15 kg), friction (1.5) and size (4.5 x 14 cm) are fixed here,
not randomized; the swing happens while the object's lower end touches the table at the end (inverted, leaning ~30 deg, held by
the pads); this is the WidowX 250 S with custom primitive pads (not the stock Panda hand or Menagerie contacts), which is a team
decision (arm and gripper choice); the stock Panda hand and the Robotiq are not tested with torsional friction on at 0.004 yet.

## Part 8: the Part 7 recipe on the stock Panda hand

`probe_pivot.py` gained `--torsion-coef`, `--pinch-force` (grip force already during the lift), `--sphere-pad R` (one sphere per
finger instead of the pad boxes, bounding data updated), `--noslip`, `--pad-mu`. Six seeds per setting (`--summary`):

- Torsion 0.002 / 0.004 / 0.008 m, pinch + swing force 3 N, dx -0.016: 0/6 reach and hold an inverted hang (flung out at 90-170
  deg, or held below 63 deg for seed 1).
- Torsion 0.004, dx -0.004 and -0.010, 2 and 4 N: object held but only tilted 12-46 deg (a few seeds ~15), 3 seeds failed in the
  approach (a pad touches the object); one seed held at 157 deg (dx -0.004, 2 N) but not reproducible across seeds.
- Torsion 0.002-0.003, dx -0.012, 1.5 / 2.5 N: seeds 0 and 1 held at about -50 deg, seed 2 held at 139-147 deg once (1.5 N),
  others held low or flung.
- Torsion 0.004 + noslip 10, dx -0.008 / -0.012 / -0.016, 3 N: held at < 55 deg or flung; no hang.
- Round pads (sphere r 10 mm) with torsion 0.004 + noslip 10, dx -0.008 / -0.014: held at ~46 deg (2 seeds) or flung.

Conclusion: the stock Panda hand (box pads, or a single round pad per finger) gives a hold or a throw, not a held inverted hang,
for the settings tried; at most 1 seed in 6 reaches a held hang, never repeatably. The WidowX recipe from Part 7 is therefore
specific to that hand (small round pads at the fingertips, lower pinch). Not tried: re-tuning the Panda pad position / pinch
height, other torsion coefficients per finger, or a different swing release.

## Part 9: randomizing mass and centre of mass on the WidowX pivot (Part 7 recipe)

`probe_widowx.py` gained `--randomize` (mass 0.10-0.30 kg, cylinder friction 0.6-1.2, COM offset +-5 mm across the axis and +-15 mm
along it, drawn per seed), `--rand-only mass,mu,com`, `--com-axial`, `--com-lateral`, `--mass`. Same command as Part 7 plus these.
Note: with pad friction 1.5 the cylinder friction (max rule) has no effect, so friction randomization is not really tested.

- Everything randomized, seeds 0-5: 0/6 (5 never reach the inverted hang, 1 topples).
- One factor at a time (8 seeds): mass only 0/8; COM only 0/8; friction only 6/8 (the pads dominate friction).
- Mass sweep with the nominal forces (10 seeds each): 0.12 kg 0/10, 0.14 0/10, 0.15 6/10, 0.16 6/10, 0.18 0/10.
- COM sweep (mass 0.15, 10 seeds): axial +-3 mm 2/10, axial +-8 mm 0/10, lateral +-3 mm 1/10.
- Forces scaled with the mass (3 N x m / 0.15 for pinch and hold): 0.10 kg 0/10, 0.12 0/10, 0.14 0/10, 0.18 7/10, 0.22 0/10, 0.30 3/10.
- Torsion 0.002-0.006 at scaled forces: 0.12 kg 0/8 everywhere; 0.22 kg: 5/8 (0.002), 0/8 (0.003), 3/8 (0.005), 0/8 (0.006).

Conclusion: the 70% of Part 7 is a narrow, hand-tuned point (about +-5% in mass, a few mm in the COM). The outcome is a
sensitive, non-monotonic function of the parameters; scaling the grip force with mass does not widen it reliably. As an open-loop
scripted baseline it is not robust to the randomization AGENTS.md calls for (mass, friction, start orientation); a grip controller
with feedback on the swing (or the RL policy) would have to provide the robustness. Report the Part 7 number only together with
this sensitivity.

**Update to Part 7 (scoring and video):** success is now scored only after the retract, once there is no contact between the robot and the
cylinder (the table does not count) and the cylinder has been at rest on its own for 1 s (|v| < 1 cm/s, |omega| < 0.1 rad/s; `settled:`
line in the output). Re-run on seeds 0-19: still 14/20. The videos used to stop after the lift (102 frames); `probe_widowx.py` now records
one frame per 40 ms of simulated time in every phase, up to the end of the settle wait (e.g. 234 frames for seed 0).

## Part 10: where the 14/20 of Part 7 comes from

At the nominal setting the only per-seed randomness is the cylinder's start position (uniform +-10 mm in x and y, `Sim.reset`).
Mass (0.15 kg), friction, COM, gripper/torque schedule and lift are identical across trials; the TCP waypoints are recomputed
from the actual start position (so the joint trajectory shifts slightly). The simulator is deterministic (seed 0 run twice: same hang
tilt 167 deg and the same result). `--start-offset X Y` [mm] fixes the offset:

- y offset (x = 0): -10 mm fail (no inverted hang), -6, -2, +2, +6 mm success, +8 and +10 mm fail (no inverted hang).
- x offset (y = 0): -10 mm fail (topples while straightening), -5, +5, +10 mm success.
- In the 20 random seeds the failures (1, 2, 7, 8, 12, 19) have a start y of +9.0, -4.0, +7.9, +9.7, +8.9, +8.5 mm (seed 2 also has
  x = -4.8 mm); the successes mostly lie within y = -6..+6 mm.

So the 70% is the share of the +-10 mm start-position box that falls inside a small success window (|y| <~ 7 mm, x >~ -7 mm),
not simulator noise and not a different gripper schedule. The mechanism (arm configuration / waist angle, finger asymmetry from the
single actuated finger plus the equality coupling, or pad alignment) is not identified. Parameter randomization (mass, COM) is a
separate effect: see Part 9.

**Decision (user, this session): do not randomize the cylinder's start position.** `probe_widowx.py` now starts the cylinder at a fixed
position (`--start-jitter MM` restores the random offset, `--start-offset X Y` sets a fixed one). With the fixed start the Part 7 recipe
is deterministic and succeeds on every seed (nothing varies), so success rates from here on must come from physics randomization
(mass, friction, COM; `--randomize`), where Part 9 gives 0-20%. The 14/20 of Part 7 is superseded by that.

## Part 11: feedback grip controller on the WidowX pivot (`--feedback`)

Idea: the pads' torsional friction brakes the swing (torque cap ~ grip force), so the force is raised when the cylinder is ahead of a
reference tilt trajectory and lowered when it lags: `F = f0 + kp (tilt - tilt_ref) + kd (rate - rate_ref)`, clipped to [fmin, fmax];
`tilt_ref` is a min-jerk ramp from 0 to `--fb-end` over `--fb-time` (1 s, the lift). It runs from the start of the lift; the last force is
used as the hold force for the straighten step. The grip loop runs at 2 ms (`PROBE_CTRL=0.002`), the arm servo stays at 20 ms.
Start position fixed (see decision above); mass-only sweeps use 1 seed (the simulator is deterministic).

Command (best config found):
`PROBE_DT=0.0005 PROBE_CTRL=0.002 uv run python scripts/probe_widowx.py --seed N --mass M --sphere-r 0.010 --pad-x 0.09 --dx -0.008 --mu 1.5 --pad-mu 1.5 --pinch-force 3 --reclamp-tilt 0 --noslip 10 --torsion 0.004 --place --feedback --fb-fmin 1.5 --fb-kp 0.08 --fb-kd 0.015 --fb-fmax 10 --fb-end 150`

- Mass sweep 0.10-0.30 kg (step 0.02), success per mass: open loop only 0.15-0.16 (Part 9). Feedback at 50 Hz, kp 0.03, kd 0.004: 0.16-0.20.
  Feedback at 500 Hz, kp 0.08, kd 0.015, final tilt 160 deg: 0.14-0.20 and 0.28-0.30 (7/11), failing at <= 0.12 and 0.22-0.26 kg.
  The same with final tilt 150 deg (the command above, 7/11): successes at 0.14, 0.16, 0.18, 0.20, 0.24, 0.28, 0.30 kg; failures at
  0.10, 0.12, 0.22, 0.26 kg. (An earlier version of this note listed the 160 deg set for the 150 deg command.)
- Tuning (11 masses each): kp 0.15/kd 0.03 6/11; reference time 1.5 s 5/11; f0 4 N 4/11; f0 2.2 N 6/11; end 150 deg 7/11; kp 0.2/kd 0.05 4/11.
  Feed-forward force proportional to the true mass (f0 = 14, 20, 26 x m): 4/11, 6/11, 4/11 - no gain.
- Traces: light cylinders lag the reference, the force drops to the floor, the swing then runs away (400 deg/s) and the object leaves
  the pads; heavy cylinders swing too fast and leave the pads even at 4-8 N.
- Physics randomization (10 seeds, fixed start; open loop in brackets): everything randomized 2/10 (0/10), COM only 2/10 (0/8),
  mass only 4/10 (0/8).

Conclusion: feedback widens the working range and lifts the randomized success rate from 0% to 20% (mass: 40%), but it is far from
robust, the success set over mass is not contiguous, and neither more gain nor a mass feed-forward fixes it. The failure is the object
leaving the pads during the fast part of the swing (60-110 deg), which a force controller at this level of sensing does not prevent.

**Revision of the place sequence (user feedback, same session).** Two flaws in the earlier place step: (1) in runs where the object was
already inverted and well held (e.g. 0.24 kg, lean 9 deg) the hand kept pushing forward for the whole 4 s straighten timeout (12 cm,
dragging the object along the table) and then carried it back 12 cm to the start; (2) in failed runs the hand kept moving (and pushing
the fallen object with the lower part of the gripper) instead of stopping. Changes in `place_and_score`: straighten only if the lean
is above `--straighten-lean` (3 deg), stop when upright / stalled for 0.6 s / 6 cm travelled; carry and lower only if needed; if the
object is more than 45 deg from vertical or farther than 12 cm from the TCP at any step, open and go straight up (no horizontal motion);
after opening, the hand moves only once no robot geometry touches the object (else the trial fails with the hand left in place);
then lift straight up (`--release-lift`, 6 cm) before retracting. 
Numbers with this sequence (same feedback config): mass sweep 0.10-0.30 kg: 5/11 (0.14, 0.18, 0.20, 0.24, 0.28 succeed; the earlier 7/11
included 0.16 and 0.30 kg: now 0.16 ends 3.3 cm from the start, and 0.30 leans back onto the palm on release and stays wedged between the
palm box and the right pad, so the hand is left in place and the trial fails); physics randomization over 10 seeds: all 3/10, COM only
2/10, mass only 3/10.

**Second revision of the place sequence (user feedback on the videos: arm vibrating at the end, object stuck to the gripper).**
- Vibration: the retract ended with an instant stop and rang for ~1.5 s (arm joint speed 0.16 rad/s, 2-3 mm of tcp wobble). All hand motions
  after the swing now use an acceleration-limited trapezoid profile (`goto`, 0.4 m/s^2) and the arm is servoed for 0.5 s after the retract;
  joint speed at rest is 0.001 rad/s.
- Object carried away by the gripper (0.30 kg): on opening, the object leaned back onto the palm box and got wedged against the palm and
  the right pad (which opens further than the left one). Changes: grip force during straightening is lowered to 1.6 N (`--straighten-force`;
  the object rests on the table, and the torsional friction of the heavier grip had stalled the straighten step at a 3.5 deg lean); the
  release is gentle: grip force eased to 0.25 N over 0.4 s so the object stands by itself, then the fingers open linearly over 0.8 s; the hand
  moves only when nothing touches the object; a light touch (< 0.5 N, the palm sits just behind the object) is cleared by backing off 1.5 cm
  first; then straight up 6 cm, then back 10 cm.
- Numbers (same feedback config, fixed start): mass sweep 0.10-0.30 kg 7/11 (0.14, 0.16, 0.18, 0.20, 0.24, 0.28, 0.30 succeed; 0.10, 0.12,
  0.22, 0.26 fail); physics randomization, 10 seeds: all 2/10, COM only 3/10, mass only 4/10.
- Brake-only variant of the controller (never loosen below f0; `--fb-brake-only`): 3-4/11 for f0 1.5-3 N and several gains - worse, not used.
- Still failing: the light cylinders (<= 0.12 kg) are flung out of the pads during the fast part of the swing (the object flies off); this
  is the same swing instability as before, not a place-sequence issue.

## Part 12: width-controlled pivot (user's controller design), `scripts/run_width_pivot.py`

Design (user): IK for the arm + finger *width* position control only. (1) Move to the bottom part of the cylinder and close; the contact width is
measured (finger position when the fingers stop). (2) Lift with a firm grip (width target `delta_hold` below the contact width; the finger
actuator is a position servo, force ~ 1000 N/m x delta). (3) Ease the width off to reduce the friction so that the cylinder rotates. (4) When the
tilt is close to 180 deg (predicted with the measured rate), close the width again to stop the rotation. (5) From the pose of the static cylinder find
its lowest point, move the hand so that this point is above the contact point of the final pose (start xy), lower until it touches the table.
(6) Open the width while retracting straight back, then wait until the cylinder stands alone with no robot contact and score it.

Implementation details that were needed:
- Ease-off: a fixed "loosen a bit" has a friction cliff (1.8 mm: slips out, 2.2 mm: never rotates, at 0.15 kg), so the width is eased off at 3 mm/s from
  `delta_hold` (5 mm) until the rotation starts (rate > 8 deg/s); then `delta` is regulated by P on the swing speed (target 120 deg/s, gain 1e-4 m per
  deg/s per s), never looser than the value at which the rotation started (without this floor the controller loosened further while the swing was
  still slow and the light cylinders slid out through the pads along their axis).
- The "lowest point" for alignment is the centre of the lower end face (it coincides with the contact point of the upright final pose; aligning the true
  lowest rim point of a tilted cylinder with the target left a 2.25 cm placement error); the table-contact test uses the true lowest rim point.
- The cylinder ends the swing leaning 5-33 deg (held in the pads, lower end on the table). A small "stand it up" step (hand moves until the axis is
  vertical, lowest point kept on the table, lighter grip; travel limit 6 cm, stall stop) is run after touching down, then the lowest point is aligned again.
- Release: open over 0.8 s, retract once the fingers have cleared the cylinder by 4 mm per side; abort rules as in Part 11 (object lost, fallen: open in place and go straight up).
- Contact model as in Part 7 (round pads, torsion 0.004, noslip 10). No mass information is used anywhere.

Command: `PROBE_DT=0.0005 uv run python scripts/run_width_pivot.py --seed 0 --mass 0.2 --delta-hold 0.005 --rate-gain 1e-4 --rate-target 120 --video outputs/videos/width_pivot_mass200.mp4`

Results (fixed start, one seed per mass because the simulator is deterministic): mass sweep 0.10, 0.12, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30 kg:
success 0 0 0 1 1 1 1 0 1 0 0 1 = 6/12 (final placement error 1-9 mm, tilt from inverted 0 deg). Failures: 0.10-0.14 kg the cylinder leaves the pads or
stops at 90-126 deg during the swing; 0.22 kg lost at 90 deg; 0.26 kg held at 142 deg (lean 38 deg) but lost while carrying; 0.28 kg lies on the table.
COM randomization at 0.18 kg (10 seeds): axial +-15 mm and lateral +-5 mm 2/10; +-8 / +-3 mm 5/10; axial +-4 mm 6/10; lateral +-4 mm 8/10.
Mass and COM randomized together (10 seeds): 0/10.

## Part 13: parallel sweeps for the width pivot (`scripts/sweep_width_pivot.py`), free swing (lift 16 cm)

Each setting is one deterministic simulation; a worker pool (30 processes, 32 cores) runs 800-2300 settings in 20-65 s. Physics is CPU only: nothing here uses
a GPU (MuJoCo's own engine is CPU; the GPU variants MJX / MuJoCo Warp are not installed in this environment; video rendering uses Mesa software EGL because NVIDIA EGL
hangs on this machine even when pinned to the first device with `MUJOCO_EGL_DEVICE_ID=0`; `nvidia-smi` cannot query the second GPU). Results: `outputs/sweeps/*.json` (git-ignored).

Findings (WidowX, round pads, torsion 0.004, noslip 10; widths are the finger-position target below the measured contact width, force ~ 1 N per mm):
1. **The 10 cm lift of the earlier runs let the lower end bump the table, and the bump stopped the rotation** (swing end at 154 deg, cup z 0.792 = resting on the table).
   With a 16 cm lift the swing is free: it reaches 500-800 deg/s before a closing at 5 mm squeeze, which cannot stop it (tilt goes past 200 deg, cylinder lost).
2. **The lift with "full contact" is only clean with a hold squeeze >= 8 mm (12 mm is safe for 0.10-0.30 kg, lift speed 0.05-0.10 m/s).** With 4-6 mm the heavier cylinders
   (>= 0.20 kg) rotate or drop during the lift (e.g. 0.24 kg flips to 179 deg while lifting, which looked like a swing success for every release width). All earlier mass sweeps
   used 3-5 mm and are contaminated by this; default hold is now 12 mm.
3. **Release width that gives a good swing (fixed width, hold 12 mm, 2244 settings; good = held, 160-200 deg, lean < 20 deg):** 0.26-0.30 kg: wide windows (release 3.5-6.5 mm, any closing
   squeeze 10-18 mm and trigger 60-120 deg: 25/50/60 good settings of 204 each); 0.24 kg: only 5.0 mm; 0.20 kg: 2.0-4.0 mm with a 10 mm closing squeeze; 0.16 kg: one setting (3.0 mm);
   0.18 and 0.22 kg: none; 0.10-0.14 kg: none (the cylinder either does not rotate or leaves the pads).
4. **The width at which the rotation starts (eased off at 3 mm/s from 12 mm) grows with mass:** 0.14 kg 3.0 mm, 0.16 3.5, 0.18 3.2, 0.20 3.7, 0.22 4.9, 0.24 4.6, 0.26 5.1, 0.28 5.7, 0.30 6.1 mm (about 19 mm per kg).
5. **Full pipeline with "ease off until it starts, then hold that width, close at the trigger tilt"** (12 closing settings per mass): 0.26 kg 9/12, 0.28 kg 6/12, 0.30 kg 8/12, 0.24 kg 2/12, <= 0.22 kg 0.
   The lighter ones stall at 2-14 deg after the start (holding the starting width only creeps) or are lost (0.22 kg). Speed regulation on top (27 combinations of floor 0.4-1.0, target 60-300 deg/s,
   gain 3e-5-3e-4, closing 14 mm at 100 deg) was no better: 0-3 of 11 masses, only the heavy ones.
6. **Pulling the hand back 0 / 2 / 4 / 6 cm while the width eases off (as suggested) made no systematic difference:** 9 / 8 / 9 / 7 successes out of 36 (12 masses x 3 closing triggers).

Open: no release-width rule yet for masses <= 0.22 kg (both ends of the friction window are narrow or empty there); a scripted open-loop controller that handles the whole 0.10-0.30 kg range is not in sight with this
pad / friction model.

## Part 14: final width-pivot controller on the WidowX: mass 0.15-0.25 kg, COM anywhere on the axis within 1/4 of the cylinder length

Target (user): mass range 0.1 kg (0.15-0.25 kg), COM on the vertical axis of the cylinder with a height range of 1/4 of the whole cylinder (+-1.75 cm for the
14 cm cylinder), fixed start position; "a controller that almost succeeds in all these scenes". Success = tilt from inverted < 15 deg, end-face centre within 3 cm of
the start xy, standing on the table at rest for 1 s with no robot contact (all runs below use the same criterion as before).

### Setup bug found on the way (affects Parts 9-13 and the mass sweeps of Parts 7-8)
The scripts changed the cylinder's mass, inertia and COM at runtime without calling `mujoco.mj_setConst`, so MuJoCo's derived constants (inverse weights that set the
contact softness) kept the values of the original cup. The resting height of the cylinder then depended on its COM (e.g. 1.6 cm sunk into the table at +1.75 cm COM;
the cylinder was also knocked tilted by the placeholder arm pose of the first scene reset) and the contact behaviour was inconsistent across masses. Fixed in
`scripts/run_width_pivot.py` (`mj_setConst` after the edits; one scene reset with the arm already in its pre-grasp pose). **Mass / COM numbers of Parts 7-13 are therefore not reliable
(the sweeps with a fixed mass were affected less).** `PandaCupEnv.reset` randomizes mass and inertia the same way; adding `mj_setConst` there changes the Robotiq rigid-flip result enough to fail
`test_rigid_flip_inverts_cup_with_robotiq` (so it was left unchanged): open item, the Panda baselines (98 % / 100 % in the rigid-flip notes) were measured with the stale constants.

### Controller (`scripts/run_width_pivot.py`, defaults = the values below)
IK for the arm + finger *width* position control only (no mass information used):
1. approach, close at 3 cm/s until the fingers stop, record the contact width;
2. lift 16 cm with a firm grip (12 mm below the contact width, about 12 N); lift 16 cm lets the cylinder swing free (at 10 cm the lower end bumped the table, which stopped the rotation);
3. **tilt the hand (and the gripped cylinder) forward by 6.1 deg about the finger axis** (your idea; with 0 deg the rotation never starts within 3 s for most scenes, with 8 deg it starts reliably);
4. ease the width off at 6 mm/s while pulling the hand back 4 cm at 1 cm/s (your idea) until the rotation starts (> 8 deg/s); then keep that width;
5. when the predicted tilt (rate x 0.08 s) reaches 64 deg close hard (24 mm below the contact width) to stop the rotation;
6. from the pose of the static cylinder: pitch the hand by the remaining lean (rigid grip: the cylinder's axis becomes vertical), move the centre of its lower end face over the start position, lower in one descent until its lowest
   point touches the table; then open the width over 0.8 s, go straight up 6 cm, retract 10 cm, wait until the cylinder stands at rest with no robot contact, score.
All hand motions use an acceleration-limited trapezoid profile. The parameters were found by random search (`scripts/search_width_pivot.py`, 200 sets x 25 scenes on a 5 x 5 grid of mass and COM
height; the search space: hand tilt 4-16 deg, pull-back 0-6 cm and speed, ease-off rate, post-start easing, closing squeeze 10-24 mm, closing trigger 60-130 deg, lead time 0-0.08 s).

### Results (simulator deterministic; every scene is one run; fixed start position)
- Held-out random scenes (mass uniform in 0.15-0.25 kg, COM height uniform in +-1.75 cm, never seen during the search): 300/300, 499/500, 500/500 = **1299/1300 (99.9 %)**. The only failure: 0.2498 kg with the COM at +1.74 cm
  (the corner: heaviest and top-heavy; the cylinder is lost while it is lifted off the table).
- 5 x 5 grid used for the search: 24/25 (same corner). The video of that corner is `outputs/videos/width_pivot_heavy_com_high.mp4`.
- Other candidates of the search on 300 scenes: 291-297/300, so the plateau is broad (many parameter sets reach 23-24 of 25 on the grid); a fast ease-off (6 mm/s) was in all the best ones.
- Time per trial: about 14-16 s of simulated time (the earlier version took 95 s: moves ran to a 12 s cap because of a wrong end condition).

### Other fixes made while building it
- The "stand it up by moving the hand over the table" step was dropped (the cylinder is rigid in the grip and just slides); the hand pitch fix replaces it (`--use-level` keeps the old step).
- A sweep that runs the simulations in a `multiprocessing.Pool` blocked forever when one run crashed; `search_width_pivot.py` now runs every scene as its own subprocess with a 120 s limit
  (`sweep_width_pivot.py` still uses the pool; it also ran the older, contaminated settings).
- Commands: `PROBE_DT=0.0005 uv run python scripts/run_width_pivot.py --seed 0 --mass=0.2 --com-z=0.0 --video outputs/videos/width_pivot_nominal.mp4`;
  `PROBE_DT=0.0005 uv run python scripts/search_width_pivot.py --eval '{...}' --random-scenes 500 --scene-seed 31 --tag val31`.
Caveats: WidowX with custom primitive pads (the hand of the Panda is not used), torsional friction 0.004 and noslip 10 on the pad / cylinder contacts, fixed start position, no friction randomization (pad friction 1.5 dominates the
max rule), cylinder 4.5 x 14 cm; the parameter set was tuned on the grid and checked on 1300 other scenes of the same ranges, not on other ranges.
