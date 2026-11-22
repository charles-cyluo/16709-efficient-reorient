# Env and scripted baselines (milestone 1)

## Scene (`assets/scene.xml`, `assets/cup.xml`)
- The robot stands on the **floor** at the origin; the table (top at z = 0.72, `reorient.TABLE_Z`) is in front
  of it, near edge at x = 0.52. Why: a *horizontal* hand (needed to flip the cup by rolling the wrist) is only
  reachable at table-like heights: with the base at table level the Panda's joint 6 hits its limit
  (planar scan: hand-horizontal needs TCP >= ~0.6 m above the base). The near edge is pushed back because
  link 6 / the wrist hangs below the hand line and otherwise jams on the edge (311 N contact seen).
- Cup: **hollow open-top** cup, 9 cm tall, outer diameter ~6.2 cm, straight wall, built from 16 box wall
  segments + a base (`scripts/make_cup_xml.py` regenerates `cup.xml`). Wall thickness **8 mm**: 4 mm walls made
  the solver explode when pinched. Body origin at the geometric center; nominal start (0.6, 0, 0.765), upright.
  Mass 0.10-0.30 kg (inertia rescaled with mass), sliding friction 0.6-1.2, xy jitter +-2 cm.
- Sim: dt **0.5 ms**, `implicitfast`, elliptic cones, `impratio=10`. 2 ms and 1 ms both gave contact chatter /
  blow-ups with heavy cups (see experiment note).
- Gripper: stock Panda hand with the patches in [repository-layout.md](repository-layout.md).
- Keyframe `ready`: side pre-grasp pose (hand horizontal along +x, fingers open along world y, TCP 8 cm in
  front of the cup), solved offline with `policies/cartesian.solve_ik`.

## Object shapes
`PandaCupEnv(cup_shape=...)`: `"cup"` (default, hollow, 9 cm) or `"cylinder"`: thin **solid** cylinder, 4.5 cm diameter x
14 cm tall, like the object in the team's reference video (the real size is unknown; only the ~3.2:1 height:width ratio was
measured from the video, the absolute size is an assumption, edit `CYL_R, CYL_H` in `scripts/make_cup_xml.py`). Both shapes live
in `cup.xml` as geoms of the same body; the env switches the unused one off (collision + alpha). The cylinder geom is
compiled with `mass=0` and the env sets mass/inertia/COM itself (analytic inertia), because geoms with
`contype=conaffinity=0` are dropped from MuJoCo's collision structures and cannot be enabled later. The object starts on
the table at table height + half its height. `flip_success` uses `env.cup_half_height`.

## Grippers
`PandaCupEnv(gripper="panda")` is the stock hand (scene.xml). `gripper="robotiq"` mounts the Robotiq 2F-85
(scene_robotiq.xml + envs/model.py); the env action is the same (7 joint targets + gripper in [0, 1], 1 = open; the
Robotiq actuator is inverted internally). Robotiq specifics: TCP 14.15 cm from the flange (vs 10.3 cm), 3.7 cm tall
pads, pads friction 0.6-0.7 (cup friction 1.0 overrides via max rule), table near edge moved from x = 0.52 to 0.565
(the gripper body is up to 4.2 cm below the TCP axis and otherwise catches the table). The ready pose is found by IK and
the redundancy is resolved so joint 7 starts near -1.5 rad, leaving room for the +pi wrist roll. `rot_tol` of the flip
was relaxed from 0.05 to 0.10 rad because the heavier gripper sags a little under load.

## `PandaCupEnv`
- Control 50 Hz (40 substeps). **Provisional action**: 7 absolute joint-position targets + gripper in
  [0, 1] (1 = open). Observation: arm+finger qpos (9), qvel (9), cup pos (3), cup quat (4).
  No reward/termination yet. Revisit the action space (e.g. delta joint targets) for milestone 2.
- `reset(seed)` randomizes cup mass, friction and xy position. Cup always starts upright (start-orientation
  randomization is TODO). Default render camera looks at the table.

## Rigid flip (`policies/rigid_flip.py`)
Side grasp (pinch across the cup diameter at mid height, cup 1 cm deeper than the pad center), lift 15 cm,
**roll the wrist 180 deg**, lower, release, retract. Details that mattered:
- The roll is a **pure joint-7 motion** (min-jerk, 2 s). Rolling through Cartesian IK made the other joints
  wobble and dragged the cup 30 deg out of the grasp; joint-7-only gave ~179 deg cup rotation.
- *Lower* servoes the **cup** (measured pose) to its start xy / table height, because the cup sits ~1 cm higher
  in the grip than the TCP, so TCP-based lowering released it in mid-air.
- *Retract* goes straight back along the approach axis from the current TCP: the open fingers clear the cup by
  only ~9 mm per side, any sideways drift knocks it over.
- Success (`flip_success`): cup axis within 15 deg of straight down, cup center within 3 cm of its start xy,
  resting on the table (z < table + 6 cm). `RigidFlip.grasped` = cup lifted >= 7.5 cm and within 3 cm of the TCP
  after the lift phase.
- By construction this baseline has no in-hand slip, so rho = 0 and wrist rotation = 180 deg: the reference
  point for the efficiency comparison.

## Known caveats
- Failures (~2%): after release the cup rolls/tips off its rim (end state lying on the table or off it).
- Far outside the randomization range the grasp can fail and the solver can go unstable.
- Headless rendering: NVIDIA EGL hung on the dev machine; `scripts/run_rigid_flip.py` defaults to Mesa
  software EGL when `/usr/share/glvnd/egl_vendor.d/50_mesa.json` exists.
- Stock Panda fingers worked for this side grasp of a 6.2 cm cup. Whether they are deep enough for a
  controlled-slip (pivot) baseline is still open; a Robotiq 2F-85 (also in Menagerie) is the fallback.
