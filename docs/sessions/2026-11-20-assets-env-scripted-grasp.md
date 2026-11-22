# 2026-11-20: assets, env, scripted grasp (agent session, requested by CL)

**Goal.** Milestone 1 first step: find assets, set up env, get successes on a scripted grasp.

**Done.** uv project (`reorient` package, Python 3.11, mujoco/gymnasium/scipy/imageio); vendored
Panda from Menagerie with patches; table + cup scene; `PandaCupEnv`; Cartesian IK controller;
scripted top-down grasp (100/100 in range); light tests; docs.

**Decisions (flag if the team disagrees).** Stock Panda gripper kept (gripper choice is open per
the proposal) but gripper strength/friction patched; cup is a solid cylinder for now; env action
is absolute joint targets (provisional); top-down grasp chosen as simplest first success.

**Problems.** Stock gripper too weak (~3 N); NVIDIA EGL hangs headless here -> Mesa EGL fallback;
instability at 0.6 kg / mu 0.3 (out of range).

**TODO / next.** Pivot and rigid-flip scripted baselines (the actual reorientation); hollow cup
mesh; randomized start orientation (e.g. lying/tilted cup); define target pose; roadmap entry.
`git init` was run in the repo (no remote; none invented).

## Update: task is cup *inversion*
AGENTS.md now states the task is inverting the cup (~180 deg). Implications noted: the solid
cylinder is top/bottom symmetric so a hollow open-top (maybe tapered) cup is needed; grasp height
must be tunable for the pivot baseline; target orientation = cup axis pointing down (<15 deg).
Added `docs/project/overview.md`. The grasp/lift result stands (upright start, top-down pinch).

## Update 2: hollow cup, side grasp, rigid flip working
User answers: target = upside-down near its start position; try the stock hand first (finger depth is a
concern; an adaptive gripper was floated: Robotiq 2F-85 from Menagerie is the fallback).
Done: hollow cup (8 mm box-ring walls), table raised to 0.72 m with the robot on the floor (needed for a
horizontal hand), side grasp, `RigidFlip` baseline: grasp+lift 200/200, flip 196/200. The old top-down grasp
and its script/test were removed (unreachable at the new table height); its experiment note is marked
superseded. Details and the debugging story: [rigid-flip experiment](../experiments/2026-11-20-rigid-flip.md).
Decisions to flag: dt 0.5 ms; cup placement x = 0.6; wall thickness 8 mm; robot not on the table.
Next: pivot baseline (controlled slip), diagnose release failures, start-orientation randomization.

## Update 3: AGENTS.md refined (pivot is the target behavior)
Target behavior is now explicit: pinch near the bottom end, lift, cup pivots about the fingertips under gravity
with no wrist rotation, set down inverted. The existing `RigidFlip` (mid-height grasp, 180 deg wrist roll) is
exactly the "direct solution" the efficiency reward should beat, so it stays as the rigid-flip baseline. The
pivot baseline is next and is the real test of the stock fingers.

## Update 4: pivot probe (stock hand) -> blocked on finger depth
Probed the pivot with the stock hand: the cup tips ~30 deg then hits the palm; off-axis pinches throw the cup out.
No pivot baseline yet. Details and options: [pivot probe](../experiments/2026-11-20-pivot-feasibility-probe.md).
`scripts/make_cup_xml.py` now also takes the cup mean radius as a third argument.

## Update 5: Robotiq 2F-85 integrated
Vendored Robotiq 2F-85 (Menagerie), `envs/model.py` attaches it with MjSpec, `PandaCupEnv(gripper=...)`,
`scripts/run_rigid_flip.py --gripper`. Rigid flip with Robotiq: 200/200. Pivot with Robotiq: geometry fine now, but the
cup sticks by friction; see [pivot probe part 2](../experiments/2026-11-20-pivot-feasibility-probe.md). Bug worth
remembering: the tcp site must carry its own rotation (rotating only the attach frame cancelled out and the fingers
opened vertically).

## Update 6: thin cylinder like the reference video
`cup_shape="cylinder"` added (assumed 4.5 x 14 cm). Pivot probe with the stock hand: swings to an inverted hang in some
runs (seed 0 reaches ~160-200 deg with no wrist motion) but is unreliable and the object slips out afterwards. Videos in
`outputs/videos/`. The reference video itself sits (untracked) in the repo root; it is large (15 MB), AGENTS.md says it is
not kept in the repo, so I unstaged it (did not delete or ignore it). Also deleted the obsolete `outputs/scripted_grasp.mp4`.
Next: grip-force schedule through the swing, then lower/place inverted; later try the hollow cup again.

## Update 7 (WidowX width pivot; details in the pivot feasibility experiment note, Parts 5-14)
- The inverted-pivot baseline did not work with the stock Panda hand or the Robotiq; with the WidowX 250 S and primitive pads it works: IK + finger width control
  (lift, tilt the gripped cylinder, ease the width off while pulling the hand back, close near the end of the rotation, place by aligning the lowest point, release while retracting).
  Mass 0.15-0.25 kg and COM within 1/4 of the cylinder length on the axis: 1299/1300 on random held-out scenes. This needs the team's decision on arm / gripper (not decided).
- Found: runtime mass / inertia / COM edits need `mj_setConst`; the Panda env's reset does not call it (calling it fails the Robotiq rigid-flip test, so it was not changed): to be decided.
- New scripts: `run_width_pivot.py`, `search_width_pivot.py`, `sweep_width_pivot.py`, `probe_widowx.py`, assets `wx250s/` + `scene_widowx.xml`. Nothing is committed.

## Update 8 (2026-11-21: metrics proposal, scripted baselines on the metrics, PPO task-only vs task + efficiency)
- Metrics and reward proposal: `docs/project/metrics.md` (needs the team's decision: thresholds, reward weights, arm-link travel vs joint travel).
- WidowX RL environment `WidowXFlipEnv` (episode from the pickup state), PPO training / evaluation / video scripts, tests (`tests/test_widowx_env.py`). SB3 + PyTorch added to the project (`pyproject.toml`, `uv.lock`).
- Scripted baselines on the metrics (200 random scenes): both 100 % success; the pivot has wrist rotation 0.04 vs 3.14 rad and rho 0.91, but is slower than the roll (7.6 vs 5.3 s).
- PPO, 5 seeds per condition, matched drop penalty 10 (see the experiment note): task + efficiency 96.7 % success (5/5 seeds), wrist rotation 0.09 rad, rho 0.90, joint path 3.3 rad, always the gravity swing; task only 40 % (2/5 seeds learn: one swing, one rigid arm-sweeping rotation with wrist 4.4 rad and rho 0.07), pooled wrist rotation 3.0 rad, joint path 11.2 rad. With drop penalty 5 task-only reaches 3/5 seeds (59.9 %). Efficiency terms with drop penalty 5 collapse (immediate drop): the termination penalty must exceed the full-horizon time cost.
- PPO found four loopholes in the reward / success definitions (laying the cylinder down, parking it upright, flip-and-drop, throwing it down onto the table): thresholds tightened, documented in the note and in `metrics.md`.
- Found: runtime mass / COM edits need `mj_setConst` (fixed in the WidowX code; `PandaCupEnv.reset` still lacks it and fixing it fails the Robotiq rigid-flip test: open).
- Open for the team: arm / gripper choice (WidowX with custom pads vs Panda), reward weights, a fair task-only ablation, nothing committed.

## Update 9 (2026-11-22: learning approach + grasp, first commit)

Requested: let the policy learn approach and grasp (not start from the pickup state), first with one mass and a centred COM. Added `learn_grasp` mode to `WidowXFlipEnv` and many reward / action-space iterations (17 variants, table in `experiments/2026-11-22-learned-approach-grasp.md`); the best policy reaches the grasp pose but has not yet closed on the cup. User decision (kept): no demos / Cartesian mode, learn it from rewards. Committed the work so far, prioritising the pickup-start policies (`outputs/weights/task_vs_eff/`); the learned-grasp part is a negative / partial result and can be revisited ("focus on a policy that grasps at that specific moment later"). Remote `origin` configured at the user's request.
