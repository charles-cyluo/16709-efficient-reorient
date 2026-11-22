# WidowX flip environment and metrics (RL set-up)

Code: `src/reorient/envs/widowx_flip_env.py` (`WidowXFlipEnv`), `src/reorient/envs/widowx_sim.py` (`Sim`: model, IK helpers, scene configuration),
`src/reorient/eval/metrics.py`, `src/reorient/eval/aggregate.py`, `src/reorient/policies/widowx_expert.py` (scripted roll through the env interface),
training `scripts/train_ppo.py`, evaluation `scripts/eval_policy.py`, scripted controllers `scripts/run_width_pivot.py` (+ `eval_baselines.py`, `search_width_pivot.py`).
Metric and reward definitions: [`../project/metrics.md`](../project/metrics.md). Scene: WidowX 250 S on the table, thin cylinder 4.5 x 14 cm, round 10 mm pads,
torsional friction 0.004 on pads and cylinder, noslip 10, physics step 0.5 ms (see the pivot feasibility note, Part 14).

## Episode
Starts at the **pickup** moment: a scripted approach / close / lift (16 cm, squeeze 12 mm) is simulated once per process (`pickup_snapshot`) and every reset restores that state, then
draws the cylinder's mass (uniform 0.15-0.25 kg) and the height of its centre of mass on the axis (uniform +-1.75 cm), calls `mj_setConst` (needed after runtime mass edits, see the pivot note) and settles for
0.2 s. The policy must invert the cylinder, put it down at its start position, release it and clear the robot. Time limit 10 s (500 control steps of 20 ms; 40 physics steps each).
Termination: success (condition of `metrics.md` held for 0.5 s), drop; truncation at the time limit.

## Spaces
- Action, 7 in [-1, 1]: joint-position increments of the 6 arm joints (|a| = 1 -> 0.06 rad per control step, command clipped to the joint limits and to +-0.35 rad around the measured joint), and the gripper:
  finger target = `q_contact - (0.009 + 0.015 a)` (a = 0 firm grip, 9 N; a = -1 opens by 6 mm; a = +1 squeezes by 24 mm). The zero action keeps the cylinder held.
- Observation, 38: arm q (6), arm dq (6), finger q (2), finger dq (2), cylinder position (3), quaternion (4, w >= 0), linear velocity (3), angular velocity (3), time fraction (1), previous action (7), robot-contact flag (1).
  (`AGENTS.md` asks for joint state + accurate cylinder pose; velocities, the previous action and the contact flag are additions because the swing and the friction limit are fast.)
- Reward terms: see `metrics.md` section 5; `info["reward_terms"]` has every term, `info["episode_metrics"]` the metric dictionary at the end of an episode.

## Known simplifications (team to confirm)
- The pickup is scripted and not part of the learning problem (it is the same for every policy; the efficiency metrics count after it).
- WidowX 250 S with primitive pads instead of the Panda (the Panda hands do not give a controllable pivot, see the pivot note); fixed start position; no friction randomization.
- `PandaCupEnv` (stock hand / Robotiq rigid flip baselines) is unchanged. Its reset edits mass and inertia without `mj_setConst`: not changed because that makes the Robotiq rigid-flip test fail (open item).

## `learn_grasp=True` mode (experimental: does not learn the grasp yet)

Episode starts from a ready pose (gripper open, TCP 12 cm behind / 7 cm above the assigned grasp point) instead of the pickup snapshot; the policy must approach, grasp and lift. Differences from the default mode:
- **Gripper action** (index 6) is a finger-target **rate** (1.5 mm per control step, -1 opens, +1 closes). Closing is ignored unless the TCP is within 2 cm of the assigned grasp position (anisotropic distance, see below), a pad touches the cup, or the cup has been lifted. Opening always works.
- **Assigned grasp pose:** cup centre + (-8 mm, 0, `GRASP_DZ` = -3.5 cm), orientation `R_SIDE`. (At the scripted pickup's -4.5 cm the level hand's wrist link rests on the table.) Distance is anisotropic, x*1, y*4, z*2: the open gripper clears the cup by only about 5.5 mm per side in y.
- **Observation** 45 dims (default 38): + TCP position, cup-minus-TCP vector, finger target; `privileged_obs=True` appends mass and COM offset (default off).
- **Workspace limits** on the commanded joints: base yaw +-1 rad, forearm roll +-1 rad (wrist_rotate keeps its full range, so the wrist-roll solution stays available).
- **Pre-grasp reward terms** (potential-based, exponential kernels, frozen at their maximum once the cup is lifted so the pivot is not penalised): `approach` (w 6: 0.3 far-field slope + 0.35 exp(-d/10 cm) + 0.35 exp(-(d/3 cm)^2)), `orient` (w 4: exp(-angle/0.3 rad) of the TCP frame vs `R_SIDE`), `close` (w 3: finger closure times exp(-(d/1.5 cm)^2)), `grasp` (w 3: both pads touch the cup), `lift` (w 4: cup height up to 16 cm, fades out as the cup inverts). Non-potential: `table_contact` (-0.02 per step while any robot geom touches the table), `joint_limit` (-0.05 per joint within 10 % of a limit, shoulder / elbow / wrist pitch only).
- Weights are untuned proposals. Results and the full history of failed variants: [../experiments/2026-11-22-learned-approach-grasp.md](../experiments/2026-11-22-learned-approach-grasp.md).
- Flags: `scripts/train_ppo.py --learn-grasp --fixed-scene --max-time 15 --max-dq 0.02 [--privileged-obs] [--resume RUNDIR --gripper-log-std X --reset-gripper-head]`. After `--resume`, `--steps` counts additional steps.
