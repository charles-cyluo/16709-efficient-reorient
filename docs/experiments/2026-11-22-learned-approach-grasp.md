# Learning approach + grasp with PPO from scratch (fixed scene): negative result so far

Status: **not working yet.** Best policy (task + efficiency) learns to reach the grasp pose to a few mm but has not yet closed the gripper on the cup (0 % success, 0 lifts) after ~25M steps; see Part 2. Written by Claude Code during a session with the user; owner of the next decision: user / team.

## Question
The first PPO comparison ([2026-11-21-ppo-task-vs-efficiency.md](2026-11-21-ppo-task-vs-efficiency.md)) starts each episode at the moment of pickup (scripted approach, grasp, 16 cm lift). Can the policy also learn approach, grasp and lift?

## Setup
`WidowXFlipEnv(learn_grasp=True)` (see [../implementations/widowx-flip-env.md](../implementations/widowx-flip-env.md)): episode starts at a ready pose (TCP 12 cm behind and 7 cm above the grasp point, gripper open, +-0.03 rad joint noise). Fixed scene: mass 0.2 kg, COM centred (`--fixed-scene`). Horizon 15 s, drop penalty 20 (full-horizon time cost is 15), all efficiency terms counted from episode start. Hidden mass/COM (no privileged observation). PPO as before (15 envs, gSDE, 256x256), seed 0, task-only and task+efficiency each, 15M steps planned; every run was stopped early (1.6M-9M steps) once the failure was clear.

Command (last variant): `uv run python scripts/train_ppo.py --reward {task|efficiency} --learn-grasp --fixed-scene --max-time 15 --steps 15000000 --n-envs 15 --sde --sde-freq 8 --log-std-init -1 --ent-coef 0 --weights '{"drop":20}' --seed 0 --name <variant>`. No git revision (nothing committed). Run directories were deleted (nothing worth keeping); results below are from the training logs and checkpoint replays.

## Attempt history (each row: change -> what the replayed policy did)
| # | Change | Outcome (success 0% in every run) |
|---|---|---|
| 1 | absolute gripper target; 2D approach potential (TCP-cup xy), grasp (both pads touch), lift terms | approaches in xy, never closes: closing slams the gripper and topples the cup (-20), so it learns not to close |
| 2 | gripper action = finger-target rate (1.5 mm/step) | same plateau |
| 3 | + `close` term (aligned in xy) | hovers 23 cm above the cup and closes in the air (xy-only reward) |
| 4 | + height gate on `close` | hovers 40 cm above the cup |
| 5 | + `descend` term (aligned only) | hovers 15 cm above the cup top, gripper working in the air |
| 6 | user: 3D distance to an assigned grasp position replaces approach/descend | reaches grasp height, then learns to close early (9 cm away) and stalls with closed fingertips against the cup front |
| 7 | closing only allowed within 6 cm of the grasp position (env rule) | stalls 4 cm short, one pad off-axis, arm pressing into the table; wrist roll driven to -pi, wrist pitch at its limit |
| 8 | user: consider orientation: `orient` term (TCP frame vs side-grasp frame, until lift) | orientation exact (0.006 rad) but elbow too bent (q3 1.02 vs 0.72 scripted), wrist pitch at limit, palm box dragging on the table 8.5 cm short of the cup |
| 9 | + `table_contact` penalty (-0.1/step, any robot geom on the table) | retreats to 25 cm behind the cup in a mirrored arm configuration (joints at limits, self-contact): avoids the penalty by staying away |

## Conclusions
- The reward-term chain (approach, close, grasp, lift) is learnable in pieces but PPO in joint space finds a different degenerate posture each time (hover, early close, joint-limit pose, table press, retreat). Every patch closed one shortcut and opened another; this is not converging.
- The root cause seems to be exploration in joint space with no posture prior: the grasp needs a specific elbow/wrist configuration and a slow close, found by nothing in the reward before the cup is touched.
- Not tried yet (each changes what "learns the grasp" means, so this is the team's call): (a) scripted-demo warm-start (behaviour cloning on scripted approach/grasp/lift, then PPO fine-tune), (b) Cartesian action space for the approach phase (TCP delta + IK holds orientation, joint actions after the grasp), (c) start-state curriculum (hand at the cup, then already gripped), (d) joint-limit/posture regulariser relative to the ready pose.
- Reward-term weights, the 6 cm close rule and the `table_contact` penalty were never tuned and exist only in the code path `learn_grasp=True`; nothing here changes the numbers of the pickup-start experiment.

## Part 2: exponential tracking rewards (user requirement: the policy must learn approach and grasp, no demos / no Cartesian mode)

The requirement from the user was to keep learning approach + grasp from rewards only (no demo warm-start, no Cartesian action space). Changes, in the order they were made, each from replaying a checkpoint:

| # | Change | Finding |
|---|---|---|
| 10 | exponential tracking potentials (approach, orient, soft close gate), potentials frozen after lift, weights up, joint-limit penalty, table penalty 0.02 | the assigned grasp point (4.5 cm below centre) puts the level hand's wrist link on the table: `GRASP_DZ` -> -3.5 cm |
| 11 | far-field slope in the approach potential, base yaw limit +-1 rad | exp(-d/10 cm) is ~0 far away: policy drifted to 1 m from the cup |
| 12 | forearm roll limit +-1 rad, orient weight 4 | policy sat in a flipped arm configuration (forearm -pi, wrist +pi) |
| 13 | joint step 0.02 rad (was 0.06), `log_std_init` -2 | coarse steps + noise pushed the arm out of the grasp posture; first policy to reach 6 cm from the cup |
| 14 | anisotropic grasp distance (y x4, z x2) | open aperture clears the cup by only 5.5 mm per side: a policy 1.6 cm off in y rightly refused to advance |
| 15 | gripper-only exploration noise (`--gripper-log-std 0`), resume option | efficiency policy reached the pose to ~3 mm (7M steps) but never closed: gripper pre-clip action mean had drifted to -27 (no gradient while open/far) |
| 16 | `--reset-gripper-head` | policy then closed early (close allowed within 6 cm, paid at 5 cm) and stopped reaching the pose |
| 17 | closing allowed only within 2 cm, close gate sigma 1.5 cm; resumed from the 10M-step checkpoint | efficiency policy hovers 1-2 cm from the grasp pose, gripper preference now +35 (closing), finger command reaches only 35 mm (contact at ~30 mm) |

Runs: `grasp_fixed_*` names in `outputs/runs/` (git-ignored; early variants deleted). The latest pair, `grasp_fixed_tight_{task,efficiency}`, resumed from the 10M-step checkpoint of `grasp_fixed_gstd_*` (itself resumed from `grasp_fixed_aniso_*`), planned to ~25M steps; at 18M steps neither had touched the cup with a pad. Task-only (no time penalty) stops about 5 cm short of the cup, the efficiency policy gets within 1-2 cm: the time penalty seems to push the policy forward here. This is a single seed and unproven.

Open: the last millimetres of closing around the cup are not found yet. Ideas not tried: bring back the 6 cm region only for opening/closing penalty-free exploration, a closing reward on finger velocity, longer training, larger noise at the pose only.

Model size (`MlpPolicy` 45 -> 256 -> 256 -> {7, 1}): 158,984 parameters (actor 79,367, critic 77,825, gSDE matrix 1,792); a checkpoint zip with optimizer state is ~2 MB.
