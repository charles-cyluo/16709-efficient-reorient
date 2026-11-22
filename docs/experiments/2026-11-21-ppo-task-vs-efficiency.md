# PPO on the WidowX flip task: task reward only vs task + efficiency terms

**Date:** 2026-11-21. **Question (project):** if the policy is rewarded for efficiency (time, arm travel, wrist rotation), does it exploit gravity-driven in-hand slip instead of the 180 deg wrist roll?
**Code:** `src/reorient/envs/widowx_flip_env.py` (env, reward), `src/reorient/eval/metrics.py` (metrics), `scripts/train_ppo.py`, `scripts/eval_policy.py`, `scripts/render_policy.py`.
Environment, reward and metric definitions: [`../implementations/widowx-flip-env.md`](../implementations/widowx-flip-env.md), [`../project/metrics.md`](../project/metrics.md). Git revision: none yet (the repository has no commit; the working tree is what ran).

## Set-up
WidowX 250 S + thin cylinder (4.5 x 14 cm), episode from the pickup state (cylinder gripped, lifted 16 cm), mass uniform in 0.15-0.25 kg, COM on the axis within +-1.75 cm, fixed start position,
10 s limit, 50 Hz control, action = 6 joint increments + gripper squeeze. PPO (Stable-Baselines3 2.9.2), MLP 256x256, 15 parallel environments (subprocesses, CPU physics), gSDE exploration (sample every 8 steps), initial log-std -1, entropy coefficient 0,
n_steps 512, batch 3840, 10 epochs, lr 3e-4, gamma 0.995, lambda 0.95, observation and reward normalization, 10M steps (about 1 h on 15 cores, policy updates on the A5000).
**Reward:** task terms = potential-based shaping toward the inverted pose at the start position (inversion, position, height gated on inversion, release, retreat) + success bonus 20 - drop penalty. Efficiency run adds
-0.02 per step (time), -0.5 per rad of joint travel, -2 per rad of wrist-roll motion (weights are proposals, `metrics.md` section 5).
**Commands:**
`uv run python scripts/train_ppo.py --reward efficiency --steps 10000000 --n-envs 15 --batch-size 3840 --seed 0 --name task_eff --sde --sde-freq 8 --log-std-init -1.0 --ent-coef 0.0`
`uv run python scripts/train_ppo.py --reward task --steps 10000000 --n-envs 15 --batch-size 3840 --seed 0 --name task_only --weights '{"drop": 5.0}' --sde --sde-freq 8 --log-std-init -1.0 --ent-coef 0.0`
`uv run python scripts/eval_policy.py --runs outputs/runs/2026-11-21-task_only-0 outputs/runs/2026-11-21-task_eff-0 --names task_only task_plus_eff --baselines baselines_v3 --n 200 --tag policies_v2`
Outputs: `outputs/runs/<date>-<name>-<seed>/` (config, progress.csv, checkpoints, final.zip, vecnormalize.pkl), `outputs/eval/policies_v2.{json,md}`; videos `outputs/videos/rl_*.mp4` (git-ignored). Small final policies: `outputs/weights/task_vs_eff/`.

## Result 1: five seeds per condition, same budget (6M steps), 200 random scenes per policy, deterministic policies
`uv run python scripts/eval_policy.py --groups --ckpt-steps 6000000 --n 200 --baselines baselines_v3 --tag policies_5seeds --runs <10 run dirs> --names task_only/s0 ... task_plus_eff/s4` (output `outputs/eval/policies_5seeds.{json,md}`).
Seeds 0 and 1 were trained for 10M (seed 0) / stopped at 6M (seed 1); all are evaluated at their ~6M-step checkpoint; seeds 2-4 trained for 6M.
**Caution: in this table the task-only runs use drop penalty 5 and the task + efficiency runs use drop penalty 10** (see Result 2 for the matched comparison).

| metric | scripted pivot | scripted roll | RL task only (5 seeds pooled) | RL task + efficiency (5 seeds pooled) |
|---|---|---|---|---|
| success rate (95% CI) | 100.0% (98.1-100.0) | 100.0% (98.1-100.0) | 59.9% (56.8-62.9) | 96.7% (95.4-97.6) |
| drop rate | 0.0% | 0.0% | 0.1% | 3.1% |
| completion time [s] | 7.62 +- 0.57 | 5.31 +- 0.01 | 0.88 +- 0.16 | 1.08 +- 0.57 |
| TCP path [m] | 0.36 +- 0.01 | 0.27 +- 0.00 | 0.55 +- 0.09 | 0.53 +- 0.13 |
| arm-link travel [m] | 1.37 +- 0.07 | 0.99 +- 0.00 | 1.35 +- 0.14 | 1.04 +- 0.37 |
| joint path [rad] | 3.58 +- 0.47 | 5.36 +- 0.00 | 7.42 +- 1.75 | 3.32 +- 0.82 |
| wrist rotation [rad] | 0.04 +- 0.04 | 3.14 +- 0.00 | 1.68 +- 0.72 | 0.09 +- 0.07 |
| gravity-assist ratio rho | 0.91 +- 0.06 | 0.01 +- 0.00 | 0.84 +- 0.05 | 0.90 +- 0.03 |
| release height [mm] | -0.2 +- 0.1 | -0.3 +- 0.1 | -1.1 +- 1.3 | -0.2 +- 1.4 |
| downward speed at release [m/s] | 0.00 +- 0.00 | 0.00 +- 0.00 | 0.00 +- 0.00 | 0.01 +- 0.02 |
(efficiency rows: means over the successful episodes; +- is the spread over episodes.)

Per seed (success; means over the successful episodes: completion time [s], wrist rotation [rad], joint path [rad], arm-link travel [m], rho):
| run | success | time | wrist rot. | joint path | link travel | rho |
|---|---|---|---|---|---|---|
| task only, seed 0 | 0.0% | - | - | - | - | - |
| task only, seed 1 | 100.0% | 0.74 | 1.66 | 7.56 | 1.43 | 0.84 |
| task only, seed 2 | 0.0% | - | - | - | - | - |
| task only, seed 3 | 100.0% | 1.02 | 2.53 | 9.32 | 1.36 | 0.78 |
| task only, seed 4 | 99.5% | 0.88 | 0.83 | 5.39 | 1.26 | 0.89 |
| task + efficiency, seed 0 | 97.0% | 1.10 | 0.09 | 3.71 | 1.00 | 0.88 |
| task + efficiency, seed 1 | 100.0% | 0.63 | 0.05 | 2.31 | 0.52 | 0.90 |
| task + efficiency, seed 2 | 89.5% | 1.44 | 0.07 | 4.43 | 1.44 | 0.87 |
| task + efficiency, seed 3 | 100.0% | 0.63 | 0.05 | 2.97 | 1.39 | 0.88 |
| task + efficiency, seed 4 | 97.0% | 1.63 | 0.21 | 3.31 | 0.91 | 0.94 |

**What the policies do** (videos `outputs/videos/rl_task_plus_eff_seed4.mp4`, `rl_task_only_seed3.mp4`; frames checked): every successful policy, with or without efficiency terms, solves the task with the **gravity swing, not the 180 deg wrist roll**: a quick forward-downward hand motion, the cylinder pivots about the pads
past horizontal into an inverted hang, the hand lowers it onto the table, opens, and the cylinder stands, 0.6-1.6 s after the pickup (scripted pivot 7.6 s, scripted roll 5.3 s). Without efficiency terms the arm adds wasted motion: the forearm-roll and wrist-roll joints counter-rotate
(e.g. +1.4 / -1.2 rad in a seed 1 rollout, no net effect) and the joint path is twice as long. The two failing task-only seeds invert the cylinder and hold it in the air until the time limit, or hold still (success 0 %).

## Result 2: matched comparison (same drop penalty 10), 5 seeds each, 6M steps, 200 scenes per policy
Result 1 compared task-only with drop penalty 5 against task + efficiency with penalty 10. Task + efficiency with penalty 5 collapses (see below), so the matched comparison is task-only with penalty 10.
`uv run python scripts/eval_policy.py --groups --ckpt-steps 6000000 --n 200 --baselines baselines_v3 --tag policies_matched --runs <task_only_drop10-0..4, task_eff-0..4> ...` (output `outputs/eval/policies_matched.{json,md}`).

| metric | scripted pivot | scripted roll | RL task only, drop 10 (5 seeds pooled) | RL task + efficiency, drop 10 (5 seeds pooled) |
|---|---|---|---|---|
| success rate (95% CI) | 100.0% (98.1-100.0) | 100.0% (98.1-100.0) | 40.0% (37.0-43.1) | 96.7% (95.4-97.6) |
| drop rate | 0.0% | 0.0% | 0.1% | 3.1% |
| completion time [s] | 7.62 +- 0.57 | 5.31 +- 0.01 | 1.20 +- 0.41 | 1.08 +- 0.57 |
| TCP path [m] | 0.36 +- 0.01 | 0.27 +- 0.00 | 0.81 +- 0.06 | 0.53 +- 0.13 |
| arm-link travel [m] | 1.37 +- 0.07 | 0.99 +- 0.00 | 1.93 +- 0.14 | 1.04 +- 0.37 |
| joint path [rad] | 3.58 +- 0.47 | 5.36 +- 0.00 | 11.19 +- 1.68 | 3.32 +- 0.82 |
| wrist rotation [rad] | 0.04 +- 0.04 | 3.14 +- 0.00 | 3.00 +- 1.40 | 0.09 +- 0.07 |
| gravity-assist ratio rho | 0.91 +- 0.06 | 0.01 +- 0.00 | 0.45 +- 0.39 | 0.90 +- 0.03 |
Per seed, task only (drop 10): seeds 1, 3, 4: 0 % (hold still / invert and hold); **seed 0: 100 %, gravity swing** (time 0.84 s, wrist 1.63 rad, rho 0.84, in-hand rotation 148 deg); **seed 2: 100 %, rigid rotation** (time 1.56 s, wrist 4.37 rad, joint path 12.7 rad, rho 0.07, in-hand rotation 12 deg:
the hand carries the cylinder through a large arm sweep and the arm is flung aside after the release; video `outputs/videos/rl_task_only_drop10_seed2_wrist_roll.mp4`, frames checked). Task + efficiency: all five seeds use the gravity swing (rho 0.87-0.94).

## How it got here (every attempt, in order; the loopholes are the interesting part)
| run | change | outcome |
|---|---|---|
| task only, default PPO, drop 10 | first attempt | **lays the cylinder on its side on the table** (slams the hand down, rolls the wrist): earns height + half the inversion shaping and avoids the "airborne" drop test -> toppled-on-the-table now counts as a drop |
| task only, drop 10 | + toppled = drop | plateau: **parks the upright cylinder on the table** and holds still (height shaping paid for any orientation) -> the height term is gated on inversion |
| task only, gSDE, log-std -1 | + gated height | still plateau: holds still / occasional drops, 0 % success after 6M steps |
| task only, drop 2 | smaller drop penalty | **flips the cylinder in the air and lets go** (inversion shaping 3.4 > penalty 2) |
| task only, drop 5, 3 cm airborne limit | | inverts, holds; and the efficiency policy (same env) **throws the cylinder down from about 1.5 cm** at 0.5 m/s: 99 % "success" with completion 0.53 s, but 0 % when scored with the strict definition -> airborne limit 5 mm, release speed limit 0.2 m/s |
| both, strict definitions | the runs of Result 1 (task only: drop 5, 5 seeds; task + efficiency: drop 10, 5 seeds) | task only: 3 of 5 seeds learn (seeds 0 and 2: 0 %); task + efficiency: 5 of 5 seeds learn |
| task + efficiency with drop 5 (matched to the task-only runs), 5 seeds | drop penalty 5 | **collapse in all 5 seeds**: every episode ends after 1 step with a drop (return -5.02): with a time penalty of 1 per second a 10 s episode costs about 10, so ending it at once for 5 is the best option. The termination penalty must exceed the cost of the full horizon (here 10); the matched comparison is therefore task only with drop 10 (Result 2) |
Earlier runs are kept in `outputs/runs/*_v*` (git-ignored).

## Conclusions
- **The efficiency reward selects the gravity-assisted solution and makes learning reliable.** Task + efficiency: 5/5 seeds learn, pooled success 96.7 % (goal >= 80 %), wrist rotation 0.09 rad, rho 0.90 (goal >= 0.7), joint path 3.3 rad, all with the gravity swing; it is also much faster than both scripted controllers (1.1 s vs 5.3 / 7.6 s) and its arm-link travel is similar to the scripted roll.
- **Without efficiency terms the outcome is erratic.** Task only (drop 10): 2/5 seeds learn (pooled 40 %), one with the gravity swing and one with a rigid, arm-sweeping rotation (wrist 4.4 rad, rho 0.07) - the naive rotation-heavy solution the project expects; pooled wrist rotation 3.0 rad (33x the efficiency policy), joint path 11.2 rad (3.4x). With drop penalty 5, 3/5 seeds learn (pooled 59.9 %, wrist 1.7 rad, joint path 7.4 rad). So the "RL without efficiency reward" ablation exists and is worse on every efficiency metric and on reliability.
- **The efficiency terms also act as an exploration aid:** the time penalty turns "hold still" (free in the task-only reward) into a loss; this is part of why all efficiency seeds learn. The effect cannot be separated from the efficiency preference without term-by-term ablations (time only / wrist only / joint travel only), which were not run.
- **Reward design traps found by PPO:** laying the cylinder down, parking it upright, flip-and-drop, throwing it down from about 1.5 cm (each closed in the metric / reward definitions); a task-only result depends strongly on the drop penalty (2: flip-and-drop; 5: 3/5 seeds; 10: 2/5 seeds); with a per-step cost the termination penalty must exceed the cost of the full horizon (efficiency with drop 5: every episode ends in one step with a drop, 5/5 seeds).
**Caveats.** 5 seeds per condition (a seed is a Bernoulli trial for "does it learn": 2/5 vs 5/5 is suggestive, not significant: Fisher exact p about 0.17); pooled success rates mix seeds and are not independent samples; the pickup is scripted; WidowX with custom pads (the arm choice is open); fixed start and no friction randomization; reward weights are proposals;
efficiency numbers are over successful episodes only; all comparisons at the ~6M-step checkpoint, deterministic policies, scenes drawn from the same distribution as the scripted baselines but different samples.
**Next (not done):** term-by-term efficiency ablation; more seeds; SAC; a task-only ablation with a different exploration aid (curriculum, demonstrations) to see whether it reliably finds the wrist roll; friction randomization; the Panda hand.
