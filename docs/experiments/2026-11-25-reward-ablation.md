# Reward ablation: which efficiency term makes PPO choose the gravity swing? (in progress)

**Date:** started 2026-11-25 (MJ, agent-assisted). **Status: running, no results yet.**
**Question.** `2026-11-21-ppo-task-vs-efficiency.md` found that task + all efficiency terms gives 5/5 seeds that learn the gravity swing (wrist rotation 0.09 rad, rho 0.90), while task only gives 2/5. It could not separate the efficiency *preference* from the *exploration aid* of the time penalty (it makes "hold still" costly). Each efficiency term is trained here on its own.

**Set-up.** Same as the 2026-11-21 runs (WidowXFlipEnv pickup start, PPO MLP 256x256, gSDE every 8 steps, log-std -1, ent 0, 15 envs, batch 3840, 6M steps, drop penalty 10), only the efficiency weights differ.
Config: [`configs/ablation_v1.json`](../../configs/ablation_v1.json). Git revision of the runs: 00f18eb.

| condition | time [per step] | joint travel [per rad] | wrist rotation [per rad] |
|---|---|---|---|
| `task_only` | 0 | 0 | 0 |
| `task_eff` | 0.02 | 0.5 | 2 |
| `eff_time_only` | 0.02 | 0 | 0 |
| `eff_wrist_only` | 0 | 0 | 2 |
| `eff_travel_only` | 0 | 0.5 | 0 |

**Commands.**
`uv run python scripts/sweep_ppo.py --config configs/ablation_v1.json --conditions eff_time_only eff_wrist_only eff_travel_only --seeds 0 1 2 3 4 --seed-major --parallel 1`
Evaluation (when done): `uv run python scripts/eval_policy.py --groups --ckpt-steps 6000000 --n 200 --baselines baselines_v3 --tag ablation_v1 --runs <run dirs> --names <condition>/s<seed> ...`, figures `uv run python scripts/plot_eval.py --eval outputs/eval/ablation_v1.json --conditions ...`.
Outputs: `outputs/runs/2026-11-2*-<condition>-<seed>/`, `outputs/eval/ablation_v1.{json,md}`, `outputs/figures/ablation_v1/`.

## Results

### Partial 1 (2026-11-25 21:40): `eff_time_only` seed 0
Run `outputs/runs/2026-11-25-eff_time_only-0` (6M steps in 65 min). Compared with the committed seed-3 policies of the 2026-11-21 note (not re-trained yet) on 200 scenes, deterministic, final policy:
`uv run python scripts/eval_policy.py --groups --runs outputs/runs/2026-11-25-eff_time_only-0 outputs/runs/committed-task_only-3 outputs/runs/committed-task_eff-3 --names eff_time_only/s0 task_only/s3 task_eff/s3 --n 200 --n-envs 10 --baselines baselines_v3 --tag ablation_partial_1`
(`outputs/eval/ablation_partial_1.{json,md}`, figure `outputs/figures/ablation_partial_1/metrics_bars.png`, video `outputs/videos/rl_eff_time_only_seed0.mp4`).

| metric | scripted pivot | scripted roll | time only, s0 | task only, s3 | task + all eff., s3 |
|---|---|---|---|---|---|
| success | 100 % | 100 % | 100 % | 100 % | 100 % |
| completion time [s] | 7.62 | 5.31 | 1.08 | 1.06 | 0.64 |
| joint path [rad] | 3.58 | 5.36 | 11.32 | 9.47 | 2.97 |
| wrist rotation [rad] | 0.04 | 3.14 | 3.21 | 2.58 | 0.05 |
| rho | 0.91 | 0.01 | 0.16 | 0.79 | 0.88 |

**Reading (one seed, preliminary).** The time penalty alone gives a reliable, fast policy (100 %, 1.1 s), but it inverts the cylinder by **rigid rotation** (wrist rotation 3.2 rad, rho 0.16; in the video frames the cylinder stays aligned with the gripper through the flip), the solution the project expects without efficiency pressure. So in this seed the time term does not select the gravity swing; the joint-travel / wrist terms would be what does. The 2026-11-21 note's question (preference vs exploration aid) gets a first answer: time acts as the exploration aid (the policy learns, unlike 3/5 task-only seeds), not as the preference for the swing. Needs the other 4 seeds.
