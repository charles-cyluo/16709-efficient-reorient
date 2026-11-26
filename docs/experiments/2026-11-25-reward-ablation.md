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
(pending)
