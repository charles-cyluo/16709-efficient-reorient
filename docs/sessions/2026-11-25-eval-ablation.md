# 2026-11-25: evaluation and reward ablation (MJ, agent-assisted)

**Goal.** Milestone 3: finish the evaluation and ablation promised in the proposal slides (baselines: scripted pivot, scripted rigid flip, RL without efficiency reward; metrics: success, drop, time, path length, wrist rotation, rho; ablation: with vs without the efficiency terms).

**Scope decisions (MJ, 2026-11-25).**
- PPO only: SAC is not added.
- Friction and start-orientation randomization: not now (scene distribution stays as in `project/metrics.md` section 4).

**State found.** Proposal vs repo: baselines, metrics and the with/without-efficiency ablation already exist (`experiments/2026-11-21-ppo-task-vs-efficiency.md`, goals met: 96.7 % success, wrist rotation 0.09 rad, rho 0.90). The 2026-11-21 run directories (`outputs/runs/`, `outputs/eval/`, videos) are git-ignored and **not on this machine**; only the two committed seed-3 policies are (`outputs/weights/task_vs_eff/`).

**Plan.**
1. Re-check the pipeline: scripted baselines (`baselines_v3`) and the committed policies on 200 scenes.
2. Term-by-term efficiency ablation, 5 seeds each (new): `eff_time_only`, `eff_wrist_only`, `eff_travel_only` (`configs/ablation_v1.json`).
3. Re-train the main ablation (`task_only` with drop 10, `task_eff`), 5 seeds each, same settings as 2026-11-21 (for curves and a single evaluation over all conditions). Skip if the 2026-11-21 run dirs can be copied from the machine that produced them.
4. One `eval_policy.py --groups` evaluation of all conditions at 6M steps, figures (`scripts/plot_eval.py`), videos for the demo.

**Done so far.**
- Branch `mj/eval-ablation`.
- `configs/ablation_v1.json` (conditions = reward weight overrides; drop penalty 10 everywhere, so with the time term on the termination penalty still exceeds the horizon cost), `scripts/sweep_ppo.py` (queue of `train_ppo.py` runs, skips finished ones), `scripts/plot_eval.py` (bars with per-seed dots, training curves). `matplotlib` added as a dependency.
- Throughput on this machine: about 900 env steps/s with 15 envs (CPU and GPU device alike; the machine is shared with another heavy job, load about 15-25 of 32 cores), so one 6M-step run takes about 1.9 h.

**Open / TODO.** See plan; results go to `experiments/2026-11-25-reward-ablation.md`.
