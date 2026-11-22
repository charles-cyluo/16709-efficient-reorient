# Weights: PPO policies on WidowXFlipEnv (task-only vs task + efficiency reward)

SB3 PPO (MLP 256x256, gSDE) + VecNormalize statistics, trained for 6M steps. Experiment note: `docs/experiments/2026-11-21-ppo-task-vs-efficiency.md`; env: `docs/implementations/widowx-flip-env.md`.
Git revision: none yet (the repository has no commit); the working tree of 2026-11-21 produced them (`config.json` of each run is in `outputs/runs/`, git-ignored).

| file | reward | seed | success on 200 random scenes (6M steps) |
|---|---|---|---|
| `task_only_seed3.zip` | task terms only (drop penalty 5) | 3 | 100 % |
| `task_eff_seed3.zip` | task + efficiency terms (drop penalty 10) | 3 | 100 % |

Load: `model = PPO.load("task_eff_seed3.zip"); venv = VecNormalize.load("task_eff_seed3_vecnormalize.pkl", DummyVecEnv([lambda: WidowXFlipEnv()])); venv.training = False; venv.norm_reward = False`;
evaluate with `scripts/eval_policy.py`, render with `scripts/render_policy.py` (both take a run directory containing `final.zip` and `vecnormalize.pkl`: copy these files into one folder under those names).
