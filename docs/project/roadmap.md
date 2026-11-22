# Roadmap

Order follows `AGENTS.md`; dates/owners are the team's to fill in.

| # | Milestone | Owner | Status |
|---|---|---|---|
| 1 | MuJoCo scene, contact tuning, scripted baselines | CL (agent-assisted) | scene, env, grasp and **rigid flip baseline done** on the Panda (2026-11-20: stock hand 98%, Robotiq 2F-85 100% over 200 seeds). **Pivot baseline: not possible with the Panda hands tested (stock, Robotiq), works with the WidowX 250 S and primitive pads** (width-controlled gravity pivot + placement: 1299/1300 on random scenes with mass 0.15-0.25 kg and COM within a quarter of the cylinder length, 2026-11-21; pivot feasibility note, Parts 5-14). **Open for the team: arm / gripper choice** (WidowX with custom pads vs Panda) |
| 2 | State-based PPO / SAC, no-efficiency ablation | CL (agent-assisted) | metrics + reward proposal (`metrics.md`), scripted baselines on the metrics (`experiments/2026-11-21-scripted-baselines-metrics.md`), `WidowXFlipEnv` and PPO training scripts done; **PPO runs done (5 seeds per condition, 6M steps):** task + efficiency 96.7 % success, wrist rotation 0.09 rad, rho 0.90 (goals met); task only 40 % (2/5 seeds learn, one of them with a wrist-roll-like solution), wrist rotation 3.0 rad; see `experiments/2026-11-21-ppo-task-vs-efficiency.md`. SAC, term-by-term ablation, more seeds: TODO. **Learning approach + grasp too** (`learn_grasp` mode): not working yet, best policy reaches the grasp pose but does not close on the cup, see `experiments/2026-11-22-learned-approach-grasp.md`; the committed policies start from the pickup state (cup already held) |
| 3 | Evaluation across baselines/ablations | | |
| 4 | Vision policy (stretch) | | |
| 5 | Final deliverables | | |
