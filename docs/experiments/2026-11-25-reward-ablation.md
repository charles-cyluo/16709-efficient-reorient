# Reward ablation: which efficiency term makes PPO choose the gravity swing? (in progress)

**Date:** started 2026-11-25 (MJ, agent-assisted). **Status: term ablation done (3 conditions x 5 seeds, evaluated 2026-11-26); main ablation (task only / task + all, 5 seeds each) not re-trained on this machine yet.**
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

### Sweep interruption and relaunch (2026-11-26)
The first launch (from a desktop agent session, 2026-11-25 20:04) finished `eff_time_only`, `eff_wrist_only` and `eff_travel_only` seed 0. At 22:34, 3 min into `eff_time_only` seed 1, a memory-pressure kill on the workstation took down the desktop app the sweep was started from, and the sweep with it. Training did not cause it: the sweep used about 5.6 GB at the time. The partial run directory was deleted and its log kept as `outputs/logs/eff_time_only-1.killed-2026-11-25.log`.
Relaunched 2026-11-26 13:12 as its own low-priority user service, so that it no longer depends on the session that starts it. Same sweep command, finished runs are skipped:
`systemd-run --user --unit=ablation-v1-sweep --working-directory=<repo> -p Nice=15 -p IOSchedulingClass=idle -p CPUWeight=20 bash -c 'exec uv run python scripts/sweep_ppo.py ... >> outputs/logs/sweep_ablation_v1_terms.log 2>&1'`
(status: `systemctl --user status ablation-v1-sweep`). 12 runs remain, about 1 h each.

### Partial 2 (2026-11-26): `eff_wrist_only` and `eff_travel_only` seed 0
Training statistics at 6M steps, from the last rolling window of `progress.csv`, not the 200-scene evaluation:

| condition, seed 0 | success | drop | episode length [steps] |
|---|---|---|---|
| `eff_time_only` | 91 % | 4 % | 88 |
| `eff_wrist_only` | 90 % | 6 % | 83 |
| `eff_travel_only` | **0 %** | 4 % | 475 (limit 500) |

`eff_travel_only` was at 0 % success at 1M, 2M, 3M and 6M steps. By the same steps `eff_time_only` was at 1 / 48 / 73 % and `eff_wrist_only` at 23 / 69 / 72 %.

**Why travel-only fails (not a code bug).** The term matches the config: `arm_travel = -0.5 * sum |dq|` over all 6 arm joints, with time and wrist weights at 0. Rolling out the final policy, deterministically for 4 episodes:
`uv run python scripts/probe_episode_terms.py --run outputs/runs/2026-11-25-eff_travel_only-0 --weights '{"time": 0.0, "wrist_rotation": 0.0}'`
- Within the first 100 steps (2 s) the policy inverts the cylinder and brings it onto or just above the table (tilt 1-13 deg, lowest point within 1 cm of the table; inversion reward +2.8 to +2.95 of 3.0).
- It then **keeps squeezing it for the rest of the episode**: the grip action stays at +1, both pads stay in contact, and the cylinder never comes to rest. The success clauses "robot not touching" and "at rest" are therefore never met. In 2 of the 4 episodes the placement error (5.7 cm and 9.7 cm) is also above the 3 cm tolerance.
- Summed terms per episode: inversion +2.9, height +1.8, release +0.5, retreat +0.7, arm travel -1.4 (joint path 2.6-2.9 rad).

Reading: without a time penalty, holding the inverted cylinder costs nothing per step, while releasing and retreating costs joint travel and risks the -10 drop penalty. The +20 success bonus is only paid after release, retreat and settling, and PPO does not find it from this hold. `eff_wrist_only` also has no time penalty but learns, because it penalises only the wrist roll, so retreating with the other joints is free. This supports the 2026-11-21 note's guess that the time term mainly acts as an exploration aid; here its absence leaves the episode-ending step unexplored.

Side observation on the shaping: the `release` and `retreat` potentials are gated only on "placed" (tilt < 15 deg, within 1 cm of the table). While still squeezing a placed cylinder the policy already collects about 0.5 of `release` (fingers at contact width count as partly open) and about 0.7 of `retreat` (the TCP sits about 14 cm from the lower-end centre, near the top of the inverted cylinder). Little shaped reward is left for actually letting go. This may matter for any condition without a time penalty (also `task_only`, 3/5 seeds failed on 2026-11-21); the failure modes of those seeds were not checked here. One seed only; the remaining 4 travel-only seeds are queued.

### Full result (2026-11-26): 3 conditions x 5 seeds, 200 scenes per policy
All 15 runs finished 2026-11-26 18:24 (after the relaunch about 25 min per run, the machine was less loaded). Final policies (6.0M steps), deterministic, 200 random scenes each:
`uv run python scripts/eval_policy.py --groups --runs <15 run dirs> outputs/runs/committed-task_only-3 outputs/runs/committed-task_eff-3 --names eff_time_only/s0 ... eff_travel_only/s4 task_only/s3 task_eff/s3 --n 200 --n-envs 25 --baselines baselines_v3 --tag ablation_v1_terms`
(`outputs/eval/ablation_v1_terms.{json,md}`; figures `uv run python scripts/plot_eval.py --eval outputs/eval/ablation_v1_terms.json --conditions task_only task_eff eff_time_only eff_wrist_only eff_travel_only` -> `outputs/figures/ablation_v1_terms/`).
`task_only` / `task_eff` here are only the committed seed-3 policies, for reference; the 5-seed numbers for them are in the 2026-11-21 note.

Per seed (success; means over the successful episodes):
| run | success | drop | time [s] | wrist rot. [rad] | joint path [rad] | link travel [m] | rho | strategy |
|---|---|---|---|---|---|---|---|---|
| time only, s0 | 100 % | 0 % | 1.06 | 3.21 | 11.31 | 1.34 | 0.16 | rigid flip |
| time only, s1 | 100 % | 0 % | 0.82 | 0.49 | 5.53 | 1.96 | 0.76 | swing |
| time only, s2 | 99.5 % | 0.5 % | 1.14 | 0.07 | 4.17 | 0.94 | 0.89 | swing |
| time only, s3 | 0 % | 0 % | - | - | - | - | - | does not learn |
| time only, s4 | 100 % | 0 % | 1.28 | 3.17 | 9.32 | 0.90 | 0.13 | rigid flip |
| wrist only, s0 | 100 % | 0 % | 0.81 | 0.13 | 7.09 | 1.52 | 0.86 | swing |
| wrist only, s1 | 2 % | 0 % | 8.32 | 0.65 | 7.44 | 1.76 | 0.87 | (rarely succeeds) |
| wrist only, s2-s4 | 0 % | 0 % | - | - | - | - | - | do not learn |
| travel only, s0 | 0 % | 1.5 % | - | - | - | - | - | does not learn (holds, Partial 2) |
| travel only, s1 | 46.5 % | 17.5 % | 1.87 | 0.34 | 2.85 | 0.87 | 0.93 | swing |
| travel only, s2 | 91.5 % | 3 % | 1.16 | 0.41 | 2.35 | 0.80 | 0.93 | swing |
| travel only, s3 | 0 % | 2 % | - | - | - | - | - | does not learn |
| travel only, s4 | 100 % | 0 % | 0.61 | 0.07 | 1.34 | 0.55 | 0.89 | swing |

Pooled over 5 seeds: time only 79.9 % (95 % CI 77.3-82.3), wrist rotation 1.74 +- 1.46 rad, rho 0.48 +- 0.34; wrist only 20.4 % (18.0-23.0), wrist 0.14 rad, rho 0.86; travel only 47.6 % (44.5-50.7), wrist 0.26 rad, joint path 2.02 rad, rho 0.91.
For comparison, 2026-11-21 (5 seeds, matched drop 10): task only 40.0 %, wrist 3.00 rad, rho 0.45; task + all three terms 96.7 %, wrist 0.09 rad, rho 0.90.

Seeds that learn (>= 90 % success) / of those, seeds that use the swing (rho > 0.7):
| condition | learn | swing among learners |
|---|---|---|
| task only (2026-11-21, drop 10) | 2/5 | 1/2 |
| time only | 4/5 | 2/4 |
| wrist only | 1/5 | 1/1 |
| travel only | 2/5 (+1 at 46.5 %) | 2/2 (3/3) |
| task + all three (2026-11-21) | 5/5 | 5/5 |

**Reading.**
- **Time penalty: exploration aid, not a preference.** It gives the most reliable single term (4/5 learn), but the learners split 2/2 between rigid flip and swing, the same mix as task only. It makes the policy finish, it does not choose how.
- **Wrist-rotation and joint-travel penalties: they choose the swing.** Every seed that succeeds under either term uses the swing (rho 0.86-0.93, wrist rotation <= 0.65 rad); the rigid flip never appears. Without a time term, though, most seeds stop at "invert and hold" (3/5 wrist, 2-3/5 travel), the failure analysed in Partial 2.
- **Only the combination gives both.** Task + all three terms (5/5 learn, 5/5 swing) is the only condition that is reliable *and* always chooses the swing: time supplies the pressure to finish, wrist / travel supply the preference.
- Travel only seed 4 is the most economical policy so far (0.61 s, joint path 1.34 rad, link travel 0.55 m, all lower than task + all seed 3: 0.63 s, 2.98 rad, 1.39 m).

**Caveats.** 5 seeds per condition: "learns" is a Bernoulli outcome per seed, so 1/5 vs 4/5 is suggestive, not significant (Fisher exact p about 0.21). Several curves were still rising at 6M (wrist only s1: 13 % training success at 6M; travel only s1), so the failing counts are at this budget, not final. The rho > 0.7 split is clean here (rigid flips at 0.13-0.16, swings at >= 0.76). Pooled rates mix seeds and are not independent samples. Efficiency means are over successful episodes only (wrist only s1: 4 episodes).
