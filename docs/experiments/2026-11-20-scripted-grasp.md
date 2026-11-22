# [SUPERSEDED] Scripted top-down grasp + lift (milestone 1)

> Superseded the same day by [the rigid flip experiment](2026-11-20-rigid-flip.md). This used a solid cylinder cup
> and a robot standing at table level; the top-down grasp is no longer reachable in the current scene (table
> raised to 0.72 m, hollow cup) and its code was removed. Kept as a record: top-down pinch worked 100/100 in
> that earlier scene.

**Hypothesis.** A simple scripted top-down pinch can reliably pick up the randomized cup.

**Setup.** `PandaCupEnv` defaults: mass U(0.10, 0.30) kg, friction U(0.6, 1.2), cup xy ±2 cm,
upright start. Policy: `ScriptedGrasp` defaults (see
old implementation note (since replaced by [env-and-scripted-baselines](../implementations/env-and-scripted-baselines.md))). Git rev: not committed at
time of run (first commit of this repo; see log after commit).

**Commands.**
```
uv run python scripts/run_scripted_grasp.py --trials 20 --video outputs/scripted_grasp.mp4
```
Plus an ad-hoc stress run (seeds 0-99, extra 2 s hold, tilt measured) — not yet a script.

**Results.**
| setting | success | max cup tilt after 3 s hold |
|---|---|---|
| default randomization, 100 seeds | 100/100 | 3.5° |
| worst corner in range (0.30 kg, mu 0.6), 20 seeds | 20/20 | 4.5° |
| out of range (0.60 kg, mu 0.3), 10 seeds | 0/10 (solver unstable) | - |

Video of trial 0: `outputs/scripted_grasp.mp4` (git-ignored, regenerate with the command above).

**Conclusion.** Grasp+lift works on the whole randomization range. Caveats: the cup is a solid
cylinder and always starts upright; both limit how much this says about later tasks. The stress
run should be turned into an eval script in milestone 3.
