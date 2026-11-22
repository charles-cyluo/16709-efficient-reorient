# Scripted rigid flip: grasp, invert the cup, place it upside-down (milestone 1)

**Hypothesis.** A scripted side grasp + 180 deg wrist roll can invert the randomized cup and put it down
upside-down near its start position with the stock Panda hand (this is the "scripted rigid flip" baseline; the
grasp+lift is its first half).

**Setup.** `PandaCupEnv` defaults (see [implementation note](../implementations/env-and-scripted-baselines.md)):
hollow 9 cm cup, mass U(0.10, 0.30) kg, sliding friction U(0.6, 1.2), xy jitter +-2 cm, upright start; table top
0.72 m; dt 0.5 ms, 50 Hz control. Policy `RigidFlip` with default `FlipConfig`. Success: cup axis within 15 deg
of straight down, within 3 cm of start xy, resting on the table.
Git rev: none yet (uncommitted working tree at time of run; first commit of this repo).

**Command.**
```
uv run python scripts/run_rigid_flip.py --trials 200 --seed 0 --video outputs/rigid_flip.mp4
```
(seeds 0-199 are the trial seeds; video is of trial 0, git-ignored.)

**Results (seeds 0-199).**
| metric | value |
|---|---|
| grasp + lift (cup held 7.5 cm off the table) | 200/200 |
| full flip success | 196/200 (98%) |
| wrist rotation | 180 deg by construction |
| gravity-assist ratio rho | 0 by construction (no in-hand slip) |

Failures (seeds 44, 53, 126, 162): after release the cup rolls or tips off its rim (3 ended lying on the table
5-25 cm away, 1 tipped in place). Not yet diagnosed beyond that.

**What it took (negative/partial results worth keeping).**
1. Table at 0.72 m and robot on the floor: at table level a horizontal hand is unreachable (joint 6 limit).
   The table's near edge also had to move back (wrist links jammed on it).
2. Rolling the wrist via Cartesian IK: the cup lagged the hand by ~30 deg and wedged the fingers open; friction
   changes, zero gravity, grip force, solver `noslip`, taper and palm collisions made no difference. Rolling
   with joint 7 alone fixed it (cup rotates ~179 deg).
3. Time step: full-flip success was 75% at 2 ms (6/8 and 9/12 in small early batches), 91% at 1 ms (100 seeds)
   and 98% at 0.5 ms (100 seeds, then 196/200). The failures at the larger steps were contact chatter / blow-ups
   on heavy cups.
4. 4 mm cup walls exploded when pinched; 8 mm are stable.
5. Lowering by TCP released the cup ~1 cm in the air; lowering by measured cup pose fixed it. Retracting
   toward the pre-grasp point (instead of straight back) swept a finger into the cup.

**Robotiq 2F-85 (same policy, same 200 seeds).** `uv run python scripts/run_rigid_flip.py --gripper robotiq --trials 200`:
grasp+lift 200/200, **flip 200/200 (100%)**; the stock-hand rerun after the refactor gave 58/60 on seeds 0-59 (same two
failures, seeds 44 and 53, as before). Needed only the table-edge move and `rot_tol` 0.10 (see implementation note).

**Conclusion.** Milestone-1 first baseline works (98% flip, 100% grasp) with the stock hand. Open: pivot
baseline (needs controlled in-hand slip; finger depth is the real question for the stock hand), ~2% release
failures, start-orientation randomization, hollow-cup/gripper choice review by the team.
