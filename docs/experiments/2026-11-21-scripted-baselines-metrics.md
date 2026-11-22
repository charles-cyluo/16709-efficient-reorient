# Scripted baselines on the proposed metrics (WidowX): gravity pivot vs naive wrist roll

**Date:** 2026-11-21. **Hypothesis:** both scripted controllers invert the cylinder reliably; the pivot saves wrist rotation and joint travel (high rho), the roll is the naive 180 deg solution.
**Code:** `scripts/run_width_pivot.py --controller {pivot,roll}`, `scripts/eval_baselines.py`, metrics in `src/reorient/eval/metrics.py`. Git revision: no commit yet (working tree).
**Command:** `PROBE_DT=0.0005 uv run python scripts/eval_baselines.py --n 200 --seed 0 --tag baselines_v1`  (output: `outputs/eval/baselines_v1.{json,md}`, git-ignored).
**Scenes:** 200 random (mass 0.15-0.25 kg, COM on the axis within +-1.75 cm), fixed start. Roll: grip squeeze 12 mm, lift 16 cm, `wrist_rotate` +3.14 rad in 2 s (pure joint motion), then the same
alignment / lowering / release / retract as the pivot. Pivot: the controller of the pivot feasibility note, Part 14.

| metric | gravity pivot (width control) | naive wrist roll |
|---|---|---|
| success rate (95% CI) | 100.0% (98.1-100.0) | 100.0% (98.1-100.0) |
| drop rate | 0.0% | 0.0% |
| completion time [s] | 7.62 +- 0.57 | 5.31 +- 0.01 |
| TCP path [m] | 0.36 +- 0.01 | 0.27 +- 0.00 |
| arm-link travel [m] | 1.37 +- 0.07 | 0.99 +- 0.00 |
| joint path [rad] | 3.58 +- 0.47 | 5.36 +- 0.00 |
| wrist rotation [rad] | 0.04 +- 0.04 | 3.14 +- 0.00 |
| gravity-assist ratio rho | 0.91 +- 0.06 | 0.01 +- 0.00 |

**Conclusion.** Confirmed for wrist rotation, joint travel and rho (0.91 >= the 0.7 goal). Not confirmed for time and Cartesian travel: the scripted pivot is slower (gentle grip ease-off, slow placement) and moves the TCP / links more.
So the weights of the efficiency reward decide which solution wins (see `docs/project/metrics.md` section 6).
**Caveats.** Same set-up caveats as the pivot note (WidowX with custom pads, fixed start, pad friction 1.5). The scripted controllers are not tuned for speed; a faster pivot is likely possible.
