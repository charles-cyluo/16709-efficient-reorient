# Metrics and reward proposal (WidowX setup)

**Status: proposal** (written 2026-11-21 by the agent, for the team to confirm; thresholds and reward weights are the team's to decide, see `AGENTS.md`).
It implements the metric list of `overview.md` for the set-up that currently works (WidowX 250 S, thin cylinder; see
[`experiments/2026-11-20-pivot-feasibility-probe.md`](../experiments/2026-11-20-pivot-feasibility-probe.md), Part 14). Code: `src/reorient/eval/metrics.py`
(`EpisodeMetrics`) and `src/reorient/eval/aggregate.py`; the RL environment uses the same functions, so scripted controllers and learned policies are scored identically.

## 1. When an episode counts
An episode starts at **pickup**: the cylinder is gripped near its bottom end and lifted. All efficiency metrics count from there ("after pickup"). It ends at
success, at a drop, or at the time limit (RL: 10 s; scripted controllers run until the cylinder has settled).

## 2. Task metrics
| metric | definition |
|---|---|
| **success** | the following all hold continuously for 0.5 s: the cylinder axis is within **15 deg** of pointing straight down (inverted); the centre of its lower end face is within **3 cm** of the start position; its lowest point is on the table (< 1 cm above it); it is at rest (< 1 cm/s, < 0.1 rad/s); **no robot geometry touches it** (released and clear) |
| **drop** (failure by dropping) | at any time after pickup the cylinder is airborne (lowest point > **5 mm** above the table) with no robot contact, or it is released while moving down faster than 0.2 m/s (thrown), or is lying on the table (axis more than 60 deg from the vertical, lowest point on the table; both thresholds were tightened after the first PPO policies found shortcuts: laying the cylinder on its side, and throwing it down onto the table from about 1.5 cm), or falls out of the workspace |
| tilt error, placement error | at the end of the episode: angle from inverted [deg], horizontal distance of the lower end face to the start position [m] |
| success rate, drop rate | over N random scenes, with the 95 % Wilson interval of the success rate |

## 3. Efficiency metrics (reported over the successful episodes)
| metric | definition |
|---|---|
| **completion time** [s] | time from pickup until the success condition started to hold (the 0.5 s hold is not counted) |
| **arm-link travel** [m] | sum over the five distal arm links (upper arm, upper forearm, lower forearm, wrist, gripper) of the path length of the link frame origin, sampled every 20 ms |
| TCP path [m], joint path [rad] | path length of the tool centre point; sum of absolute joint increments of the six arm joints |
| **wrist rotation** [rad] | cumulative absolute rotation of the `wrist_rotate` joint (the roll about the approach axis). `roll_joint_rotation_rad` adds the `forearm_roll` joint |
| **gravity-assist ratio rho** | rotation of the cylinder relative to the hand divided by its rotation in the world, both as net rotation angles between the pickup and the last control step at which the cylinder is still gripped (>= 2 pad contacts), clipped to [0, 1]. A rigid wrist roll gives rho ~ 0, a pure in-hand swing ~ 1 |

## 4. Evaluation protocol
Random scenes: cylinder mass uniform in 0.15-0.25 kg, centre of mass on the axis uniform within +-1/8 of the cylinder length (+-1.75 cm), fixed start position (decided
with the team; friction is not randomized yet: the pad friction dominates, see the pivot note, Part 14 for the friction sensitivity). N >= 200 scenes per method, same
scene distribution for all methods, success rate with Wilson 95 % interval, mean +- std of the efficiency metrics over successes. `scripts/eval_baselines.py` (scripted) and
`scripts/eval_policy.py` (learned) write `outputs/eval/<tag>.json` and a markdown table.

## 5. Reward (as implemented in `WidowXFlipEnv`, weights are proposals)
**Task terms** (always on): potential-based shaping toward the goal, `w (phi_t - phi_{t-1})` with phi_inversion = -(angle from inverted)/pi (w = 3), phi_position = -min(placement error, 0.25 m)/0.25 m
(w = 1), phi_height = g (1 - min(lowest point above the table, 0.2 m)/0.2 m) with the gate g = clip(1 - (angle from inverted)/(pi/2), 0, 1), i.e. it only pays for an (almost) inverted cylinder (w = 2; without the gate the first policies parked the upright cylinder on the table), phi_release = (finger opening) while the cylinder stands inverted on the table (w = 1), phi_retreat = distance of the TCP from
the cylinder (capped 0.2 m) while it stands inverted on the table (w = 1); **success bonus +20**; **drop penalty -10**.
**Efficiency terms** (only for the second policy): **time** -0.02 per control step (-1 per second), **arm travel** -0.5 per rad of summed absolute joint increments, **wrist rotation** -2 per rad of
`wrist_rotate` motion.

## 6. Scripted baselines on this metric set (200 random scenes each, `outputs/eval/baselines_v1`)
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

**Reading it.** The pivot saves the wrist rotation (0.04 vs 3.14 rad), the joint travel (3.6 vs 5.4 rad) and has rho ~ 0.9 (goal >= 0.7), but it is **slower** (7.6 vs 5.3 s, because of the
gentle ease-off of the grip and the careful placement) and its Cartesian travel (TCP path, arm-link travel) is larger. So whether an efficiency reward "prefers the slip" depends on the weights:
wrist rotation and joint travel favour the pivot, time and Cartesian link travel favour the roll. The proposed weights (wrist rotation 2/rad, joint travel 0.5/rad, time 1/s) make the roll cost about
6.3 + 2.7 + 5.3 = 14 and the scripted pivot about 0.1 + 1.8 + 7.6 = 9.5 per episode, i.e. a margin of about 4.5 points in favour of the pivot (success bonus 20 in both). Arm-link travel (Cartesian) is
reported but not used in the reward because it would favour the roll; the team should decide whether to use link travel instead of joint travel.
