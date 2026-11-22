# Project overview (from the 16-709 proposal, Team 1)

**Learning Efficient In-Hand Reorientation Using Gravity** (Track 2: MuJoCo + standard RL baselines).
Team: Chengyuan Luo, Muqing Jiang, Ethan Fisk.

**Question.** Humans use gravity and contacts to manipulate efficiently. If an RL policy is rewarded
for efficiency (time, arm travel, wrist rotation), does it learn to exploit gravity-driven in-hand
slip instead of re-grasping?

**Task.** Franka Panda picks up an upright cup and **inverts it (about 180°)**, then sets it down inverted at the
target pose. Reference behavior (hand-scripted video, not in repo): the gripper **pinches the cup near its
bottom end and lifts; the cup pivots about the fingertips under gravity** (upright, tilted, horizontal, hanging)
for an end-over-end flip of ~180° with **no wrist rotation**; then it is set down inverted. The direct solution
is a 180° wrist rotation, so the efficiency reward is what should make the policy prefer the slip.
Randomize cup mass, friction, start orientation.

**Algorithms.** PPO, SAC (Stable-Baselines3). Stage 1: joint state + accurate cup pose, MLP.
Stage 2 (stretch): joint state + camera image, pretrained image encoder + MLP.

**Reward.** Task: success (target orientation + placement), failure (drop). Efficiency penalties:
completion time, arm-link travel after pickup, cumulative wrist rotation.

**Success.** Inverted within 15° tilt, placement error < 3 cm, no drop.
**Metrics.** Success rate, drop rate; completion time, path length, wrist rotation; gravity-assist
ratio rho = in-hand cup rotation / total cup rotation.
**Baselines.** Scripted pivot, scripted rigid flip, RL without efficiency reward (main ablation).
**Goals.** >= 80% success; efficiency reward lowers wrist rotation; rho >= 0.7. Stretch: vision >= 60%.
**Resources.** Panda (gripper may be replaced after testing finger depth/controlled slip), custom
rigid cup + table, simulated camera.

## Decided
- Target pose: cup upside-down, near its start position on the table (2026-11-20, user).
- Gripper: start with the stock Panda hand (user, 2026-11-20); revisit if finger depth blocks controlled slip
  (adaptive/Robotiq 2F-85 is the candidate).

## Open (not yet decided by the team)
- Final gripper choice (see above). Stock hand: cup hits the palm when pivoting. Robotiq 2F-85: geometry OK, but the cup sticks by friction (narrow hold-vs-slip window). See experiments/2026-11-20-pivot-feasibility-probe.md.
- Start-orientation randomization (what range, besides upright).
