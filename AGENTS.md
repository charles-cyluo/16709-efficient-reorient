# AGENTS.md

Guidance for coding agents (Claude Code, Codex, Cursor, etc.) and humans working in this repo.
`CLAUDE.md` points here, so Claude Code loads it automatically.

> **Hygiene:** never write machine host names (of laptops, workstations, clusters, remote
> servers) anywhere in this repo: code, configs, docs, session notes, experiment notes, commit
> messages. Describe hardware generically (e.g. "1x consumer GPU, 16 GB"). Also avoid absolute
> home paths and usernames; use repo-relative paths.

## What this project is about

Course project for 16-709 (Robotics Foundations: Software Engineering), Team 1: **Learning
Efficient In-Hand Reorientation Using Gravity** (Track 2: MuJoCo + standard RL baselines).
Team: Chengyuan Luo, Muqing Jiang, Ethan Fisk.

Humans use gravity and contacts to manipulate efficiently (e.g. letting an object slip and
swing in the fingers instead of re-grasping). Question: if an RL policy is rewarded for
**efficiency** (short time, little arm travel, little wrist rotation), will it learn to exploit
gravity-driven in-hand slip to reorient an object, trading arm motion for controlled slip?

- **Task.** A Franka Panda arm in MuJoCo picks up an upright cup from a table and **inverts
  it (flips it upside-down, about 180°)**, then places it at the target pose. The desired
  behavior (seen in a hand-scripted reference video, not kept in the repo): the gripper
  pinches the cup near its bottom end and lifts, and the cup pivots about the fingertips under
  gravity (upright, tilted, horizontal, then hanging) for an end-over-end flip of about 180°
  with **no wrist rotation**; it is then set down inverted. A direct solution would rotate
  the wrist by 180°, so the efficiency reward is what should make the policy prefer the
  slip. "Target orientation" in the reward and success terms means the inverted cup.
  Randomize cup mass, friction and start orientation.
- **Algorithms.** PPO and SAC via Stable-Baselines3 (PyTorch).
- **Stage 1 (state).** Observation = robot joint state + accurate cup pose, MLP policy.
- **Stage 2 (vision, stretch).** Observation = joint state + camera image, pretrained image
  encoder, then MLP policy.
- **Reward.** Task terms: success (target orientation and placement), failure (cup dropped).
  Efficiency penalties: completion time, arm-link travel after pickup, cumulative wrist
  rotation.
- **Success.** cup inverted with tilt < 15° from the target orientation, placement error < 3 cm, no drop.
- **Metrics.** Task: success rate, drop rate. Efficiency: completion time, path length, wrist
  rotation. Gravity-assist ratio ρ = in-hand cup rotation / total cup rotation (the total is
  about 180° for the inversion).
- **Baselines.** Scripted pivot, scripted rigid flip, RL without efficiency reward (this is
  also the main ablation).
- **Goals.** ≥ 80% success; efficiency reward lowers wrist rotation; ρ ≥ 0.7. Stretch: vision
  policy ≥ 60% success.
- **Deliverables.** Live demo (policy with vs. without efficiency reward), short report,
  3-5 min video, clean documented code repo with setup scripts.

This summary may go out of date. **`docs/project/` is the source of truth** for goals and
design (it should carry the proposal content, since the slides are not kept in the repo). Read it before making design decisions.

## Roadmap

The proposal's week-by-week schedule is **not** binding; ignore it. Work through these
milestones in order, and keep dates, owners and status in `docs/project/roadmap.md`.

1. MuJoCo scene (Panda, table, cup), contact tuning, scripted baselines (pivot, rigid flip), both of which must invert the cup.
2. State-based PPO / SAC with efficiency rewards, plus the no-efficiency-reward ablation.
3. Evaluation across baselines and ablations (success, efficiency metrics, ρ).
4. Vision policy (stretch).
5. Final deliverables: live demo, report, video, clean repo.

## Setup (uv)

Python environments are managed with [`uv`](https://docs.astral.sh/uv/). Do not use bare
`pip install` or conda.

```bash
uv sync                      # create .venv from pyproject.toml + uv.lock
uv run python -m <module>    # or: uv run python scripts/<entry>.py
uv add <package>             # add a dependency (commit pyproject.toml and uv.lock)
```

- Commit `pyproject.toml` and `uv.lock`; never commit `.venv/`.
- Pin the Python version in `pyproject.toml` / `.python-version`.
- Put optional heavy deps (e.g. the vision stage) in an optional group
  (`uv sync --extra vision`).

## Documentation map

| Folder | What goes there | File name |
|---|---|---|
| `docs/project/` | Living project-level docs maintained by the team: overview, roadmap, methods, reward design, baselines/ablations, open questions. **Read before design work, and don't overwrite the team's decisions.** If code or results conflict with these docs, flag it rather than silently editing. | `<topic>.md` |
| `docs/sessions/` | **Every working session** (human or agent) leaves a note: goal, what was done, decisions and why, problems hit, open TODOs, links to experiments/implementation notes | `YYYY-MM-DD-<topic>.md` |
| `docs/experiments/` | **Every experiment** (training run, sweep, evaluation, scripted baseline): hypothesis, exact commands, config, git rev, output dirs, results, plots/videos, conclusion | `YYYY-MM-DD-<experiment>.md` |
| `docs/implementations/` | Structure and implementation details: repo layout, env/observation/action spec, reward terms, interfaces, deviations from references, known caveats. Update in place when the code changes. | `<component>.md` |

Rules:

- Write the session note **during** the session, not just at the end. If a session spans
  several days, append to the same file under dated headings. Put your name/initials in the
  note so teammates know who to ask.
- Experiments must be reproducible from their note: exact command, config, seed, git revision,
  and where the outputs/weights are.
- Report negative and partial results honestly.
- Link between docs with relative links instead of duplicating content.
- When you add, move, rename or remove a module or top-level folder, update
  [`docs/implementations/repository-layout.md`](docs/implementations/repository-layout.md)
  in the same change.

## Repository layout

See [`docs/implementations/repository-layout.md`](docs/implementations/repository-layout.md).
Intended structure (create as needed, keep it simple):

- `src/<package>/` — importable code (package name: short and generic, e.g. `reorient`).
  - `envs/` — MuJoCo environment (Gymnasium-style): scene loading, observations, actions,
    reward terms, termination, randomization.
  - `assets/` — MJCF/XML for the cup, table, Panda and gripper, plus meshes.
  - `policies/` — scripted baselines (pivot, rigid flip) and learned-policy wrappers; the image
    encoder for the vision stage.
  - `eval/` — metrics (success, drop, time, path length, wrist rotation, ρ) and rollout/video
    utilities.
- `scripts/` — thin entry points (train, evaluate, render demo). Logic lives in `src/`.
- `configs/` — experiment configs (env, reward weights, algorithm hyperparameters).
- `tests/` — light tests (env steps, reward terms, metric definitions). Don't over-invest.
- `outputs/` — generated artifacts, mostly **git-ignored**, except `outputs/weights/`
  (see below). See [`docs/implementations/outputs.md`](docs/implementations/outputs.md).
- `docs/` — see the documentation map above.

## Experiments and outputs

- Configure experiments through config files or CLI flags, not by editing constants.
- Each run writes to `outputs/runs/<YYYY-MM-DD>-<name>-<seed>/` with its full resolved config,
  git revision, logs/TensorBoard curves and eval results. Its `docs/experiments/` note cites
  that path.
- Fix and record seeds. Evaluate over N independent trials with randomized mass, friction and
  start orientation, and report mean ± spread, not a single rollout.
- Always compare against the baselines and the no-efficiency-reward ablation using the same
  evaluation script and metrics.
- **Lightweight weights may be committed.** Small final or milestone policy checkpoints
  (roughly < 20 MB each, e.g. SB3 `.zip` MLP policies) go in
  `outputs/weights/<experiment>/` with a short `README`/note linking the experiment doc and
  git revision. Everything else stays git-ignored: datasets, replay buffers, intermediate
  checkpoints, videos, large vision models (share these out-of-band and link them in the
  experiment note). Don't commit more than a handful of checkpoints; if size becomes a
  problem, switch to Git LFS and document it.

## Collaboration

- The git upstream is **not configured yet**. Don't invent a remote URL or push anywhere;
  work locally and commit in small steps. Once the team sets a remote, record it in
  `docs/implementations/repository-layout.md`.
- Work on short-lived branches (`<initials>/<topic>`), merge to `main` via PR or quick review
  by a teammate. Don't force-push `main`.
- Before starting, skim recent `docs/sessions/` and `docs/experiments/` so you don't repeat
  work or collide with a teammate. Claim a task by noting it in `docs/project/roadmap.md`.
- Keep the env interface (observation/action spaces, reward term names, metric names) stable
  and documented, since three people build on it. Announce changes in a session note.
- Commit messages: short, imperative, no host names.

## Working principles

- Ask before decisions that belong to the team: gripper choice, reward term weights and
  definitions, success thresholds, track/scope changes, anything marked open in
  `docs/project/`.
- Get the simplest end-to-end loop working first (scene → scripted baseline → PPO on state),
  then iterate. Scope is a course project: prefer clear and working over general.
- Never use a local checkout outside this repo as a hidden dependency. If external code or
  assets are needed (e.g. the Panda MJCF), vendor or pin them inside this repo and document
  the source in `docs/implementations/`.
- Don't commit large artifacts (datasets, videos, big checkpoints) to git.
