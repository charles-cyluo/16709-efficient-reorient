"""Figures for the final evaluation: per-method bars with per-seed dots, and training curves per condition.

    uv run python scripts/plot_eval.py --eval outputs/eval/ablation_v1.json --conditions task_only task_eff eff_time_only eff_wrist_only eff_travel_only

--eval is a file written by `scripts/eval_policy.py --groups` (run names <condition>/<seed>); its scripted-baseline aggregates are drawn as reference bars.
Training curves are read from outputs/runs/*-<condition>-<seed>/progress.csv (all seeds found). Figures go to outputs/figures/<eval tag>/ (git-ignored).
"""
import argparse
import csv
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (dataviz skill, light mode): categorical slots in fixed order, text and grid tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SCRIPTED = "#a9a8a3"
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

LABELS = {"task_only": "RL task only", "task_eff": "RL task + all eff.", "eff_time_only": "RL + time", "eff_wrist_only": "RL + wrist rot.",
          "eff_travel_only": "RL + joint travel", "scripted pivot": "scripted pivot", "scripted roll": "scripted roll"}
METRICS = [("success", "success rate [%]", 100.0), ("wrist_rotation_rad", "wrist rotation [rad]", 1.0), ("joint_path_rad", "joint path [rad]", 1.0),
           ("completion_time_s", "completion time [s]", 1.0), ("rho", "gravity-assist ratio ρ", 1.0)]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def label_bar(ax, i, top, key, shown=None):
    """Value label above a bar (near-zero bars are otherwise invisible)."""
    v = top if shown is None else shown
    if np.isnan(v):
        return
    ax.annotate(("%.0f" if key == "success" else "%.2f") % v, (i, top), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7, color=TEXT2)


def seed_value(rows, key):
    """Per-seed value: success rate over all episodes, efficiency means over the successful ones (as in aggregate.py)."""
    if key == "success":
        return float(np.mean([bool(r["success"]) for r in rows]))
    v = [r[key] for r in rows if r.get("success") and r.get(key) is not None]
    return float(np.mean(v)) if v else float("nan")


def bars(data, conditions, out):
    agg, rows = data["aggregate"], data["rows"]
    per_cond = {c: [rs for n, rs in rows.items() if n.split("/")[0] == c] for c in conditions}
    methods = [m for m in ("scripted pivot", "scripted roll") if m in agg] + [c for c in conditions if per_cond[c]]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.1 * len(METRICS), 3.6), facecolor=SURFACE)
    x = np.arange(len(methods))
    for ax, (key, label, scale) in zip(axes, METRICS):
        style(ax)
        for i, m in enumerate(methods):
            if m.startswith("scripted"):
                a = agg[m]
                val = a["success_rate"] if key == "success" else a[key + "_mean"]
                ax.bar(i, scale * val, width=0.62, color=SCRIPTED, edgecolor=SURFACE, linewidth=2)
                label_bar(ax, i, scale * val, key)
                continue
            seeds = [scale * seed_value(rs, key) for rs in per_cond[m]]
            pooled = [r for rs in per_cond[m] for r in rs]
            val = scale * seed_value(pooled, key)  # pooled over the episodes of all seeds, as in the eval table
            ax.bar(i, val, width=0.62, color=SERIES[0], alpha=0.85, edgecolor=SURFACE, linewidth=2)
            jitter = np.linspace(-0.16, 0.16, len(seeds)) if len(seeds) > 1 else [0.0]
            ax.scatter(i + np.asarray(jitter), seeds, s=22, color=TEXT, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            label_bar(ax, i, max([val] + [v for v in seeds if not np.isnan(v)]), key, shown=val)
        ax.set_title(label, fontsize=9, color=TEXT, loc="left")
        ax.set_xticks(x, [LABELS.get(m, m) for m in methods], rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        if key == "rho":
            ax.set_ylim(0, 1.12)
        if key == "success":
            ax.set_ylim(0, 112)
    fig.text(0.01, 0.01, "bars: pooled over seeds (RL) or 200 scenes (scripted, grey); dots: one per seed; efficiency metrics over successful episodes only",
             fontsize=7, color=TEXT2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, "metrics_bars.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def read_progress(path, key):
    steps, vals = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get(key) not in (None, "") and r.get("time/total_timesteps"):
                steps.append(float(r["time/total_timesteps"]))
                vals.append(float(r[key]))
    return np.array(steps), np.array(vals)


def curves(conditions, out, max_steps):
    keys = [("task/success_rate", "training success rate [%]", 100.0), ("eff/wrist_rotation_rad", "wrist rotation, successful episodes [rad]", 1.0)]
    fig, axes = plt.subplots(1, len(keys), figsize=(6.0 * len(keys), 3.6), facecolor=SURFACE)
    grid = np.linspace(0, max_steps, 121)
    for ax, (key, label, scale) in zip(axes, keys):
        style(ax)
        for ci, c in enumerate(conditions):
            paths = sorted(glob.glob("outputs/runs/*-%s-*/progress.csv" % c))
            paths = [p for p in paths if re.search(r"-%s-\d+/progress.csv$" % re.escape(c), p)]
            ys = []
            for p in paths:
                s, v = read_progress(p, key)
                if len(s) > 1:
                    ys.append(np.interp(grid, s, scale * v, left=np.nan, right=np.nan))
            if not ys:
                continue
            ys = np.array(ys)
            color = SERIES[ci % len(SERIES)]
            for y in ys:  # one thin line per seed (min-max bands of several conditions overlap into an unreadable area)
                ax.plot(grid / 1e6, y, color=color, linewidth=0.8, alpha=0.35)
            ax.plot(grid / 1e6, np.nanmean(ys, 0), color=color, linewidth=2, label="%s (%d seeds)" % (LABELS.get(c, c), len(ys)))
        ax.set_title(label, fontsize=9, color=TEXT, loc="left")
        ax.set_xlabel("environment steps [M]", fontsize=8, color=TEXT2)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=7, frameon=False, labelcolor=TEXT2)
    fig.text(0.01, 0.01, "thick line: mean over seeds; thin lines: single seeds; rolling window of the last 300 training episodes (stochastic policy)", fontsize=7, color=TEXT2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out, "training_curves.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--max-steps", type=float, default=6e6)
    a = ap.parse_args()
    tag = os.path.splitext(os.path.basename(a.eval))[0]
    out = os.path.join("outputs/figures", tag)
    os.makedirs(out, exist_ok=True)
    data = json.load(open(a.eval))
    bars(data, a.conditions, out)
    curves(a.conditions, out, a.max_steps)
    print("figures in", out)


if __name__ == "__main__":
    main()
