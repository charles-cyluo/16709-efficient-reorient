"""Aggregate per-episode metric dictionaries (from `EpisodeMetrics.summary`) into the table used to compare controllers and policies."""
from __future__ import annotations

import math

import numpy as np

TASK_KEYS = ("success", "drop")
EFFICIENCY_KEYS = ("release_height_mm", "release_speed_mps", "completion_time_s", "tcp_path_m", "arm_link_travel_m", "joint_path_rad", "wrist_rotation_rad", "roll_joint_rotation_rad", "rho")
ERROR_KEYS = ("tilt_error_deg", "placement_error_m")


def wilson(k, n, z=1.96):
    """95 % Wilson interval of a success rate."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - w) / den, (c + w) / den)


def aggregate(rows):
    """rows: list of metric dicts. Efficiency numbers are means +- std over the successful episodes only (a failed episode has no meaningful time)."""
    n = len(rows)
    ok = [r for r in rows if r.get("success")]
    lo, hi = wilson(len(ok), n)
    out = {"n": n, "success_rate": len(ok) / max(n, 1), "success_ci95": (lo, hi), "drop_rate": sum(bool(r.get("drop")) for r in rows) / max(n, 1)}
    for k in ERROR_KEYS:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k + "_mean"] = float(np.mean(v)) if v else float("nan")
    for k in EFFICIENCY_KEYS:
        v = [r[k] for r in ok if r.get(k) is not None and not math.isnan(r[k])]
        out[k + "_mean"] = float(np.mean(v)) if v else float("nan")
        out[k + "_std"] = float(np.std(v)) if v else float("nan")
    return out


def format_table(named):
    """named: {name: aggregate dict} -> markdown table."""
    rows = [("success rate (95% CI)", lambda a: "%.1f%% (%.1f-%.1f)" % (100 * a["success_rate"], 100 * a["success_ci95"][0], 100 * a["success_ci95"][1])),
            ("drop rate", lambda a: "%.1f%%" % (100 * a["drop_rate"])),
            ("completion time [s]", lambda a: "%.2f +- %.2f" % (a["completion_time_s_mean"], a["completion_time_s_std"])),
            ("TCP path [m]", lambda a: "%.2f +- %.2f" % (a["tcp_path_m_mean"], a["tcp_path_m_std"])),
            ("arm-link travel [m]", lambda a: "%.2f +- %.2f" % (a["arm_link_travel_m_mean"], a["arm_link_travel_m_std"])),
            ("joint path [rad]", lambda a: "%.2f +- %.2f" % (a["joint_path_rad_mean"], a["joint_path_rad_std"])),
            ("wrist rotation [rad]", lambda a: "%.2f +- %.2f" % (a["wrist_rotation_rad_mean"], a["wrist_rotation_rad_std"])),
            ("gravity-assist ratio rho", lambda a: "%.2f +- %.2f" % (a["rho_mean"], a["rho_std"])),
            ("release height [mm]", lambda a: "%.1f +- %.1f" % (a["release_height_mm_mean"], a["release_height_mm_std"])),
            ("downward speed at release [m/s]", lambda a: "%.2f +- %.2f" % (a["release_speed_mps_mean"], a["release_speed_mps_std"]))]
    names = list(named)
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, f in rows:
        lines.append("| %s | " % label + " | ".join(f(named[n]) for n in names) + " |")
    lines.append("| episodes | " + " | ".join(str(named[n]["n"]) for n in names) + " |")
    return "\n".join(lines)
