"""Generate src/reorient/assets/cup.xml: a hollow open-top cup (straight wall by default) built from
box geoms (16 wall segments + base), body origin at the cup's geometric center.

    uv run python scripts/make_cup_xml.py
"""
import numpy as np
from pathlib import Path

import sys
H, WALL, BASE, N = 0.09, float(sys.argv[1]) if len(sys.argv) > 1 else 0.008, 0.004, 16  # meters
R_MEAN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.031
TAPER = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0  # R_top - R_bottom (0 = straight wall; taper did not matter for grasp stability)
R_TOP, R_BOT = R_MEAN + TAPER / 2, R_MEAN - TAPER / 2
OUT = Path(__file__).resolve().parents[1] / "src/reorient/assets/cup.xml"

alpha = np.arctan2(R_TOP - R_BOT, H)  # wall slant from vertical
r_mid = 0.5 * (R_TOP + R_BOT) - WALL / 2  # wall center radius
half_h = H / 2 / np.cos(alpha)
half_t = (r_mid + WALL / 2) * np.tan(np.pi / N) * 1.03  # tangential half width (slight overlap)


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


lines = ['<mujoco>', '  <body name="cup" pos="0.6 0 0.765">', '    <freejoint name="cup_joint"/>']
for i in range(N):
    phi = 2 * np.pi * i / N
    q_phi = np.array([np.cos(phi / 2), 0, 0, np.sin(phi / 2)])
    # box axes: x = radial (thickness), y = tangential, z = height. Tilt about y so the wall flares outward.
    q_tilt = np.array([np.cos(-alpha / 2), 0, np.sin(-alpha / 2), 0])
    q = quat_mul(q_phi, q_tilt)
    pos = (r_mid * np.cos(phi), r_mid * np.sin(phi), 0.0)
    lines.append(f'    <geom name="wall{i}" class="cupgeom" type="box" size="{WALL/2:.5f} {half_t:.5f} {half_h:.5f}" '
                 f'pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}"/>')
# thin solid cylinder variant (like the reference video). Compiled as collidable with mass 0 (geoms with
# contype=conaffinity=0 are dropped from the collision structures, so they cannot be enabled at runtime);
# the env switches the unused shape off.
CYL_R, CYL_H = 0.0225, 0.14
lines.append(f'    <geom name="cyl" class="cupgeom" type="cylinder" size="{CYL_R} {CYL_H/2}" pos="0 0 0" '
             f'mass="0" rgba="0.15 0.3 0.9 0"/>')
lines.append(f'    <geom name="base" class="cupgeom" type="cylinder" size="{R_BOT:.5f} {BASE/2:.5f}" pos="0 0 {-H/2 + BASE/2:.5f}"/>')
lines += ['    <site name="cup_top" pos="0 0 0.045" size="0.004" rgba="0 0 0 1"/>',
          '  </body>', '</mujoco>']
OUT.write_text("\n".join(lines) + "\n")
print("wrote", OUT)
