"""Build the MuJoCo model for a given gripper.

- ``"panda"``: stock Franka hand, loaded straight from ``assets/scene.xml``.
- ``"robotiq"``: Robotiq 2F-85 (MuJoCo Menagerie) attached to the Panda flange with MjSpec, from
  ``assets/scene_robotiq.xml`` (arm only). A ``tcp`` site is added at the pad center with the same frame
  convention as the stock hand: z = approach, y = finger-opening axis.
"""
from __future__ import annotations

import mujoco
import numpy as np

from reorient import ASSETS_DIR, SCENE_XML, TABLE_Z

GRIPPERS = ("panda", "robotiq")
ROBOTIQ_TCP_Z = 0.1415  # pad center along the approach axis, from the flange [m]


def build_model(gripper: str = "panda") -> mujoco.MjModel:
    if gripper == "panda":
        return mujoco.MjModel.from_xml_path(str(SCENE_XML))
    if gripper != "robotiq":
        raise ValueError(f"unknown gripper {gripper!r}; choose from {GRIPPERS}")
    spec = mujoco.MjSpec.from_file(str(ASSETS_DIR / "scene_robotiq.xml"))
    grip = mujoco.MjSpec.from_file(str(ASSETS_DIR / "robotiq_2f85/2f85.xml"))
    frame = spec.body("attachment").add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0])
    frame.attach_body(grip.body("base_mount"), "rq_", "")
    # The Robotiq opens along its own x axis. The tcp site is rotated +90 deg about the approach axis so
    # that the TCP frame has y = finger-opening axis (x' = gripper y, y' = -gripper x).
    c = np.cos(np.pi / 4)
    spec.body("rq_base_mount").add_site(name="tcp", pos=[0, 0, ROBOTIQ_TCP_Z], quat=[c, 0, 0, c],
                                        size=[0.004] * 3, rgba=[1, 0, 0, 1])
    return spec.compile()


WIDOWX_TCP_X = 0.0845 # centre of the box pads, along the gripper x axis from gripper_link [m]
WIDOWX_BASE_POS = (0.2, 0.0, TABLE_Z)  # arm base on the table top
# Pad box half-sizes [m] (x along the finger, y thickness, z height). Thin in z so the contact with a vertical
# cylinder is short and the object can pivot about the finger axis; long in x so the contact can slide along the pad.
WIDOWX_PAD_HALF = (0.0245, 0.002, 0.005)  # pad spans 0.060 .. 0.109 along the gripper x axis (0.109 = fingertip)
WIDOWX_PALM_BOX = ((0.029, 0.0, 0.0), (0.029, 0.05, 0.02))  # replaces the palm mesh hull: front face at x = 2 * half-x (0.058)
WIDOWX_PAD_FACE_Y = 0.009  # finger inner face, measured from the finger joint position (ray cast on the mesh)


def build_widowx_model(pad_half=WIDOWX_PAD_HALF, pad_friction=1.0, pad_sphere_radius=None, pad_x=None,
                       palm_half=WIDOWX_PALM_BOX[1]) -> mujoco.MjModel:
    """Trossen WidowX 250 S (Menagerie) mounted on the table, from ``assets/scene_widowx.xml``.

    Menagerie's finger and palm collision (mesh hulls + two tiny tip spheres per finger) is switched off and replaced
    by one primitive box pad per finger on the flat inner face and one palm box. A ``tcp`` site sits at the pad centres with the same frame
    convention as the other grippers (z = approach = gripper x, y = finger-opening axis)."""
    spec = mujoco.MjSpec.from_file(str(ASSETS_DIR / "scene_widowx.xml"))
    arm = mujoco.MjSpec.from_file(str(ASSETS_DIR / "wx250s/wx250s.xml"))
    spec.worldbody.add_frame(pos=list(WIDOWX_BASE_POS)).attach_body(arm.body("wx250s/base_link"), "", "")
    for side, sgn in (("left", -1.0), ("right", 1.0)):  # the inner face looks toward the other finger
        body = spec.body(f"wx250s/{side}_finger_link")
        for g in body.geoms:
            g.contype = g.conaffinity = 0  # drop Menagerie's collision geoms (visual meshes stay)
        px = (WIDOWX_TCP_X if pad_x is None else pad_x) - 0.066
        if pad_sphere_radius is None:
            body.add_geom(name=f"{side}_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(pad_half),
                          pos=[px, sgn * (WIDOWX_PAD_FACE_Y - pad_half[1]), 0.0],
                          friction=[pad_friction, 0.005, 0.0001], rgba=[0.9, 0.5, 0.1, 1])
        else:  # round pad: a single contact that stays on the sphere as the object pivots
            body.add_geom(name=f"{side}_pad", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[pad_sphere_radius, 0, 0],
                          pos=[px, sgn * (WIDOWX_PAD_FACE_Y - pad_sphere_radius), 0.0],
                          friction=[pad_friction, 0.005, 0.0001], rgba=[0.9, 0.5, 0.1, 1])
    palm = spec.body("wx250s/gripper_link")
    for g in palm.geoms:
        g.contype = g.conaffinity = 0  # the palm mesh hull would fill the gap between the fingers
    palm.add_geom(name="palm_box", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[palm_half[0], 0.0, 0.0],
                  size=list(palm_half), rgba=[0.3, 0.3, 0.3, 0.0])
    c = np.cos(np.pi / 4)  # rotation +90 deg about y: tcp z = gripper x, tcp x = -gripper z
    spec.body("wx250s/gripper_link").add_site(name="tcp", pos=[WIDOWX_TCP_X if pad_x is None else pad_x, 0, 0], quat=[c, 0, c, 0],
                                              size=[0.004] * 3, rgba=[1, 0, 0, 1])
    return spec.compile()
