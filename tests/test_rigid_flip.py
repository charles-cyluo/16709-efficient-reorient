import numpy as np

from reorient.envs import PandaCupEnv
from reorient.policies.rigid_flip import RigidFlip, flip_success


def test_env_steps_and_randomizes():
    env = PandaCupEnv()
    _, i0 = env.reset(seed=0)
    _, i1 = env.reset(seed=1)
    assert i0["cup_mass"] != i1["cup_mass"]
    obs, *_ = env.step(np.concatenate([env.arm_qpos(), [1.0]]))
    assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()


def test_rigid_flip_inverts_cup():
    env = PandaCupEnv()
    for seed in range(3):
        env.reset(seed=seed)
        pol = RigidFlip(env)
        start_xy = env.cup_pose()[0][:2].copy()
        while not pol.done:
            env.step(pol.act())
        assert pol.grasped and not pol.failed
        assert flip_success(env, start_xy)


def test_rigid_flip_inverts_cup_with_robotiq():
    env = PandaCupEnv(gripper="robotiq")
    for seed in range(2):
        env.reset(seed=seed)
        pol = RigidFlip(env)
        start_xy = env.cup_pose()[0][:2].copy()
        while not pol.done:
            env.step(pol.act())
        assert pol.grasped and not pol.failed
        assert flip_success(env, start_xy)


def test_cylinder_shape_grasp_and_lift():
    env = PandaCupEnv(cup_shape="cylinder")
    env.reset(seed=0)
    pol = RigidFlip(env)
    while pol.phase != "roll" and not pol.done:
        env.step(pol.act())
    assert pol.grasped and not pol.failed
