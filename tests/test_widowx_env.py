import numpy as np

from reorient.envs.widowx_flip_env import WidowXFlipEnv
from reorient.policies.widowx_expert import RollExpert


def test_env_reset_step_and_zero_action_holds():
    env = WidowXFlipEnv(efficiency=True)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and 0.15 <= info["mass"] <= 0.25
    for _ in range(50):
        obs, r, term, trunc, info = env.step(np.zeros(7))
    assert not term and env.metrics.robot_contacts() >= 2  # the zero action keeps the cylinder gripped
    assert set(info["reward_terms"]) >= {"inversion", "time", "arm_travel", "wrist_rotation"}


def test_scripted_roll_succeeds_in_env_and_metrics_are_sane():
    env = WidowXFlipEnv(efficiency=False, max_time=14.0)
    for seed in range(2):
        env.reset(seed=seed)
        ex, done = RollExpert(env), False
        while not done:
            _, _, term, trunc, info = env.step(ex.act())
            done = term or trunc
        m = info["episode_metrics"]
        assert m["success"] and not m["drop"]
        assert 3.0 < m["wrist_rotation_rad"] < 3.3 and m["rho"] < 0.1  # a wrist roll: rotation about pi, no in-hand rotation
        assert m["release_speed_mps"] < 0.2 and m["tilt_error_deg"] < 15


def test_learn_grasp_env_steps():
    """learn_grasp mode: ready-pose reset, 45-dim observation, reward terms present, gripper only closes near the grasp pose."""
    import numpy as np
    from reorient.envs.widowx_flip_env import WidowXFlipEnv

    env = WidowXFlipEnv(efficiency=True, learn_grasp=True, mass_range=(0.2, 0.2), com_range=(0.0, 0.0), max_dq=0.02)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (45,)
    a = np.zeros(7)
    a[6] = 1.0  # try to close far from the cup: ignored
    for _ in range(10):
        obs, r, term, trunc, info = env.step(a)
    assert env.grip_cmd > 0.0369
    assert {"approach", "orient", "close", "grasp", "lift", "time"} <= set(info["reward_terms"])
