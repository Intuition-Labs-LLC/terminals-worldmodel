# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Intuition Labs LLC
import numpy as np
import pytest

from terminals_worldmodel import (
    ENV_ID,
    FORMAT_NAME,
    CoherenceCost,
    assess_rollout,
    register,
    register_env,
    register_format,
)


def test_coherence_converging_beats_wandering():
    conv = assess_rollout([np.ones(3) * (0.5 ** i) for i in range(10)])
    wander = assess_rollout([np.random.default_rng(i).normal(size=3) for i in range(10)])
    assert conv.crystallized and conv.coherence_R > 0.8
    assert conv.coherence_R > wander.coherence_R


def test_env_registers_and_steps():
    import gymnasium as gym

    assert register_env() == ENV_ID
    env = gym.make(ENV_ID)
    obs, info = env.reset(seed=0)
    assert obs.shape == (64,) and "goal" in info
    obs2, reward, term, trunc, info2 = env.step(env.action_space.sample())
    assert obs2.shape == (64,)


def test_swm_world_wraps_env():
    swm = pytest.importorskip("stable_worldmodel")
    register_env()
    world = swm.World(ENV_ID, num_envs=1, add_pixels=False, goal_conditioned=False)
    assert world is not None
    try:
        world.close()
    except Exception:
        pass


def test_format_registers_and_is_listed():
    swm = pytest.importorskip("stable_worldmodel")
    assert register_format() == FORMAT_NAME
    assert FORMAT_NAME in swm.data.list_formats()


def test_coherence_cost_is_a_solver_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_worldmodel")
    from stable_worldmodel.solver import CEMSolver

    cost = CoherenceCost(lambda z, a: z + 0.05 * np.resize(a, z.shape), horizon=3)
    c = cost.get_cost({"state": np.zeros(8), "goal": np.ones(8)}, torch.randn(4, 8))
    assert tuple(c.shape) == (4,)
    CEMSolver(model=cost, num_samples=4, n_steps=1, device="cpu")


def test_register_wires_both():
    pytest.importorskip("stable_worldmodel")
    reg = register()
    assert reg["env"] == ENV_ID and reg["format"] == FORMAT_NAME
