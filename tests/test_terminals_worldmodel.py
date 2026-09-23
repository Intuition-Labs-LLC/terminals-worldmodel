# SPDX-License-Identifier: AGPL-3.0-or-later
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


def test_nucleation_law_locks_above_critical_coupling_and_decays_below():
    import math

    scale = 0.1
    steps = 31
    # d_tail below scale: K = scale / d_tail >= 1, the locked atom branch
    locked = assess_rollout([np.full(3, i * 0.02 / math.sqrt(3)) for i in range(steps)], scale)
    K = scale / (locked.tail_displacement + 1e-9)
    assert K >= 1.0
    assert locked.coherence_R == pytest.approx(math.cos(math.asin(1.0 / K) / 2.0), abs=1e-9)
    # d_tail above scale: K < 1, the sub-critical branch
    drifting = assess_rollout([np.full(3, i * 0.5 / math.sqrt(3)) for i in range(steps)], scale)
    assert scale / drifting.tail_displacement < 1.0
    assert drifting.coherence_R == pytest.approx(math.exp(-drifting.tail_displacement / scale), abs=1e-9)
    assert locked.coherence_R > drifting.coherence_R
    # the fixed point: no displacement at all reads R = 1
    still = assess_rollout([np.ones(3) for _ in range(steps)], scale)
    assert still.coherence_R == pytest.approx(1.0, abs=1e-9)
