# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Intuition Labs LLC
"""The agent's configuration state as a stable-worldmodel environment.

stable-worldmodel's standard envs are physical (the DM-Control suite, gymnasium-
robotics). This one is not: the "world" is an agent / OS configuration state — a
structured vector, no pixels, no simulator. It registers as a Gymnasium env so
``swm.World(env_id)`` wraps it like any other.

The dynamics here are a transparent stand-in so the harness is runnable end to end;
the production dynamics is a swm world model trained on collected traces.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ENV_ID = "terminals-os/Config-v0"
ACTION_KINDS = ("co_edit", "navigate", "explain", "no_op")
STATE_DIM = 64


class TerminalsOSEnv(gym.Env):
    """Goal-conditioned config world: obs = config-state vector, action = a config
    change, reward = progress of the goal-coordinate (the projection on a goal
    direction). ``no_op`` leaves the world unchanged."""

    metadata = {"render_modes": []}

    def __init__(self, dim: int = STATE_DIM, max_steps: int = 64, seed: int = 0):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps
        self.observation_space = spaces.Box(-1.0, 1.0, (dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_KINDS))
        rng = np.random.default_rng(seed)
        g = rng.normal(size=dim).astype(np.float32)
        self._g = g / (np.linalg.norm(g) + 1e-9)
        self._effect = rng.normal(scale=0.3, size=(len(ACTION_KINDS), dim)).astype(np.float32)
        self._effect[ACTION_KINDS.index("no_op")] = 0.0
        self._state = np.zeros(dim, dtype=np.float32)
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        r = np.random.default_rng(seed if seed is not None else 0)
        self._state = (r.normal(size=self.dim) * 0.1).astype(np.float32)
        self._t = 0
        return self._state.copy(), {"goal": self._g.copy()}

    def step(self, action):
        self._state = np.clip(
            self._state + 0.2 * self._effect[int(action)], -1.0, 1.0
        ).astype(np.float32)
        self._t += 1
        reward = float(self._state @ self._g)
        return self._state.copy(), reward, bool(reward > 0.95), bool(self._t >= self.max_steps), {"goal": self._g.copy()}


def register_env() -> str:
    """Idempotently register the env in the Gymnasium registry; returns its id."""
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point=TerminalsOSEnv, max_episode_steps=64)
    return ENV_ID
