# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""Terminals OS as a first-class swm env — the env-harness isomorphism.

swm's standard envs (the DM-Control suite via dm_control, gymnasium-robotics) are
Gymnasium envs registered under the ``swm/`` namespace, which ``swm.World(env_name)``
wraps for collect / evaluate / plan. We adopt that PATTERN — not their robotics
source. The OS's own configuration state is registered as a Gymnasium env
``terminals-os/Config-v0``, so swm's whole harness operates on the OS-world exactly
as it does on Cartpole or Reacher.

Deliberately NO dm_control / MuJoCo dependency: an OS state is structured (a route
plus a few metrics), not rendered pixels, so the heavy robotics simulators are
neither needed nor vendored. This module only adopts the isomorphism: a goal-
conditioned Gymnasium env whose observation is the encoded page-state and whose
actions are config changes.
"""
from __future__ import annotations

import numpy as np

from .action import ACTION_KINDS
from .config import LATENT_DIM

ENV_ID = "terminals-os/Config-v0"

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except Exception:  # pragma: no cover - gymnasium ships with swm
    _HAS_GYM = False


if _HAS_GYM:

    class TerminalsOSEnv(gym.Env):
        """A goal-conditioned OS-config world.

        observation : the encoded page-state vector (Box, ``LATENT_DIM``)
        action      : which config change to apply (Discrete over ``ACTION_KINDS``)
        reward      : progress of the goal-coordinate (projection on a goal direction)

        The dynamics here are a transparent stand-in — the production world model is
        swm-trained on collected traces; this env is the harness that model trains
        and plans in, the same role DM-Control envs play for robotics world models.
        """

        metadata = {"render_modes": []}

        def __init__(self, dim: int = LATENT_DIM, max_steps: int = 64, seed: int = 0):
            super().__init__()
            self.dim = dim
            self.max_steps = max_steps
            self.observation_space = spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)
            self.action_space = spaces.Discrete(len(ACTION_KINDS))
            rng = np.random.default_rng(seed)
            g = rng.normal(size=dim).astype(np.float32)
            self._g = g / (np.linalg.norm(g) + 1e-9)
            self._effect = (rng.normal(scale=0.3, size=(len(ACTION_KINDS), dim))).astype(np.float32)
            self._effect[ACTION_KINDS.index("no_op")] = 0.0  # no_op leaves the world unchanged
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
            reward = float(self._state @ self._g)  # goal-coordinate progress
            terminated = bool(reward > 0.95)
            truncated = bool(self._t >= self.max_steps)
            return self._state.copy(), reward, terminated, truncated, {"goal": self._g.copy()}

else:  # pragma: no cover
    TerminalsOSEnv = None


def register_env() -> object:
    """Idempotently register the OS-config env in the Gymnasium registry.

    Uses a class entry_point so registration does not depend on an import path
    being resolvable in a subprocess. Returns the env id (truthy) if registered/
    available, else False — richer than a bool, and any truthy check still works.
    """
    if not _HAS_GYM:
        return False
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point=TerminalsOSEnv, max_episode_steps=64)
    return ENV_ID
