# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""End-to-end demo of the simulate-before-commit loop — runnable on pure numpy.

It exercises the whole paradigm: observe a page, generate a realistic candidate
set, simulate each forward through a world model, and let the planner surface the
change that best moves the system toward the goal. It proves the *planner loop*
recovers the right action; iter-2's micro_eval separately proves such a model is
*learnable from data*; swm is the production learned model. The two concerns are
orthogonal, so this demo uses a transparent analytic stand-in (``DemoWorldModel``)
to keep the planner test deterministic and dependency-free.

This is exactly the data the Zero overlay renders: a ranked list of proposed
changes, each with the predicted-outcome confidence and goal-progress attached.
"""
from __future__ import annotations

import numpy as np

from .action import OSAction
from .config import LATENT_DIM
from .observation import PageObservation
from .planner import ZeroPlanner

# Index of the no_op kind in OSAction.vector()'s leading one-hot (see action.py).
_NO_OP_SLOT = 3


class DemoWorldModel:
    """Analytic stand-in world model for exercising the planner end-to-end.

    It contracts the state's goal-coordinate (its projection on ``g_dir``) toward
    the action's alignment with the "good" action — so a useful action drives the
    state to the goal and settles (high coherence-R), a ``no_op`` leaves the world
    unchanged, and an unrelated action goes nowhere useful. Deterministic. swm is
    the learned production replacement; the planner code path is identical.
    """

    def __init__(self, g_dir: np.ndarray, w_norm: np.ndarray, gamma: float = 0.4):
        self.g = np.asarray(g_dir, dtype=np.float64)
        self.w = np.asarray(w_norm, dtype=np.float64)
        self.gamma = gamma

    def predict(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        if a[_NO_OP_SLOT] > 0.5:  # no_op → the world does not change
            return z
        align = max(0.0, float((a / (np.linalg.norm(a) + 1e-9)) @ self.w))
        p = float(z @ self.g)
        return z + self.gamma * (align - p) * self.g  # contract goal-coord toward align


def _candidate_actions() -> list[OSAction]:
    """A realistic candidate set for a /bench page. The first is, by construction,
    the action that moves the system toward the goal."""
    return [
        OSAction("co_edit", "raise per-turn timeout 180s->300s", {"lever": "L1", "value": 300}),
        OSAction("co_edit", "enable best-of-N=3 retry sampling", {"lever": "L5", "n": 3}),
        OSAction("navigate", "open the bench page", {"route": "/bench"}),
        OSAction("explain", "explain the current pass-rate", {}),
        OSAction("no_op", "leave the config unchanged", {}),
    ]


def run_planning_demo(seed: int = 0, horizon: int = 12) -> dict:
    """Run the full observe → simulate → rank loop. Pure numpy, deterministic."""
    cands = _candidate_actions()
    good = cands[0]
    w = good.vector().astype(np.float64)
    w_norm = w / (np.linalg.norm(w) + 1e-9)

    rng = np.random.default_rng(seed)
    g_dir = rng.normal(size=LATENT_DIM)
    g_dir /= np.linalg.norm(g_dir)

    wm = DemoWorldModel(g_dir, w_norm)
    obs = PageObservation(route="/bench", visible_metrics={"pass_rate": 0.5})
    z0 = obs.vector().astype(np.float64)
    p0 = float(z0 @ g_dir)
    goal_vector = z0 + (1.0 - p0) * g_dir  # goal: drive the goal-coordinate to 1.0

    plan = ZeroPlanner(world_model=wm, horizon=horizon).propose(
        obs, cands, goal="raise the pass rate", goal_vector=goal_vector
    )
    return {
        "ran": True,
        "status": plan.status,
        "top_proposal": plan.proposals[0] if plan.proposals else None,
        "proposals": plan.proposals,
        "n_proposals": len(plan.proposals),
        "ranking": plan.scored,
        "good_action_summary": good.summary,
    }
