# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""CoherenceCost — make the lab's coherence-R the planning cost swm optimizes.

swm's solvers (CEMSolver / MPPISolver / ICEMSolver / ...) optimize a ``Costable``:
an object exposing ``get_cost(info_dict, action_candidates) -> cost-per-candidate``
(and ``criterion`` as its alias). We implement that protocol with

    cost = 1 - (goal_progress × coherence_R)

so ANY swm solver searches for the config change that maximizes the lab's coherence
objective. coherence-R *is* the cost. This is the API-correct bridge: swm's planning
model is a Costable, not a ``predict()`` object — the latent rollout lives behind
``predict_fn``, and the cost is what the solver minimizes.

Honest scope: the protocol is implemented and instantiation + CEMSolver acceptance
are verified. The exact ``action_candidates`` tensor layout is solver-specific; the
full closed-loop solve against a swm ``World`` is the production (GPU) step.
"""
from __future__ import annotations

import numpy as np

from .rollout_stability import assess_rollout


def _to_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


class CoherenceCost:
    """A swm ``Costable`` scoring action candidates by ``1 - goal_progress*coherence_R``.

    ``predict_fn(z, a) -> z'`` rolls the latent forward (a swm-trained predictor or a
    local one). ``info_dict`` carries the current latent under ``'state'`` and the
    goal latent under ``'goal'``.
    """

    def __init__(self, predict_fn, horizon: int = 8):
        self.predict_fn = predict_fn
        self.horizon = horizon

    def criterion(self, info_dict, action_candidates):
        return self.get_cost(info_dict, action_candidates)

    def get_cost(self, info_dict, action_candidates):
        import torch

        z0 = _to_np(info_dict.get("state"))
        goal = _to_np(info_dict.get("goal"))
        cand = (
            action_candidates.detach().cpu().numpy()
            if hasattr(action_candidates, "detach")
            else np.asarray(action_candidates)
        )
        n = cand.shape[0] if cand.ndim >= 1 else 1
        d0 = float(np.linalg.norm(z0 - goal)) if (goal is not None) else None

        costs = []
        for i in range(n):
            a = np.asarray(cand[i], dtype=np.float64).ravel()
            z = z0.astype(np.float64)
            traj = [z]
            for _ in range(self.horizon):
                z = np.asarray(self.predict_fn(z, a), dtype=np.float64)
                traj.append(z)
            r = assess_rollout(traj).coherence_R
            if d0 and d0 > 1e-9:
                progress = (d0 - float(np.linalg.norm(z - goal))) / d0
                score = max(0.0, progress) * r
            else:
                score = r
            costs.append(1.0 - score)
        return torch.tensor(costs, dtype=torch.float32)
