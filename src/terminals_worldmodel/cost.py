# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Intuition Labs LLC
"""CoherenceCost: coherence-R as a stable-worldmodel planning cost.

swm solvers (CEMSolver / MPPISolver / ICEMSolver / ...) minimize a ``Costable``:
an object exposing ``get_cost(info_dict, action_candidates) -> cost-per-candidate``.
This implements that protocol with

    cost = 1 - (goal_progress × coherence_R)

so any swm solver, unchanged, searches for the action that moves the latent toward
the goal AND yields a settled (stable) rollout. Rollout stability is measured from
the outside by :func:`coherence.assess_rollout`, not by the model's own head.
"""
from __future__ import annotations

import numpy as np

from .coherence import assess_rollout


def _np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).ravel()


class CoherenceCost:
    """A swm ``Costable``. ``predict_fn(z, a) -> z'`` is the latent predictor (a
    swm-trained world model, or a local one); ``info_dict`` carries the current
    latent under ``'state'`` and the goal latent under ``'goal'``."""

    def __init__(self, predict_fn, horizon: int = 8):
        self.predict_fn = predict_fn
        self.horizon = horizon

    def criterion(self, info_dict, action_candidates):
        return self.get_cost(info_dict, action_candidates)

    def get_cost(self, info_dict, action_candidates):
        import torch

        z0 = _np(info_dict.get("state"))
        goal = _np(info_dict.get("goal"))
        cand = (
            action_candidates.detach().cpu().numpy()
            if hasattr(action_candidates, "detach")
            else np.asarray(action_candidates)
        )
        n = cand.shape[0] if cand.ndim >= 1 else 1
        d0 = float(np.linalg.norm(z0 - goal)) if goal is not None else None

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
