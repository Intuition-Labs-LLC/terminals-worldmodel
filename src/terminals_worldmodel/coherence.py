# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Intuition Labs LLC
"""Coherence-R: a model-independent rollout-stability measure.

A world model rolls a latent forward ``z_0 -> z_1 -> ... -> z_H``. The rollout has
settled when the latent stops moving. We score that settling with

    R = exp(-d_tail / scale)        # d_tail = mean step displacement over the final third

``R`` in [0, 1]: a rollout that converges to a fixed point has ``d_tail -> 0`` and
``R -> 1``; one that keeps wandering has ``R -> 0``. R reads the rollout from the
outside — it does not use the model's own confidence head — so it is a stability
signal *independent of the model*. (A definition, not a theorem.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class RolloutVerdict:
    crystallized: bool
    coherence_R: float
    stability_step: Optional[int]
    tail_displacement: float


def assess_rollout(
    rollout: Sequence[np.ndarray], scale: float = 0.1, settle_eps: float = 0.02
) -> RolloutVerdict:
    if len(rollout) < 2:
        return RolloutVerdict(False, 0.0, None, float("inf"))
    arr = [np.asarray(z, dtype=np.float64).ravel() for z in rollout]
    d = np.array([float(np.linalg.norm(arr[i] - arr[i - 1])) for i in range(1, len(arr))])
    n = len(d)
    tail = d[max(1, (2 * n) // 3) :]
    d_tail = float(np.mean(tail)) if len(tail) else float(d[-1])
    R = max(0.0, min(1.0, float(np.exp(-d_tail / max(scale, 1e-9)))))
    step = next((i + 1 for i in range(n) if bool(np.all(d[i:] < settle_eps))), None)
    return RolloutVerdict(step is not None, R, step, d_tail)
