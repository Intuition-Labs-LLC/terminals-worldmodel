# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""Rollout stability — the grokker contribution to the world model.

A JEPA world model rolls a latent forward under a candidate action sequence:
``z_0 -> z_1 -> ... -> z_H``. The rollout is STABLE / CRYSTALLIZED when the latent
stops moving (Δz -> 0): the model has settled on a predicted outcome. That is
exactly the searching -> knowing transition this repo's collapse detector measures,
read on the latent-displacement signal instead of on token perplexity.

``coherence_R`` in [0, 1] is DEFINED here (a definition, not a theorem):

    R = exp(-d_tail / scale)

where ``d_tail`` is the mean step displacement over the final third of the rollout.
A rollout that converges to a fixed point has ``d_tail -> 0`` and ``R -> 1``; a
rollout that keeps wandering has large ``d_tail`` and ``R -> 0``. This R is the
world model's own contribution to the coherence-R bus (provenance kind would be
ROLLOUT_STABILITY) — an independent stability measure that does not trust the
model's own confidence head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:  # standalone: the grokker collapse cross-check is optional
    # Put the grokker repo on sys.path so the installed package resolves the
    # collapse detector regardless of cwd — same convention as zero_router.py.
    import os
    import sys

    _g = os.environ.get("GROKKER_REPO", "/var/home/zero/irc-strix-halo")
    _g in sys.path or sys.path.insert(0, _g)
    from src.irc.collapse_detector import CollapseDetector
except Exception:  # noqa: BLE001
    CollapseDetector = None


@dataclass
class RolloutVerdict:
    crystallized: bool
    coherence_R: float
    stability_step: Optional[int]  # first step the rollout settled, if it did
    tail_displacement: float
    quality: float  # clean, R-derived selection score
    collapse_quality: Optional[float] = None  # cross-check from the collapse detector


def _displacements(rollout: Sequence[np.ndarray]) -> np.ndarray:
    arr = [np.asarray(z, dtype=np.float64).ravel() for z in rollout]
    return np.array(
        [float(np.linalg.norm(arr[i] - arr[i - 1])) for i in range(1, len(arr))]
    )


def assess_rollout(
    rollout: Sequence[np.ndarray], scale: float = 0.1, settle_eps: float = 0.02
) -> RolloutVerdict:
    """Score a latent rollout for stability/crystallization. Pure numpy."""
    if len(rollout) < 2:
        return RolloutVerdict(False, 0.0, None, float("inf"), 0.0, None)

    d = _displacements(rollout)
    n = len(d)
    tail = d[max(1, (2 * n) // 3) :]
    d_tail = float(np.mean(tail)) if len(tail) else float(d[-1])
    R = max(0.0, min(1.0, float(np.exp(-d_tail / max(scale, 1e-9)))))

    stability_step: Optional[int] = None
    for i in range(n):
        if bool(np.all(d[i:] < settle_eps)):
            stability_step = i + 1
            break
    crystallized = stability_step is not None
    quality = R if crystallized else R * 0.5

    # Cross-check with the collapse detector on the displacement signal. Best-effort:
    # it demonstrates the grokker linkage but never gates the clean verdict above.
    collapse_quality: Optional[float] = None
    try:
        events = CollapseDetector().detect_in_trajectory(
            [float(max(x, 1e-6)) for x in d.tolist()]
        )
        if events:
            collapse_quality = float(max(e.quality_score for e in events))
    except Exception:
        collapse_quality = None

    return RolloutVerdict(crystallized, R, stability_step, d_tail, quality, collapse_quality)
