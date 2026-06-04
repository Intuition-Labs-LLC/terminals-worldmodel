# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""ZeroPlanner — simulate-before-commit. The world model's face on the OS.

Given the current screen (a ``PageObservation``) and a set of candidate actions,
the planner consults like a surgeon before cutting: each candidate is ROLLED
FORWARD through the world model and scored, and only changes predicted to improve
the system surface as ProposedChanges, best-first.

**Scoring** = ``goal_progress × coherence_R``:

* ``coherence_R`` is CONFIDENCE — is the predicted rollout stable / did it settle?
* ``goal_progress`` is DESIRABILITY — does the predicted outcome move toward the goal?

Confidence alone would rank *doing nothing* highest (a ``no_op`` trivially settles,
so its rollout is maximally stable). The goal term is what makes the planner prefer
a useful change. Without a ``goal_vector`` it falls back to confidence-only ranking
and says so in the result.

The planner runs whenever a ``world_model`` (any object exposing
``predict(latent, action_vec) -> next_latent``) is provided — that may be a swm
model in the ROCm toolbox or a local predictor. With no model it DEGRADES CLOSED:
``propose()`` returns ``not_deployed`` and never fabricates a proposal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .action import OSAction
from .observation import PageObservation
from .rollout_stability import assess_rollout


@dataclass
class PlanResult:
    status: str  # "ok" | "not_deployed"
    proposals: List[dict]  # ProposedChange dicts, best-first (empty if not_deployed)
    reason: Optional[str] = None
    scored: Optional[list] = None  # [(summary, score)] for transparency


class ZeroPlanner:
    def __init__(self, world_model=None, horizon: int = 10, num_samples: int = 64):
        self.world_model = world_model
        self.horizon = horizon
        self.num_samples = num_samples

    def propose(
        self,
        observation: PageObservation,
        candidate_actions: Sequence[OSAction],
        goal: Optional[str] = None,
        goal_vector: Optional[np.ndarray] = None,
    ) -> PlanResult:
        if self.world_model is None:
            return PlanResult(
                status="not_deployed",
                proposals=[],
                reason=(
                    "no world model loaded; the planner degrades closed and proposes "
                    "nothing (no fabricated proposals). Load a trained model — swm in "
                    "the ROCm toolbox, or a local predictor — to enable planning."
                ),
            )

        o0 = np.asarray(observation.vector(), dtype=np.float64)
        g = None if goal_vector is None else np.asarray(goal_vector, dtype=np.float64)
        d0 = float(np.linalg.norm(o0 - g)) if g is not None else None

        scored = []
        for act in candidate_actions:
            rollout = self._rollout(o0, act)
            verdict = assess_rollout(rollout)
            z_h = np.asarray(rollout[-1], dtype=np.float64)
            if g is not None and d0 and d0 > 1e-9:
                progress = (d0 - float(np.linalg.norm(z_h - g))) / d0
                score = max(0.0, progress) * verdict.coherence_R
            else:
                progress = None
                score = verdict.coherence_R
            scored.append((act, verdict, progress, score))

        scored.sort(key=lambda t: t[3], reverse=True)

        proposals: List[dict] = []
        for act, v, progress, score in scored:
            if score <= 0.0:
                continue  # never surface a change that doesn't help
            pc = act.to_proposed_change()
            pc["_score"] = round(score, 4)
            pc["_coherence_R"] = round(v.coherence_R, 4)
            pc["_goal_progress"] = round(progress, 4) if progress is not None else None
            pc["_crystallized"] = v.crystallized
            # The rollout's terminal latent — the world model's PREDICTION of the
            # state this action settles into. ``_``-prefixed so the proposal-strip
            # in _rank_proposals_by_world_model keeps it off the surfaced wire; it
            # is an internal carrier so the producer can log predicted_next.
            pc["_predicted_next"] = z_h  # np.ndarray, dim=LATENT_DIM
            pc["_collapse_quality"] = (
                round(v.collapse_quality, 4)
                if v.collapse_quality is not None
                else None
            )
            proposals.append(pc)

        return PlanResult(
            status="ok",
            proposals=proposals,
            scored=[(a.summary, round(s, 4)) for a, _, _, s in scored],
        )

    def _rollout(self, o0: np.ndarray, action: OSAction) -> List[np.ndarray]:
        """Roll the world model forward ``horizon`` steps under ``action``."""
        z = np.asarray(o0, dtype=np.float64)
        a = action.vector()
        traj = [z]
        for _ in range(self.horizon):
            z = np.asarray(self.world_model.predict(z, a), dtype=np.float64)
            traj.append(z)
        return traj
