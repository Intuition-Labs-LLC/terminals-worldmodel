# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""TerminalsWorld — the OS state as an swm-shaped environment.

The "world" is the operator's live session on Terminals OS. An observation is the
page context Zero already captures; an action is a ProposedChange the operator
accepted; a transition is what the screen/metrics became afterward. Unlike a
robotics environment, this world is not simulated — it RECORDS real transitions for
the world model to learn from (bootstrap-from-void). The data pipeline runs whether
or not a model is trained yet: collecting is the first half of the loop.

When swm is installed this adapts to swm.World's collect()/evaluate() interface;
without swm it still records transitions to the trace.
"""
from __future__ import annotations

from typing import Optional

from . import capabilities
from .action import OSAction
from .observability import log_transition
from .observation import PageObservation

WORLD_ID = "terminals-os-v0"


class TerminalsWorld:
    def __init__(self, session_id: Optional[str] = None, trace_path=None):
        self.session_id = session_id
        self._trace_path = trace_path
        self._last_obs: Optional[PageObservation] = None

    def observe(
        self, route, selected_text=None, visible_metrics=None, user_role_hint=None
    ) -> PageObservation:
        obs = PageObservation(
            route=route,
            selected_text=selected_text,
            visible_metrics=dict(visible_metrics or {}),
            user_role_hint=user_role_hint,
        )
        self._last_obs = obs
        return obs

    def record(
        self,
        action: OSAction,
        next_obs: PageObservation,
        coherence_R: float,
        crystallized: bool,
        predicted_next=None,
    ) -> dict:
        """Record a REAL (obs, action, next_obs) transition — one training row."""
        prev = self._last_obs
        row = log_transition(
            route=(prev.route if prev else next_obs.route),
            action_kind=action.kind,
            action_summary=action.summary,
            coherence_R=coherence_R,
            crystallized=crystallized,
            observation=(prev.vector() if prev else None),
            predicted_next=predicted_next,
            actual_next=next_obs.vector(),
            session_id=self.session_id,
            path=self._trace_path,
        )
        self._last_obs = next_obs
        return row

    @property
    def has_model_backend(self) -> bool:
        return capabilities.HAS_SWM
