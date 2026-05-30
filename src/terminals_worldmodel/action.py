# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""The OS World Model action space: the changes Zero can propose to the screen.

``ACTION_KINDS`` mirrors the dashboard ``ProposedChange.kind`` literals exactly
(``co_edit`` / ``navigate`` / ``explain`` / ``no_op``) so a planned action
serializes straight into a Zero chat ProposedChange with no translation layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .config import LATENT_DIM
from .observation import _bucket

ACTION_KINDS = ("co_edit", "navigate", "explain", "no_op")


@dataclass
class OSAction:
    """One candidate change. ``kind`` must be an ``ACTION_KINDS`` literal."""

    kind: str
    summary: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise ValueError(
                f"unknown action kind {self.kind!r}; expected one of {ACTION_KINDS}"
            )

    def to_proposed_change(self) -> dict:
        """Serialize to the dashboard ProposedChange wire shape."""
        return {"kind": self.kind, "summary": self.summary, "payload": dict(self.payload)}

    def vector(self, dim: int = LATENT_DIM) -> np.ndarray:
        """Encode the action for conditioning the world-model predictor.

        Slots 0..3 one-hot the kind; the summary and payload hash into [4, dim).
        """
        v = np.zeros(dim, dtype=np.float32)
        v[ACTION_KINDS.index(self.kind)] = 1.0
        v[_bucket(self.summary[:128], dim, "asum", lo=4)] += 0.5
        for k, val in sorted(self.payload.items(), key=lambda kv: str(kv[0])):
            v[_bucket(f"{k}={val}"[:128], dim, "apay", lo=4)] += 0.5
        return v
