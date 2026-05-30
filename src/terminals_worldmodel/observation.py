# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""Encode a Zero ``PageContext`` into a fixed-dimension observation vector.

The "world" the OS World Model models is the OS's own state: which page the
operator is on, what is selected, and the metrics that page surfaced. This module
turns that page context into a deterministic numeric vector — the observation
``o_t`` the world model conditions on. It is pure numpy + feature hashing: no
torch, no swm, so the encoder is identical on the host and inside the ROCm toolbox
and runs anywhere.

This is the v0 *structured-state* encoder. When the swm JEPA encoder is wired, a
visual/DOM-image observation can be concatenated; the structured features here stay
the stable, fast backbone for the route+metrics state that dominates OS config.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from .config import LATENT_DIM

# Slots 0..2 are reserved, named features. Hashed features land in [3, dim).
_RESERVED = 3


def _digest_int(text: str, salt: str) -> int:
    h = hashlib.blake2b((salt + "\x00" + text).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big")


def _bucket(text: str, dim: int, salt: str, lo: int = _RESERVED) -> int:
    span = max(dim - lo, 1)
    return lo + (_digest_int(text, salt) % span)


def _unit(text: str, salt: str) -> float:
    return (_digest_int(text, salt) % 1_000_003) / 1_000_003.0


@dataclass
class PageObservation:
    """The screen state Zero sees, mirroring the dashboard ``PageContext`` model."""

    route: str
    selected_text: Optional[str] = None
    visible_metrics: Mapping[str, Any] = field(default_factory=dict)
    user_role_hint: Optional[str] = None

    def vector(self, dim: int = LATENT_DIM) -> np.ndarray:
        return encode_page_context(
            self.route, self.selected_text, self.visible_metrics, self.user_role_hint, dim
        )


def encode_page_context(
    route: str,
    selected_text: Optional[str] = None,
    visible_metrics: Optional[Mapping[str, Any]] = None,
    user_role_hint: Optional[str] = None,
    dim: int = LATENT_DIM,
) -> np.ndarray:
    """Deterministically encode page context to an L2-normalized float32 vector."""
    v = np.zeros(dim, dtype=np.float32)

    segs = [s for s in route.split("/") if s]
    v[_bucket(route, dim, "route")] += 1.0
    for i, s in enumerate(segs):
        v[_bucket(s, dim, f"seg{min(i, 4)}")] += 0.5
    v[0] = min(len(segs) / 6.0, 1.0)  # path depth

    if selected_text:
        v[1] = min(len(selected_text) / 4096.0, 1.0)  # selection size
        v[_bucket(selected_text[:256], dim, "seltext")] += 0.5

    if visible_metrics:
        for k, val in sorted(visible_metrics.items(), key=lambda kv: str(kv[0])):
            b = _bucket(str(k), dim, "mkey")
            if isinstance(val, bool):
                v[b] += 1.0 if val else -1.0
            elif isinstance(val, (int, float)) and math.isfinite(float(val)):
                v[b] += float(np.tanh(float(val)))  # squash unbounded metrics to [-1, 1]
            else:
                v[b] += _unit(str(val)[:128], "mval")

    if user_role_hint:
        v[2] = _unit(user_role_hint, "role")

    norm = float(np.linalg.norm(v))
    if norm > 0.0:
        v = (v / norm).astype(np.float32)
    return v
