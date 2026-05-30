# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Tej Desai / Intuition Labs LLC
"""coherence — model-independent rollout-stability (R = exp(-d_tail/scale)).

Union-distinct: a published clean-name alias over the one canonical implementation.
"""
from .rollout_stability import RolloutVerdict, assess_rollout

__all__ = ["RolloutVerdict", "assess_rollout"]
