# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Tej Desai / Intuition Labs LLC
"""cost — the swm Costable (CoherenceCost: 1 - goal_progress * coherence_R).

Union-distinct: a published clean-name alias over the one canonical implementation.
"""
from .swm_cost import CoherenceCost

__all__ = ["CoherenceCost"]
