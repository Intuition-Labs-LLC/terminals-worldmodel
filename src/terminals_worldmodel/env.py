# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Tej Desai / Intuition Labs LLC
"""env — the Gymnasium env swm.World wraps (terminals-os/Config-v0).

Union-distinct: a published clean-name alias over the one canonical implementation.
"""
from .swm_env import ENV_ID, TerminalsOSEnv, register_env

__all__ = ["ENV_ID", "TerminalsOSEnv", "register_env"]
