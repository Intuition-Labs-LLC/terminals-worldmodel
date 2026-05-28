# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Intuition Labs LLC
"""terminals-worldmodel — stable-worldmodel for the agent / OS configuration world.

Registers an agent's configuration state as a stable-worldmodel environment, its
usage trace as a swm dataset format, and a coherence-stability planning cost any
swm solver can minimize. ``register()`` wires the env + format into swm.
"""
from .coherence import RolloutVerdict, assess_rollout
from .cost import CoherenceCost
from .dataformat import FORMAT_NAME, register_format
from .env import ENV_ID, TerminalsOSEnv, register_env

__version__ = "0.1.0"


def register() -> dict:
    """Register the env + dataset format with stable-worldmodel. Returns what
    registered, e.g. ``{'env': 'terminals-os/Config-v0', 'format': 'terminals-os-trace'}``."""
    return {"env": register_env(), "format": register_format()}


__all__ = [
    "CoherenceCost",
    "ENV_ID",
    "FORMAT_NAME",
    "RolloutVerdict",
    "TerminalsOSEnv",
    "__version__",
    "assess_rollout",
    "register",
    "register_env",
    "register_format",
]
