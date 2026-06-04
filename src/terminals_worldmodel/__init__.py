# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Tej Desai / Intuition Labs LLC
"""terminals-worldmodel — the union-distinct OS/agent world-model substrate.

First-class, standalone stable-worldmodel (swm) citizen: registers the OS/agent
configuration state as an swm env (which swm.World wraps), its usage trace as
an swm dataset format, and coherence-R (a model-independent rollout-stability cost)
any swm solver minimizes. Adds the grokker contribution: rollout stability read on
latent displacement (the searching->knowing collapse), a proof-of-life + micro-eval,
an observability trace, and ZeroPlanner (scores goal_progress x coherence_R, so
inaction never wins on stability alone).

Standalone + degrade-closed: the grokker collapse detector, swm, and torch are all
optional. Built on the MIT stable-worldmodel as a runtime dependency, never
vendored. This package is AGPL-3.0-or-later.
"""
from __future__ import annotations

from . import capabilities
from .action import ACTION_KINDS, OSAction
from .capabilities import swm_usable
from .config import LATENT_DIM, trace_path, world_model_enabled
from .demo import DemoWorldModel, run_planning_demo
from .micro_eval import MicroEvalResult, run_micro_eval
from .observability import (
    SCHEMA_VERSION,
    fold_actual_next,
    log_transition,
    read_transitions,
)
from .observation import PageObservation, encode_page_context
from .planner import PlanResult, ZeroPlanner
from .proof_of_life import proof_of_life
from .rollout_stability import RolloutVerdict, assess_rollout
from .swm_cost import CoherenceCost
from .swm_env import ENV_ID, TerminalsOSEnv, register_env
from .swm_format import FORMAT_NAME, register_terminals_format

# published clean-name alias (one impl, two names — union-distinct)
register_format = register_terminals_format
from .world import WORLD_ID, TerminalsWorld

__version__ = "0.2.0"
HAS_SWM = capabilities.HAS_SWM
HAS_TORCH = capabilities.HAS_TORCH
HAS_TORCH_GPU = capabilities.HAS_TORCH_GPU


def register() -> dict:
    """Register the env + dataset format with swm (published API). Returns the ids."""
    return {"env": (ENV_ID if register_env() else None), "format": register_terminals_format()}


def register_swm_native() -> dict:
    """Register the OS as a first-class swm citizen — env + trace format (grokker API).
    Idempotent. Values are None when swm / gymnasium are unavailable."""
    return {"env": (ENV_ID if register_env() else None), "format": register_terminals_format()}


__all__ = [
    "ACTION_KINDS", "CoherenceCost", "DemoWorldModel", "ENV_ID", "FORMAT_NAME",
    "HAS_SWM", "HAS_TORCH", "HAS_TORCH_GPU", "LATENT_DIM", "MicroEvalResult",
    "OSAction", "PageObservation", "PlanResult", "RolloutVerdict", "SCHEMA_VERSION",
    "TerminalsOSEnv", "TerminalsWorld", "WORLD_ID", "ZeroPlanner", "__version__",
    "assess_rollout", "encode_page_context", "fold_actual_next", "log_transition",
    "proof_of_life",
    "read_transitions", "register", "register_env", "register_format",
    "register_swm_native", "register_terminals_format", "run_micro_eval",
    "run_planning_demo", "swm_usable", "trace_path", "world_model_enabled",
]
