# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""Configuration and gates for the OS World Model substrate.

Shipped-inert by default: the substrate is OFF unless ``TERMINALS_WORLD_MODEL=1``.
This is the lab's heavy-asset default — wire it complete, ship it OFF, enable it
with one flag. The observation/stability halves still run for tests and tooling;
the gate governs whether Zero is *allowed to plan against it in production*.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_ENABLE = "TERMINALS_WORLD_MODEL"
ENV_TRACE_PATH = "TERMINALS_WORLD_MODEL_TRACE"

# Observation/latent width. Small on purpose: an OS state is structured (a route
# plus a handful of metrics), not a megapixel image, and a narrow latent keeps the
# planner's inner rollout inside a single NPU GEMM tile.
LATENT_DIM = 64

_TRUE = {"1", "true", "yes", "on"}


def world_model_enabled() -> bool:
    """True iff the operator opted the substrate in. Default OFF (shipped-inert)."""
    return os.environ.get(ENV_ENABLE, "0").strip().lower() in _TRUE


def trace_path() -> Path:
    """Where transition rows are appended. Override with ``TERMINALS_WORLD_MODEL_TRACE``.

    Default mirrors the lab's solve_history location convention under ``~/.codebox``.
    """
    raw = os.environ.get(ENV_TRACE_PATH)
    if raw:
        return Path(raw).expanduser()
    base = Path(os.environ.get("HOME", "/tmp")) / ".codebox" / "logs"
    return base / "world_model_trace.jsonl"
