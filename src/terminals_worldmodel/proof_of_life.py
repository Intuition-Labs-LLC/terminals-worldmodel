# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""Proof-of-life for the OS World Model substrate.

The contract has two clauses. (1) The substrate must be HONEST about whether it is
deployed. (2) When deployed, it must demonstrate it predicts better than a no-model
baseline. This function is what a codebox-doctor world-model section would call.

On the host (no swm, CPU-only torch) it returns ``status="not_deployed"`` — a
GREEN-honest result, because the substrate is shipped-inert by design. The
always-live half (the pure-numpy stability detector) is still proven on a synthetic
rollout, and — when torch is present — a micro world-model eval proves the learned
predictor beats a no-model baseline, so the doctor has real signal even before the
GPU backend lands.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from . import capabilities
from .config import world_model_enabled
from .micro_eval import run_micro_eval
from .rollout_stability import assess_rollout


def proof_of_life(quick: bool = True) -> dict:
    base = {
        "component": "os_world_model",
        "enabled": world_model_enabled(),
        "has_swm": capabilities.HAS_SWM,
        "swm_importable": capabilities.swm_usable(),
        "swm_import_error": capabilities.swm_import_error(),
        "has_torch_gpu": capabilities.HAS_TORCH_GPU,
    }

    # Always-live half: prove the stability detector on a known-converging rollout
    # and a known-wandering one. Deterministic (seeded), so the verdict is stable.
    converging = [np.array([1.0, 0.0, 0.0]) * (0.5 ** i) for i in range(8)]
    wandering = [np.random.default_rng(i).normal(size=3) for i in range(8)]
    v_conv = assess_rollout(converging)
    v_wander = assess_rollout(wandering)
    stability_ok = bool(
        v_conv.crystallized
        and v_conv.coherence_R > 0.8
        and v_conv.coherence_R > v_wander.coherence_R
    )
    base["stability_detector"] = {
        "ok": stability_ok,
        "R_converging": round(v_conv.coherence_R, 4),
        "R_wandering": round(v_wander.coherence_R, 4),
        # Grokker cross-check: the collapse detector's own quality score on the
        # converging rollout's displacement signal. Honest None until the grokker
        # CollapseDetector import resolves (it gates itself in assess_rollout).
        "collapse_quality": v_conv.collapse_quality,
    }

    # Mechanism proof: a tiny learned predictor beats the no-model (identity)
    # baseline on held-out OS transitions. Runs on host CPU torch; swm is the
    # production scale-up of exactly this.
    base["micro_eval"] = asdict(run_micro_eval())

    if not (capabilities.swm_usable() and capabilities.HAS_TORCH_GPU):
        base["status"] = "not_deployed"
        base["ok"] = stability_ok
        base["reason"] = (
            "world model backend not deployed (swm / GPU torch absent). The "
            "stability detector is verified live and the planner degrades closed. "
            "Install stable-worldmodel in the ROCm toolbox to deploy the full substrate."
        )
        return base

    base["status"] = "deployed"
    base["model_eval"] = _swm_eval(quick=quick)
    base["ok"] = bool(stability_ok and base["model_eval"].get("beats_baseline"))
    return base


def _swm_eval(quick: bool) -> dict:
    """Production train/eval against swm. Reached only when swm + GPU torch present.

    Imports are local so the module stays import-clean on a host without swm. Until
    the production harness lands it reports honestly rather than fabricating a pass.
    """
    try:
        import stable_worldmodel as swm  # noqa: F401

        return {
            "beats_baseline": None,
            "note": "swm present; production micro train/eval harness wires next",
        }
    except Exception as e:  # pragma: no cover - only on a broken swm install
        return {"beats_baseline": False, "error": str(e)}
