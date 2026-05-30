# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""Capability probes — what the substrate can actually do on THIS machine, now.

Honest degrade depends on these being truthful rather than aspirational. Two real
gotchas this encodes:

* Host torch on a Strix Halo box is CPU-only (``cuda_avail False``); the real GPU
  path lives in the ROCm toolbox. ``HAS_TORCH_GPU`` is computed from the live
  runtime, not assumed.
* ``stable-worldmodel`` 0.1.0's top-level import eagerly pulls its env subsystem
  (needs cv2/imageio/...), so "installed" (``find_spec``) does NOT imply
  "importable". ``swm_usable()`` does the real, lazy, cached import check that the
  ``find_spec`` flag cannot.
"""
from __future__ import annotations

import importlib.util
from typing import Optional


def _have(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


HAS_SWM = _have("stable_worldmodel")  # distribution installed (cheap, may not import)
HAS_TORCH = _have("torch")

_SWM_IMPORT_CHECK = "unchecked"


def swm_import_error() -> Optional[str]:
    """None if swm imports cleanly, else a short reason. Lazy + cached; only
    diagnostics call this (a full swm import loads torch + its env subsystem)."""
    global _SWM_IMPORT_CHECK
    if _SWM_IMPORT_CHECK == "unchecked":
        if not HAS_SWM:
            _SWM_IMPORT_CHECK = "stable-worldmodel not installed"
        else:
            try:
                import stable_worldmodel  # noqa: F401

                _SWM_IMPORT_CHECK = None
            except Exception as e:
                _SWM_IMPORT_CHECK = f"{type(e).__name__}: {e}"
    return _SWM_IMPORT_CHECK


def swm_usable() -> bool:
    """True iff swm is installed AND actually importable on this machine."""
    return HAS_SWM and swm_import_error() is None


def has_torch_gpu() -> bool:
    """True only if torch is importable AND a GPU device is actually available."""
    if not HAS_TORCH:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


HAS_TORCH_GPU = has_torch_gpu()
