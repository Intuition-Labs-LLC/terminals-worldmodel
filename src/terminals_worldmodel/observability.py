# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""Append-only JSONL trace of world-model transitions.

This is observability and the bootstrap-from-void training corpus in one file.
Every time Zero observes the screen and (maybe) acts, we log one transition:

    (observation o_t, action a_t, predicted next, actual next o_{t+1},
     coherence_R, crystallized, ts)

These rows ARE the dataset the world model self-trains on — the OS bootstraps its
own model of itself from real usage, from void. The shape mirrors the lab's
solve_history.jsonl: passive, additive, schema-stable, safe to tail.
"""
from __future__ import annotations

import json
import time
from typing import List

import numpy as np

from .config import trace_path

SCHEMA_VERSION = 1


def _to_list(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return [round(float(x), 6) for x in v.ravel().tolist()]
    return v


def log_transition(
    *,
    route,
    action_kind,
    action_summary,
    coherence_R,
    crystallized,
    observation=None,
    predicted_next=None,
    actual_next=None,
    session_id=None,
    extra=None,
    path=None,
) -> dict:
    """Append one transition row; returns the row dict that was written."""
    row = {
        "schema": SCHEMA_VERSION,
        "ts": time.time(),
        "route": route,
        "action_kind": action_kind,
        "action_summary": action_summary,
        "coherence_R": round(float(coherence_R), 6),
        "crystallized": bool(crystallized),
        "observation": _to_list(observation),
        "predicted_next": _to_list(predicted_next),
        "actual_next": _to_list(actual_next),
        "session_id": session_id,
    }
    if extra:
        row["extra"] = extra
    p = path or trace_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def read_transitions(path=None) -> List[dict]:
    """Read all transition rows (skips any malformed line)."""
    p = path or trace_path()
    if not p.exists():
        return []
    out: List[dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
