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


def fold_actual_next(session_id, actual_obs_vector, path=None) -> bool:
    """Close the predict->actual loop: patch the newest OPEN row for ``session_id``
    (one with ``predicted_next`` set and ``actual_next`` still ``None``) by writing
    this turn's observation into its ``actual_next``.

    The producer logs ONE row at proposal time — predicted_next set, actual_next
    None. The NEXT same-session observation IS the actual outcome of the prior
    turn's predicted action, so it folds back here. Once folded, the pair
    ``(predicted_next, actual_next)`` is a closed, measurable transition:
    ``error = ||predicted_next - actual_next||``.

    Append-only-safe: rewrites the file in place ONLY when exactly one open row is
    found (the newest), and is a no-op (returns False) otherwise — no row to fold,
    no trace file, or a bad vector all degrade closed. For n=1 single-user the
    in-place rewrite is correct and simple; higher concurrency would swap a sidecar
    index. Returns True iff a row was folded.
    """
    p = path or trace_path()
    if not p.exists():
        return False
    try:
        folded = _to_list(actual_obs_vector)
    except Exception:
        return False
    if folded is None:
        return False
    rows = read_transitions(p)
    if not rows:
        return False
    target = None
    for i in range(len(rows) - 1, -1, -1):  # newest first
        r = rows[i]
        if (
            r.get("session_id") == session_id
            and r.get("predicted_next") is not None
            and r.get("actual_next") is None
        ):
            target = i
            break
    if target is None:
        return False  # no open row for this session → no-op
    rows[target]["actual_next"] = folded
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(p)  # atomic on POSIX
    return True
