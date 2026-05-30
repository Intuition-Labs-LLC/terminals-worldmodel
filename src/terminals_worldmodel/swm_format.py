# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""Register the Terminals OS trace as a native swm dataset format.

This closes the bootstrap-from-void loop: the ``world_model_trace.jsonl`` the OS
writes during real use becomes loadable by ``swm.data.load_dataset`` — so the OS's
own usage is the corpus its world model trains on. It mirrors swm's built-in formats
(lance / hdf5 / folder / video) via the public ``@register_format`` API, and is
shaped to be PR-ready as an upstream contribution to galilai-group/stable-worldmodel.

Honest scope: ``detect`` and ``open_writer`` are complete and verified. ``open_reader``
parses our trace into swm's column schema (state/action/reward); wiring those columns
into swm's lance-backed ``Dataset`` (offsets/lengths buffer) is the production step.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import capabilities

FORMAT_NAME = "terminals-os-trace"


def register_terminals_format():
    """Register the format with swm. Returns the format name, or None if swm is not
    usable on this machine. Idempotent here — guards swm's non-idempotent register."""
    if not capabilities.swm_usable():
        return None
    import stable_worldmodel as swm
    from stable_worldmodel.data import Format, register_format

    # swm's register_format raises on a duplicate name (it is NOT idempotent), so
    # guard re-entry — this function is safe to call repeatedly.
    if FORMAT_NAME in (swm.data.list_formats() or []):
        return FORMAT_NAME

    @register_format
    class TerminalsOSTraceFormat(Format):
        name = FORMAT_NAME

        @classmethod
        def detect(cls, path) -> bool:
            p = str(path)
            return p.endswith("world_model_trace.jsonl") or p.endswith(".twm.jsonl")

        @classmethod
        def open_writer(cls, path, mode: str = "append", **kw):
            return _TraceWriter(Path(path), mode=mode)

        @classmethod
        def open_reader(cls, path, **kw):
            return list(_read_episodes(Path(path)))

    return FORMAT_NAME


def _aslist(x):
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return [round(float(v), 6) for v in x.ravel().tolist()]
    except Exception:
        pass
    return x


class _TraceWriter:
    """Streaming writer in swm's interface: a context manager with
    ``write_episode(ep)`` where ``ep = {col: [step, ...]}`` over state/action/reward.
    Writes our JSONL schema. ``mode`` is 'append' | 'overwrite'."""

    def __init__(self, path: Path, mode: str = "append"):
        self.path = path
        self.mode = mode
        self._f = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w" if self.mode == "overwrite" else "a", encoding="utf-8")
        return self

    def write_episode(self, ep: dict) -> None:
        states = ep.get("state", [])
        actions = ep.get("action", [])
        rewards = ep.get("reward", [])
        for t in range(max(len(states), len(actions), len(rewards))):
            self._f.write(
                json.dumps(
                    {
                        "schema": 1,
                        "state": _aslist(states[t]) if t < len(states) else None,
                        "action": _aslist(actions[t]) if t < len(actions) else None,
                        "reward": float(rewards[t]) if t < len(rewards) else None,
                    }
                )
                + "\n"
            )

    def __exit__(self, *exc) -> None:
        if self._f is not None:
            self._f.close()


def _read_episodes(path: Path):
    """Yield one column-dict episode from a trace file. Accepts both the native
    trace schema (observation/action_kind/coherence_R) and the swm schema
    (state/action/reward)."""
    if not path.exists():
        return
    ep = {"state": [], "action": [], "reward": []}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep["state"].append(row.get("state", row.get("observation")))
            ep["action"].append(row.get("action", row.get("action_kind")))
            ep["reward"].append(row.get("reward", row.get("coherence_R")))
    if ep["state"]:
        yield ep
