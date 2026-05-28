# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Intuition Labs LLC
"""Register an agent-usage trace as a stable-worldmodel dataset format.

The trace an agent writes during use — ``(state, action, reward)`` rows — becomes a
first-class swm dataset, loadable via ``swm.data.load_dataset``, so the agent's own
usage is the corpus its world model trains on. Registered through swm's public
``@register_format`` API, alongside the built-in ``lance`` / ``folder`` / ``video``
formats.
"""
from __future__ import annotations

import json
from pathlib import Path

FORMAT_NAME = "terminals-os-trace"


def register_format() -> str:
    """Register the format with swm; returns its name. Idempotent (swm's
    ``register_format`` raises on a duplicate, so re-entry is guarded)."""
    import stable_worldmodel as swm
    from stable_worldmodel.data import Format
    from stable_worldmodel.data import register_format as _register

    if FORMAT_NAME in (swm.data.list_formats() or []):
        return FORMAT_NAME

    @_register
    class TerminalsOSTraceFormat(Format):
        name = FORMAT_NAME

        @classmethod
        def detect(cls, path) -> bool:
            p = str(path)
            return p.endswith("trace.jsonl") or p.endswith(".twm.jsonl")

        @classmethod
        def open_writer(cls, path, mode: str = "append", **kw):
            return _TraceWriter(Path(path), mode)

        @classmethod
        def open_reader(cls, path, **kw):
            return list(_read_episodes(Path(path)))

    return FORMAT_NAME


def _as_list(x):
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return [round(float(v), 6) for v in x.ravel().tolist()]
    except Exception:
        pass
    return x


class _TraceWriter:
    """swm streaming writer: a context manager with ``write_episode(ep)`` where
    ``ep = {col: [step, ...]}`` over state/action/reward. ``mode`` is
    'append' | 'overwrite'."""

    def __init__(self, path: Path, mode: str = "append"):
        self.path = path
        self.mode = mode
        self._f = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w" if self.mode == "overwrite" else "a", encoding="utf-8")
        return self

    def write_episode(self, ep: dict) -> None:
        s, a, r = ep.get("state", []), ep.get("action", []), ep.get("reward", [])
        for t in range(max(len(s), len(a), len(r))):
            self._f.write(
                json.dumps(
                    {
                        "state": _as_list(s[t]) if t < len(s) else None,
                        "action": _as_list(a[t]) if t < len(a) else None,
                        "reward": float(r[t]) if t < len(r) else None,
                    }
                )
                + "\n"
            )

    def __exit__(self, *exc) -> None:
        if self._f is not None:
            self._f.close()


def _read_episodes(path: Path):
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
            ep["state"].append(row.get("state"))
            ep["action"].append(row.get("action"))
            ep["reward"].append(row.get("reward"))
    if ep["state"]:
        yield ep
