#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Tej Desai / Intuition Labs LLC
#
# build_union_distinct.sh — deterministic repo->lobe BUILD block.
#
# Performs the SQL UNION DISTINCT over two world-model "sematon sets":
#   SRC  = the in-repo superset implementation (grokker contribution)
#   DST  = the published standalone package (clean public names)
# Result: ONE implementation per concept, the published clean names kept as
# thin aliases (two public names, one impl), grokker collapse-detector made
# optional so the package stays standalone. Idempotent: re-running from SRC
# reproduces byte-for-byte. This is the deterministic core the staging
# workflow calls; vary SRC/DST/LICENSE to repeat for the next lobe.
#
# Usage: build_union_distinct.sh <SRC_pkg_dir> <DST_pkg_dir> [LICENSE_SPDX] [VERSION]
set -euo pipefail

SRC="${1:?SRC package dir (e.g. /var/home/zero/irc-strix-halo/src/worldmodel)}"
DST="${2:?DST package dir (e.g. /var/home/zero/terminals-worldmodel/src/terminals_worldmodel)}"
LICENSE_SPDX="${3:-AGPL-3.0-or-later}"
VERSION="${4:-0.2.0}"
COPYRIGHT="Copyright (c) 2026 Tej Desai / Intuition Labs LLC"
PYPROJECT="$(cd "$DST/../.." && pwd)/pyproject.toml"

# --- 1. union: bring the full in-repo implementation surface into the package ---
IMPL_MODULES=(action capabilities config demo micro_eval observability observation
              planner proof_of_life rollout_stability swm_cost swm_env swm_format world)
# Idempotent + survives SRC retirement: graft from source if present; once SRC is
# retired to a shim the package IS canonical, so keep DST. Only a true gap is fatal.
for m in "${IMPL_MODULES[@]}"; do
  if [ -f "$SRC/$m.py" ]; then
    cp "$SRC/$m.py" "$DST/$m.py"           # one-time graft from source
  elif [ -f "$DST/$m.py" ]; then
    :                                       # already grafted (SRC retired) — DST canonical
  else
    echo "ERROR: $m.py missing from SRC ($SRC) and DST ($DST)" >&2; exit 1
  fi
done

# --- 2. relicense every package module to the lobe's SPDX (distinct, coherent) ---
for f in "$DST"/*.py; do
  sed -i "s#SPDX-License-Identifier: \(MIT\|Apache-2.0\)#SPDX-License-Identifier: ${LICENSE_SPDX}#" "$f"
done

# --- 3. standalone: make the grokker collapse_detector an OPTIONAL import ---
python3 - "$DST/rollout_stability.py" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
needle = "from src.irc.collapse_detector import CollapseDetector"
guarded = ("try:  # standalone: the grokker collapse cross-check is optional\n"
           "    from src.irc.collapse_detector import CollapseDetector\n"
           "except Exception:  # noqa: BLE001\n"
           "    CollapseDetector = None")
if needle in s and "except Exception:  # noqa: BLE001" not in s:
    s = s.replace(needle, guarded)
    p.write_text(s)
    print("patched: collapse_detector import is now optional")
else:
    print("skip: collapse_detector import already optional / absent")
PY

# --- 4. distinct: published clean names become thin aliases over the one impl ---
write_alias () { # $1=file  $2=docline  $3=body
  cat > "$DST/$1" <<EOF
# SPDX-License-Identifier: ${LICENSE_SPDX}
# ${COPYRIGHT}
"""$2

Union-distinct: a published clean-name alias over the one canonical implementation.
"""
$3
EOF
}
write_alias coherence.py  "coherence — model-independent rollout-stability (R = exp(-d_tail/scale))." \
$'from .rollout_stability import RolloutVerdict, assess_rollout\n\n__all__ = ["RolloutVerdict", "assess_rollout"]'
write_alias cost.py       "cost — the swm Costable (CoherenceCost: 1 - goal_progress * coherence_R)." \
$'from .swm_cost import CoherenceCost\n\n__all__ = ["CoherenceCost"]'
write_alias dataformat.py "dataformat — the swm dataset format (register_format == register_terminals_format)." \
$'from .swm_format import FORMAT_NAME, register_terminals_format\n\nregister_format = register_terminals_format\n\n__all__ = ["FORMAT_NAME", "register_format", "register_terminals_format"]'
write_alias env.py        "env — the Gymnasium env swm.World wraps (terminals-os/Config-v0)." \
$'from .swm_env import ENV_ID, TerminalsOSEnv, register_env\n\n__all__ = ["ENV_ID", "TerminalsOSEnv", "register_env"]'

# --- 5. union public API: both register() (published) and register_swm_native() (grokker) ---
cat > "$DST/__init__.py" <<EOF
# SPDX-License-Identifier: ${LICENSE_SPDX}
# ${COPYRIGHT}
"""terminals-worldmodel — the union-distinct OS/agent world-model substrate.

First-class, standalone stable-worldmodel (swm) citizen: registers the OS/agent
configuration state as an swm env (which ``swm.World`` wraps), its usage trace as
an swm dataset format, and coherence-R (a model-independent rollout-stability cost)
any swm solver minimizes. Adds the grokker contribution: rollout stability read on
latent displacement (the searching->knowing collapse), a proof-of-life + micro-eval,
an observability trace, and ``ZeroPlanner`` (scores goal_progress x coherence_R, so
inaction never wins on stability alone).

Standalone + degrade-closed: the grokker collapse detector, swm, and torch are all
optional. Built on the MIT ``stable-worldmodel`` as a runtime dependency, never
vendored. This package is ${LICENSE_SPDX}.
"""
from __future__ import annotations

from . import capabilities
from .action import ACTION_KINDS, OSAction
from .capabilities import swm_usable
from .config import LATENT_DIM, trace_path, world_model_enabled
from .demo import DemoWorldModel, run_planning_demo
from .micro_eval import MicroEvalResult, run_micro_eval
from .observability import SCHEMA_VERSION, log_transition, read_transitions
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

__version__ = "${VERSION}"
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
    "assess_rollout", "encode_page_context", "log_transition", "proof_of_life",
    "read_transitions", "register", "register_env", "register_format",
    "register_swm_native", "register_terminals_format", "run_micro_eval",
    "run_planning_demo", "swm_usable", "trace_path", "world_model_enabled",
]
EOF

# --- 6. stamp the release block (version + license) into pyproject ---
if [ -f "$PYPROJECT" ]; then
  sed -i "s#^version = \".*\"#version = \"${VERSION}\"#" "$PYPROJECT"
  sed -i "s#^license = \".*\"#license = \"${LICENSE_SPDX}\"#" "$PYPROJECT"
  # PEP 639: an SPDX license expression and a "License ::" classifier cannot coexist.
  sed -i '/^[[:space:]]*"License :: OSI Approved/d' "$PYPROJECT"
fi

# --- 7. observe: emit a deterministic manifest of the built lobe ---
echo "{\"lobe\":\"$(basename "$(dirname "$(dirname "$DST")")")\",\"version\":\"${VERSION}\",\"license\":\"${LICENSE_SPDX}\",\"modules\":$(ls -1 "$DST"/*.py | wc -l | tr -d ' '),\"impl\":${#IMPL_MODULES[@]},\"aliases\":4}"
