# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""Micro world-model eval — the runnable proof behind the proof-of-life contract.

The contract clause is: a learned model must predict the OS's next state better
than assuming nothing changes. This module proves that with a real, deterministic
run: it synthesizes OS transitions ``(z, a) -> z'`` from a fixed linear dynamics,
trains a tiny next-latent predictor, and checks it beats the no-model (identity)
baseline on held-out data.

It runs on host CPU torch — no GPU, no swm. ``stable-worldmodel`` is the
production-grade replacement: same contract (predict-better-than-baseline), a
bigger JEPA model, a real GPU, and real collected traces instead of synthetic ones.
Gated on torch being importable; honest-skips otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import capabilities


@dataclass
class MicroEvalResult:
    ran: bool
    beats_baseline: Optional[bool]
    model_mse: Optional[float]
    baseline_mse: Optional[float]
    margin: Optional[float]
    reason: Optional[str] = None


def run_micro_eval(
    dim: int = 16, n: int = 1024, epochs: int = 400, seed: int = 0
) -> MicroEvalResult:
    """Train a tiny predictor on synthetic OS transitions; beat the identity baseline.

    Deterministic given ``seed`` so the result is a stable, checkable number.
    """
    if not capabilities.HAS_TORCH:
        return MicroEvalResult(
            ran=False,
            beats_baseline=None,
            model_mse=None,
            baseline_mse=None,
            margin=None,
            reason="torch absent; micro-eval skipped (install torch or the [training] extra)",
        )

    import torch

    a_dim = 4
    g = torch.Generator().manual_seed(seed)

    # Synthetic OS dynamics: an action nudges the state. next = z + 0.1 * A[z; a] + noise.
    A = torch.randn(dim, dim + a_dim, generator=g) * 0.3

    def dynamics(z: "torch.Tensor", a: "torch.Tensor") -> "torch.Tensor":
        return z + (torch.cat([z, a], dim=-1) @ A.T) * 0.1

    Z = torch.randn(n, dim, generator=g)
    acts = torch.zeros(n, a_dim)
    idx = torch.randint(0, a_dim, (n,), generator=g)
    acts[torch.arange(n), idx] = 1.0
    noise = torch.randn(n, dim, generator=g) * 0.01
    Znext = dynamics(Z, acts) + noise

    X = torch.cat([Z, acts], dim=-1)
    ntr = int(n * 0.8)
    Xtr, Ytr = X[:ntr], Znext[:ntr]
    Xte, Yte, Zte = X[ntr:], Znext[ntr:], Z[ntr:]

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(dim + a_dim, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, dim),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(model(Xtr), Ytr).backward()
        opt.step()

    with torch.no_grad():
        model_mse = float(loss_fn(model(Xte), Yte))
        baseline_mse = float(loss_fn(Zte, Yte))  # identity: "nothing changes"

    return MicroEvalResult(
        ran=True,
        beats_baseline=bool(model_mse < baseline_mse),
        model_mse=round(model_mse, 6),
        baseline_mse=round(baseline_mse, 6),
        margin=round(baseline_mse - model_mse, 6),
    )
