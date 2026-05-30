# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Intuition Labs LLC
"""Tests for the OS World Model substrate.

The swm-independent half runs anywhere; swm-gated and torch-gated tests skip
honestly when those backends are not installed/usable (the host case).
"""
import sys

import numpy as np
import pytest


from terminals_worldmodel import (  # noqa: E402
    ACTION_KINDS,
    ENV_ID,
    FORMAT_NAME,
    HAS_SWM,
    HAS_TORCH,
    CoherenceCost,
    OSAction,
    PageObservation,
    PlanResult,
    TerminalsWorld,
    ZeroPlanner,
    assess_rollout,
    encode_page_context,
    log_transition,
    proof_of_life,
    read_transitions,
    register_env,
    register_swm_native,
    register_terminals_format,
    run_micro_eval,
    run_planning_demo,
    swm_usable,
    world_model_enabled,
)

_SWM_USABLE = swm_usable()


class TestObservation:
    def test_deterministic_and_fixed_dim(self):
        a = encode_page_context("/solve", visible_metrics={"phi": 0.9})
        b = encode_page_context("/solve", visible_metrics={"phi": 0.9})
        assert a.shape == (64,)
        assert np.allclose(a, b)

    def test_route_changes_vector(self):
        assert not np.allclose(encode_page_context("/solve"), encode_page_context("/bench"))

    def test_metrics_change_vector(self):
        a = encode_page_context("/density", visible_metrics={"R": 0.2})
        b = encode_page_context("/density", visible_metrics={"R": 0.95})
        assert not np.allclose(a, b)


class TestAction:
    def test_kinds(self):
        for k in ACTION_KINDS:
            assert OSAction(kind=k, summary="x").kind == k

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            OSAction(kind="frobnicate", summary="x")

    def test_to_proposed_change_shape(self):
        pc = OSAction("co_edit", "raise timeout to 300s", {"lever": "L1"}).to_proposed_change()
        assert pc["kind"] == "co_edit"
        assert pc["payload"]["lever"] == "L1"


class TestRolloutStability:
    def test_converging_rollout_crystallizes(self):
        v = assess_rollout([np.array([1.0, 0.0, 0.0]) * (0.5 ** i) for i in range(10)])
        assert v.crystallized
        assert v.coherence_R > 0.8

    def test_wandering_rollout_low_R(self):
        v = assess_rollout([np.random.default_rng(i).normal(size=3) for i in range(10)])
        assert v.coherence_R < 0.5
        assert not v.crystallized

    def test_converging_beats_wandering(self):
        conv = assess_rollout([np.ones(3) * (0.6 ** i) for i in range(10)])
        wander = assess_rollout([np.random.default_rng(i).normal(size=3) for i in range(10)])
        assert conv.coherence_R > wander.coherence_R


class TestObservability:
    def test_log_and_read_roundtrip(self, tmp_path):
        p = tmp_path / "trace.jsonl"
        row = log_transition(
            route="/solve",
            action_kind="co_edit",
            action_summary="s",
            coherence_R=0.9,
            crystallized=True,
            path=p,
        )
        rows = read_transitions(p)
        assert len(rows) == 1
        assert rows[0]["route"] == "/solve"
        assert rows[0]["coherence_R"] == 0.9
        assert rows[0]["schema"] == row["schema"]


class TestWorldRecord:
    def test_record_writes_transition(self, tmp_path):
        p = tmp_path / "t.jsonl"
        w = TerminalsWorld(session_id="s1", trace_path=p)
        w.observe("/bench", visible_metrics={"pass_rate": 0.5})
        nxt = PageObservation(route="/bench", visible_metrics={"pass_rate": 0.76})
        w.record(OSAction("co_edit", "raise timeout"), nxt, coherence_R=0.8, crystallized=True)
        rows = read_transitions(p)
        assert len(rows) == 1
        assert rows[0]["session_id"] == "s1"
        assert rows[0]["observation"] is not None
        assert rows[0]["actual_next"] is not None


class TestPlannerDegradesClosed:
    def test_no_model_returns_not_deployed(self):
        res = ZeroPlanner(world_model=None).propose(
            PageObservation(route="/solve"),
            [OSAction("co_edit", "x"), OSAction("no_op", "y")],
        )
        assert isinstance(res, PlanResult)
        assert res.status == "not_deployed"
        assert res.proposals == []  # NEVER fabricates a proposal
        assert res.reason


class TestProofOfLife:
    def test_shape_and_honest_status(self):
        pol = proof_of_life()
        assert pol["component"] == "os_world_model"
        assert pol["status"] in ("not_deployed", "deployed")
        assert pol["stability_detector"]["ok"] is True  # the live half must pass
        assert "micro_eval" in pol
        assert "swm_importable" in pol
        # no GPU on host → not deployed regardless of swm being installed
        if not pol["has_torch_gpu"]:
            assert pol["status"] == "not_deployed"
            assert pol["ok"] is True  # not_deployed is green-honest (shipped-inert)


class TestShippedInert:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("TERMINALS_WORLD_MODEL", raising=False)
        assert world_model_enabled() is False

    def test_opt_in(self, monkeypatch):
        monkeypatch.setenv("TERMINALS_WORLD_MODEL", "1")
        assert world_model_enabled() is True


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (install the [training] extra)")
class TestMicroEval:
    def test_beats_no_model_baseline(self):
        r = run_micro_eval(seed=0)
        assert r.ran is True
        assert r.beats_baseline is True
        assert r.model_mse is not None and r.baseline_mse is not None
        assert r.model_mse < r.baseline_mse


class TestPlanningLoop:
    def test_recovers_good_action_and_filters_noop(self):
        out = run_planning_demo(seed=0)
        assert out["ran"] is True
        assert out["status"] == "ok"
        assert out["n_proposals"] >= 1
        # the goal-moving action (raise timeout) wins the simulate-and-rank
        assert out["top_proposal"]["summary"].startswith("raise per-turn timeout")
        # no_op produces zero goal-progress, so it is never surfaced
        summaries = [p["summary"] for p in out["proposals"]]
        assert "leave the config unchanged" not in summaries
        # proposals are sorted best-first by score
        scores = [p["_score"] for p in out["proposals"]]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.skipif(not HAS_SWM, reason="stable-worldmodel not installed (deploy in ROCm toolbox)")
class TestSwmBacked:
    def test_world_has_backend(self):
        assert TerminalsWorld().has_model_backend is True


@pytest.mark.skipif(not _SWM_USABLE, reason="stable-worldmodel not usable on this machine (env deps)")
class TestSwmNative:
    """The OS as a first-class swm citizen — env harness, data format, cost."""

    def test_register_swm_native(self):
        reg = register_swm_native()
        assert reg["env"] == ENV_ID
        assert reg["format"] == FORMAT_NAME

    def test_env_wraps_in_swm_world(self):
        import stable_worldmodel as swm

        register_env()
        w = swm.World(ENV_ID, num_envs=1, add_pixels=False, goal_conditioned=False)
        assert w is not None
        try:
            w.close()
        except Exception:
            pass

    def test_trace_format_listed_alongside_builtins(self):
        import stable_worldmodel as swm

        register_terminals_format()
        assert FORMAT_NAME in swm.data.list_formats()

    def test_coherence_cost_is_a_solver_model(self):
        import torch
        from stable_worldmodel.solver import CEMSolver

        cost = CoherenceCost(lambda z, a: z + 0.05 * np.resize(a, z.shape), horizon=3)
        c = cost.get_cost({"state": np.zeros(8), "goal": np.ones(8)}, torch.randn(4, 8))
        assert tuple(c.shape) == (4,)
        CEMSolver(model=cost, num_samples=4, n_steps=1, device="cpu")  # accepts our Costable
