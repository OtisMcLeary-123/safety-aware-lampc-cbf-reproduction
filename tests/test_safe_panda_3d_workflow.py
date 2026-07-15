from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lampc_cbf.hf_llm import GammaDecision
from lampc_cbf.table4_scenarios import load_scenarios


SCRIPT_PATH = Path("scripts/run_safe_panda_3d_benchmark.py")
SETUP_PATH = Path("configs/safe_panda_3d_avoidance_50_setup.json")


def _module():
    spec = importlib.util.spec_from_file_location("safe_panda_3d_workflow", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decision(index: int) -> GammaDecision:
    return GammaDecision(
        gamma=0.1,
        safety_level=4,
        explanation="test",
        latency_seconds=0.02,
        fallback_used=False,
        raw_response="{\"gamma\": 0.1}",
        provider="nvidia_nim",
        model="meta/llama-3.1-405b-instruct",
        cache_hit=False,
        requested_at_unix=float(index),
        prompt_hash=f"prompt-{index}",
        request_hash=f"request-{index}",
        error_type=None,
    )


def _result() -> SimpleNamespace:
    return SimpleNamespace(
        outcome="goal", reached_goal=True, collision=False, steps=4,
        minimum_true_clearance=0.1, minimum_true_cbf_residual=0.01,
        final_goal_distance=0.01, final_gamma=0.1, gamma_updates_applied=1,
        solver_failures=0, solver_rejections=0, deadline_misses=0,
        emergency_fallbacks=0,
    )


def test_setup_is_an_opt_in_three_dimensional_profile() -> None:
    setup = json.loads(SETUP_PATH.read_text(encoding="utf-8"))
    overrides = _module()._profile_overrides(setup)
    assert overrides["reference_mode"] == "behind_spline"
    assert overrides["reference_route_profile"] == "3d_waypoints"
    assert overrides["avoidance_waypoint_offsets"][0][0] != 0.0
    assert overrides["avoidance_waypoint_offsets"][0][2] != 0.0
    assert overrides["reflex_tangential_subgoal_enabled"] is True


def test_resumable_runner_skips_completed_rows(tmp_path: Path) -> None:
    module = _module()
    setup = json.loads(SETUP_PATH.read_text(encoding="utf-8"))
    scenarios = load_scenarios(setup["scenario_path"])
    decisions = tuple(_decision(index) for index in range(50))
    calls = []

    def runner(config):
        calls.append(config)
        return _result()

    summary = module.run_resumable_benchmark(
        scenarios, decisions, setup, output_dir=tmp_path, runner=runner
    )
    assert len(calls) == 100
    assert summary["rows"] == 100
    calls.clear()
    resumed = module.run_resumable_benchmark(
        scenarios, decisions, setup, output_dir=tmp_path, runner=runner
    )
    assert calls == []
    assert resumed == summary
    assert len((tmp_path / "run_checkpoint.json").read_text(encoding="utf-8")) > 0


def test_setup_change_is_rejected_when_resuming(tmp_path: Path) -> None:
    module = _module()
    setup = json.loads(SETUP_PATH.read_text(encoding="utf-8"))
    scenarios = load_scenarios(setup["scenario_path"])
    decisions = tuple(_decision(index) for index in range(50))
    module.run_resumable_benchmark(
        scenarios, decisions, setup, output_dir=tmp_path, runner=lambda config: _result()
    )
    changed = json.loads(json.dumps(setup))
    changed["controller"]["position_q_weights"][0] = 2.0
    with pytest.raises(ValueError, match="different setup"):
        module.run_resumable_benchmark(
            scenarios, decisions, changed, output_dir=tmp_path, runner=lambda config: _result()
        )
