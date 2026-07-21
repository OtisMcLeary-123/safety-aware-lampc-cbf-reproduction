import json

import pytest

from lampc_cbf.liveness_development import (
    LivenessDevelopmentConfig,
    PhysicsTimeoutInputs,
    derive_physics_timeout,
    summarize_liveness_development_rows,
    validate_liveness_prerequisite,
    wilson_interval,
)


def test_physics_timeout_derives_frozen_220_step_budget() -> None:
    budget = derive_physics_timeout(PhysicsTimeoutInputs())

    assert budget.direct_path_time == pytest.approx(3.75)
    assert budget.detour_radius == pytest.approx(0.215)
    assert budget.detour_extra_distance == pytest.approx(0.2454424, rel=1e-6)
    assert budget.sensing_and_recovery_reserve == pytest.approx(
        1.967189683956094
    )
    assert budget.total_seconds == pytest.approx(8.785219940478664)
    assert budget.max_steps == 220


def test_prerequisite_requires_passed_liveness_protocol(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "protocol": "collision-cone-liveness-ablation-v1",
                "gate": {"passed": True},
                "methods": {"policy_library_tangential_subgoal": {}},
            }
        ),
        encoding="utf-8",
    )
    assert validate_liveness_prerequisite(path)["gate"]["passed"]

    path.write_text(
        json.dumps(
            {
                "protocol": "collision-cone-liveness-ablation-v1",
                "gate": {"passed": False},
                "methods": {"policy_library_tangential_subgoal": {}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not pass"):
        validate_liveness_prerequisite(path)


def test_wilson_interval_handles_zero_events() -> None:
    lower, upper = wilson_interval(0, 100)
    assert lower == 0.0
    assert upper == pytest.approx(0.0369935, rel=1e-5)


def test_development_summary_applies_safety_and_timing_gate() -> None:
    config = LivenessDevelopmentConfig(episodes=2, workers=1)
    timeout = derive_physics_timeout(config.timeout_inputs)
    rows = [
        {
            "episode": episode,
            "outcome": "goal" if episode == 0 else "safety_timeout",
            "success": episode == 0,
            "collision": False,
            "steps": timeout.max_steps,
            "minimum_true_clearance": 0.04,
            "net_goal_progress": 0.25,
            "reflex_interventions": 10,
            "reflex_backups": 5,
            "reflex_robust_recoveries": 2,
            "reflex_side_switches": 0,
            "solver_rejections": 0,
            "deadline_misses": 0,
            "p99_solve_time": 0.02,
        }
        for episode in range(2)
    ]

    summary = summarize_liveness_development_rows(rows, config, timeout)

    assert summary["gate"]["passed"]
    assert summary["success_rate"] == 0.5
    assert summary["physics_timeout"]["max_steps"] == 220
    assert summary["outcomes"]["safety_timeout"] == 1
    assert summary["total_control_steps"] == 440
    assert summary["reflex_intervention_rate"] == pytest.approx(20 / 440)
