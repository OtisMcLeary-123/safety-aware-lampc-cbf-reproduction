import pytest

from lampc_cbf.timeout_recovery_benchmark import (
    VARIANTS,
    TimeoutRecoveryBenchmarkConfig,
    summarize_timeout_recovery_rows,
)


def test_timeout_ablation_is_factorial_and_excludes_uncalibrated_gain() -> None:
    factors = {
        (variant.profile_recovery, variant.observed_velocity)
        for variant in VARIANTS
    }
    assert factors == {(False, False), (True, False), (False, True), (True, True)}


def test_timeout_ablation_summary_reports_progress_and_profile_release() -> None:
    rows = []
    for variant in VARIANTS:
        rows.append(
            {
                "variant": variant.name,
                "success": False,
                "collision": False,
                "outcome": "safety_timeout",
                "steps": 140,
                "final_goal_distance": 0.2,
                "net_goal_progress": 0.1,
                "mean_model_transition_error": 0.001,
                "solver_rejections": 0,
                "deadline_misses": 0,
                "p99_solve_time": 0.01,
                "feedback_causal_opportunity": True,
                "final_speed_scale": 1.0 if variant.profile_recovery else 0.65,
                "final_clearance_margin": 0.0 if variant.profile_recovery else 0.02,
            }
        )
    summary = summarize_timeout_recovery_rows(
        rows,
        TimeoutRecoveryBenchmarkConfig(episodes=1, workers=1),
    )

    assert summary["action_calibration_included"] is False
    assert summary["methods"]["recovery_observed_velocity"][
        "nominal_profile_at_end_rate"
    ] == pytest.approx(1.0)
    assert summary["methods"]["sticky_command_velocity"][
        "nominal_profile_at_end_rate"
    ] == pytest.approx(0.0)
    assert summary["gate"]["passed"] is True
