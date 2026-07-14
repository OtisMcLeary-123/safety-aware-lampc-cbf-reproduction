import pytest

from lampc_cbf.dpcbf_ablation import (
    VARIANTS,
    DPCBFAblationConfig,
    summarize_dpcbf_rows,
)


def test_dpcbf_ablation_has_paired_barriers() -> None:
    assert VARIANTS == (
        ("collision_cone_policy_library", "collision_cone"),
        ("dynamic_parabolic_policy_library", "dynamic_parabolic"),
    )


def test_dpcbf_summary_reports_paired_effect() -> None:
    config = DPCBFAblationConfig(episodes=1, workers=1)
    rows = []
    for name, _ in VARIANTS:
        target = name == "dynamic_parabolic_policy_library"
        rows.append(
            {
                "episode": 0,
                "variant": name,
                "outcome": "safety_timeout",
                "success": False,
                "collision": False,
                "steps": 140,
                "minimum_true_clearance": 0.05,
                "final_goal_distance": 0.15,
                "net_goal_progress": 0.15 if target else 0.10,
                "reflex_interventions": 4 if target else 10,
                "reflex_backups": 2,
                "reflex_side_switches": 0,
                "reflex_policy_selections": 4,
                "reflex_robust_recoveries": 1,
                "solver_rejections": 0,
                "deadline_misses": 0,
                "p99_solve_time": 0.02,
            }
        )

    summary = summarize_dpcbf_rows(rows, config)

    assert summary["gate"]["passed"]
    assert summary["paired_target_effect"]["mean_progress_delta"] == pytest.approx(0.05)
    assert summary["paired_target_effect"]["intervention_delta"] == -6
