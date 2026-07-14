from lampc_cbf.collision_cone_liveness_ablation import (
    VARIANTS,
    CollisionConeLivenessAblationConfig,
    summarize_collision_cone_liveness_rows,
)


def test_liveness_variants_are_cumulative() -> None:
    assert [variant.name for variant in VARIANTS] == [
        "collision_cone_baseline",
        "side_latch",
        "policy_library",
        "policy_library_tangential_subgoal",
    ]
    assert not VARIANTS[0].side_latch
    assert VARIANTS[-1].side_latch
    assert VARIANTS[-1].policy_library
    assert VARIANTS[-1].tangential_subgoal


def test_liveness_summary_applies_target_effect_gates() -> None:
    config = CollisionConeLivenessAblationConfig(
        episodes=1,
        workers=1,
        minimum_intervention_reduction=0.5,
    )
    rows = []
    for variant in VARIANTS:
        target = variant.name == "policy_library_tangential_subgoal"
        rows.append(
            {
                "episode": 0,
                "variant": variant.name,
                "outcome": "safety_timeout",
                "success": False,
                "collision": False,
                "steps": 140,
                "minimum_true_clearance": 0.01,
                "final_goal_distance": 0.20,
                "net_goal_progress": 0.10 if target else 0.05,
                "reflex_interventions": 4 if target else 10,
                "reflex_backups": 1,
                "reflex_side_switches": 0,
                "reflex_policy_selections": 4 if target else 0,
                "reflex_robust_recoveries": 0,
                "solver_rejections": 0,
                "deadline_misses": 0,
                "p99_solve_time": 0.02,
            }
        )

    summary = summarize_collision_cone_liveness_rows(rows, config)

    assert summary["gate"]["passed"]
    assert (
        summary["paired_target_effect"][
            "intervention_reduction_vs_collision_cone_baseline"
        ]
        == 0.6
    )
    assert summary["paired_target_effect"]["mean_progress_delta"] == 0.05
