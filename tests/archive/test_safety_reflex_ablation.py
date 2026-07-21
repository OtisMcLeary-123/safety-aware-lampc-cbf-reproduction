from lampc_cbf.safety_reflex_ablation import (
    VARIANTS,
    SafetyReflexAblationConfig,
    summarize_safety_reflex_rows,
)


def test_ablation_variants_are_cumulative():
    assert [variant.name for variant in VARIANTS] == [
        "max_clearance_legacy",
        "task_consistent",
        "task_consistent_committed",
        "committed_sampled_data",
        "committed_sampled_data_collision_cone",
    ]
    assert not VARIANTS[0].sampled_data_margin
    assert VARIANTS[-1].barrier_mode == "collision_cone"


def test_ablation_summary_applies_target_gate():
    config = SafetyReflexAblationConfig(episodes=1, workers=1)
    rows = []
    for variant in VARIANTS:
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
                "net_goal_progress": 0.10,
                "reflex_interventions": 1,
                "reflex_backups": 1,
                "final_speed_scale": 1.0,
                "final_clearance_margin": 0.0,
                "solver_rejections": 0,
                "deadline_misses": 0,
                "p99_solve_time": 0.02,
            }
        )

    summary = summarize_safety_reflex_rows(rows, config)

    assert summary["gate"]["passed"]
    assert summary["methods"]["task_consistent"]["mean_net_goal_progress"] == 0.1
