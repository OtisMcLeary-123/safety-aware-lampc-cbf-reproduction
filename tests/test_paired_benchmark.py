import pytest

from lampc_cbf.paired_benchmark import (
    METHODS,
    PairedBenchmarkConfig,
    evaluate_confirmatory_efficacy_gate,
    exact_mcnemar_pvalue,
    summarize_paired_rows,
)


def test_default_protocol_has_500_common_conditions_and_isolates_feedback():
    config = PairedBenchmarkConfig()
    assert config.stage == "confirmatory"
    assert config.episodes == 500
    assert config.bootstrap_resamples == 10_000
    feedback = next(method for method in METHODS if method.online_feedback)
    fixed = next(method for method in METHODS if method.name == feedback.comparator)
    assert feedback.gamma == fixed.gamma == pytest.approx(0.15)
    assert feedback.prediction_mode == fixed.prediction_mode == "velocity_tube"
    assert feedback.safety_reflex_enabled == fixed.safety_reflex_enabled
    assert feedback.optimal_decay_weight == fixed.optimal_decay_weight


def test_protocol_separates_paper_fidelity_from_robust_extension():
    paper = next(method for method in METHODS if method.name == "proactive_cbf_static_g002")
    robust = next(method for method in METHODS if method.name == "predictive_reflex_g002")
    assert paper.experiment_profile == "paper_fidelity"
    assert paper.delta_u_weight == pytest.approx(0.5)
    assert paper.reference_mode == "direct_target"
    assert robust.experiment_profile == "robust_extension"
    assert robust.delta_u_weight == pytest.approx(2.0)
    assert robust.reference_mode == "straight"


def test_exact_mcnemar_handles_ties_and_one_sided_discordance():
    assert exact_mcnemar_pvalue([True, False], [True, False]) == 1.0
    assert exact_mcnemar_pvalue([False] * 6, [True] * 6) == pytest.approx(0.03125)


def _row(method, episode, success):
    return {
        "episode": episode,
        "method": method.name,
        "comparator": method.comparator,
        "outcome": "goal" if success else "safety_timeout",
        "joint_success": success,
        "success": success,
        "reached_goal": success,
        "collision": False,
        "minimum_true_clearance": 0.01 if success else -0.01,
        "avoidance_onset_time": 0.2,
        "minimum_predicted_ttc": 0.4,
        "path_length": 0.3,
        "jerk_rms": 1.0,
        "mean_solve_time": 0.008,
        "max_solve_time": 0.012,
        "gamma_updates_applied": int(method.online_feedback),
        "gamma_updates_rejected": 0,
        "reflex_interventions": int(method.safety_reflex_enabled),
        "reflex_backups": 0,
        "mean_optimal_decay": 0.99 if method.optimal_decay_weight else 1.0,
        "minimum_optimal_decay": 0.95 if method.optimal_decay_weight else 1.0,
    }


def test_summary_uses_paired_differences_and_holm_adjustment(tmp_path):
    rows = []
    for method in METHODS:
        rows.extend([_row(method, 0, True), _row(method, 1, method.online_feedback)])
    config = PairedBenchmarkConfig(
        episodes=2,
        bootstrap_resamples=100,
        workers=1,
        output_dir=str(tmp_path),
    )
    summary = summarize_paired_rows(rows, config)
    online = summary["robust_stack_async_feedback"]
    fixed = summary["robust_stack_fixed_g015"]
    assert online["success_rate"] == 1.0
    assert fixed["success_rate"] == 0.5
    assert online["paired_success_difference"] == 0.5
    assert online["gamma_updates_applied"] == 2
    assert online["mcnemar_holm_adjusted_p"] >= online["mcnemar_exact_p"]


def test_timeout_is_joint_failure_even_if_stored_success_is_true(tmp_path):
    rows = []
    for method in METHODS:
        row = _row(method, 0, False)
        row["success"] = True
        row["joint_success"] = True
        rows.append(row)
    config = PairedBenchmarkConfig(
        stage="confirmatory", episodes=1, bootstrap_resamples=20,
        workers=1, output_dir=str(tmp_path),
    )
    summary = summarize_paired_rows(rows, config)
    online = summary["robust_stack_async_feedback"]
    assert online["success_rate"] == 0.0
    assert online["timeout_failures"] == 1
    assert online["stored_success_mismatches"] == 1


def test_confirmatory_efficacy_gate_requires_paired_superiority(tmp_path):
    rows = []
    for episode in range(20):
        for method in METHODS:
            success = method.name == "robust_stack_async_feedback"
            rows.append(_row(method, episode, success))
    config = PairedBenchmarkConfig(
        stage="confirmatory", episodes=20, bootstrap_resamples=200,
        workers=1, output_dir=str(tmp_path),
    )
    summaries = summarize_paired_rows(rows, config)
    gate = evaluate_confirmatory_efficacy_gate(summaries, config)
    assert gate["passed"] is True
    assert gate["paired_joint_success_difference"] == pytest.approx(1.0)
    assert gate["checks"]["bootstrap_95_ci_lower_above_margin"] is True


def test_development_reports_but_does_not_apply_efficacy_gate(tmp_path):
    rows = []
    for method in METHODS:
        rows.append(_row(method, 0, False))
    config = PairedBenchmarkConfig(
        stage="development", episodes=1, bootstrap_resamples=20,
        workers=1, output_dir=str(tmp_path),
    )
    gate = evaluate_confirmatory_efficacy_gate(
        summarize_paired_rows(rows, config), config
    )
    assert gate["evaluated"] is False
    assert gate["passed"] is None


def test_summary_rejects_unpaired_episode_ids(tmp_path):
    rows = []
    for method in METHODS:
        row = _row(method, 0, False)
        if method.name == "robust_stack_async_feedback":
            row["episode"] = 1
        rows.append(row)
    config = PairedBenchmarkConfig(
        episodes=1, bootstrap_resamples=20, workers=1, output_dir=str(tmp_path)
    )
    with pytest.raises(ValueError, match="not paired"):
        summarize_paired_rows(rows, config)
