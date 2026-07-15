import json
from dataclasses import replace

import pytest

from lampc_cbf.hf_llm import GammaDecision
from lampc_cbf.paper_manifest import PaperFidelityManifest
from lampc_cbf.paired_benchmark import (
    METHODS,
    PairedBenchmarkConfig,
    _conditions,
    configured_methods,
    evaluate_confirmatory_efficacy_gate,
    evaluate_formal_contract_gate,
    exact_mcnemar_pvalue,
    summarize_paired_rows,
    load_feedback_checkpoint,
    validate_feedback_decision_trace,
)


def test_default_protocol_has_500_common_conditions_and_isolates_feedback():
    config = PairedBenchmarkConfig()
    assert config.stage == "confirmatory"
    assert config.episodes == 500
    assert config.bootstrap_resamples == 10_000
    assert config.efficacy_method == "paper_async_feedback_static"
    assert config.efficacy_comparator == "fixed_cbf_static_g015"
    feedback = next(method for method in METHODS if method.name == config.efficacy_method)
    fixed = next(method for method in METHODS if method.name == feedback.comparator)
    assert feedback.gamma == fixed.gamma == pytest.approx(0.15)
    assert feedback.prediction_mode == fixed.prediction_mode == "static"
    assert feedback.safety_reflex_enabled == fixed.safety_reflex_enabled
    assert feedback.optimal_decay_weight == fixed.optimal_decay_weight
    assert feedback.feedback_trigger_mode == "elapsed_time"


def test_protocol_separates_paper_fidelity_from_robust_extension():
    paper = next(method for method in METHODS if method.name == "proactive_cbf_static_g002")
    robust = next(method for method in METHODS if method.name == "predictive_reflex_g002")
    assert paper.experiment_profile == "paper_fidelity"
    assert paper.delta_u_weight == pytest.approx(0.5)
    assert paper.reference_mode == "direct_target"
    assert robust.experiment_profile == "robust_extension"
    assert robust.delta_u_weight == pytest.approx(2.0)
    assert robust.reference_mode == "straight"
    formal = next(method for method in METHODS if method.name == "formal_stack_fixed_g015")
    assert formal.experiment_profile == "formal_extension"
    assert formal.formal_safety_filter_enabled
    assert formal.bounded_measurement_noise
    assert formal.known_obstacle_velocity


def test_paper_replication_stage_uses_manifest_locked_pair_and_scene():
    manifest = PaperFidelityManifest.load("configs/paper_fidelity.json")
    config = PairedBenchmarkConfig(**manifest.benchmark_kwargs())

    assert config.stage == "paper-replication"
    assert config.episodes == 50
    assert config.max_steps == 220
    assert config.feedback_schedule_mode == "elapsed_time"
    assert config.feedback_request_policy == "one_shot_per_feedback_episode"
    assert config.feedback_requests_per_episode == 1
    assert config.latency_trace_mode == "precollected_uncached_per_episode_replay"
    assert [method.name for method in configured_methods(config)] == [
        "fixed_cbf_static_g015",
        "paper_async_feedback_static",
    ]
    assert {condition["lateral_offset"] for condition in _conditions(config)} == {0.0}


def test_paper_replication_rejects_non_manifest_episode_override():
    manifest = PaperFidelityManifest.load("configs/paper_fidelity.json")
    kwargs = manifest.benchmark_kwargs() | {"episodes": 49}

    with pytest.raises(ValueError, match="exactly 50"):
        PairedBenchmarkConfig(**kwargs)


def _feedback_decision(index: int, *, cache_hit: bool = False) -> GammaDecision:
    return GammaDecision(
        gamma=0.05,
        safety_level=2,
        explanation="test",
        model="gpt-4o",
        provider="openai",
        latency_seconds=2.0 + index / 1000.0,
        requested_at_unix=1_700_000_000.0 + index,
        prompt_hash="prompt",
        request_hash="request",
        raw_response=None,
        fallback_used=False,
        cache_hit=cache_hit,
        error_type=None,
    )


def test_paper_feedback_trace_accepts_one_uncached_request_per_episode():
    manifest = PaperFidelityManifest.load("configs/paper_fidelity.json")
    config = PairedBenchmarkConfig(**manifest.benchmark_kwargs())
    decisions = [_feedback_decision(index) for index in range(config.episodes)]

    assert validate_feedback_decision_trace(decisions, config) == tuple(decisions)


def test_paper_feedback_trace_rejects_single_frozen_latency_sample():
    manifest = PaperFidelityManifest.load("configs/paper_fidelity.json")
    config = PairedBenchmarkConfig(**manifest.benchmark_kwargs())
    frozen = _feedback_decision(0)

    with pytest.raises(ValueError, match="single frozen latency sample"):
        validate_feedback_decision_trace([frozen] * config.episodes, config)


def test_paper_feedback_trace_rejects_cached_request():
    manifest = PaperFidelityManifest.load("configs/paper_fidelity.json")
    config = PairedBenchmarkConfig(**manifest.benchmark_kwargs())
    decisions = [_feedback_decision(index) for index in range(config.episodes)]
    decisions[17] = _feedback_decision(17, cache_hit=True)

    with pytest.raises(ValueError, match="uncached"):
        validate_feedback_decision_trace(decisions, config)


def test_feedback_checkpoint_resumes_only_validated_prefix(tmp_path):
    manifest = PaperFidelityManifest.load(
        "configs/paper_fidelity_nvidia_nim_llama31.json"
    )
    first = replace(_feedback_decision(0), model="meta/llama-3.1-8b-instruct", provider="nvidia-nim")
    second = replace(_feedback_decision(1), model="meta/llama-3.1-8b-instruct", provider="nvidia-nim")
    checkpoint = tmp_path / "feedback.json"
    checkpoint.write_text(json.dumps([first.as_dict(), second.as_dict()]), encoding="utf-8")

    resumed = load_feedback_checkpoint(
        checkpoint, episodes=50, paper_manifest=manifest
    )

    assert resumed == [first, second]


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
        "minimum_true_barrier": 0.001 if success else -0.001,
        "minimum_true_cbf_residual": 0.0001 if method.safety_mode == "cbf" else None,
        "true_cbf_violation_steps": 0,
        "formal_filter_interventions": 0,
        "formal_filter_uncertified_steps": 0,
        "formal_terminal_backup_uncertified_steps": 0,
        "minimum_robust_filter_residual": None,
        "minimum_backup_authority_margin": None,
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
        "formal_initial_safe": True,
        "formal_raw_discrete_cbf_satisfied": method.safety_mode == "cbf",
        "formal_exact_or_bounded_observation": False,
        "formal_applied_input_matches_mpc": not method.safety_reflex_enabled,
        "formal_final_input_certified": False,
        "formal_robust_filter_satisfied": False,
        "formal_model_match_verified": False,
        "formal_terminal_safe_set_or_backup_certified": False,
        "formal_stepwise_certificate_eligible": False,
        "formal_recursive_certificate_eligible": False,
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
            success = method.name == "paper_async_feedback_static"
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


def test_summary_does_not_promote_empirical_safety_to_formal_certificate(tmp_path):
    rows = [_row(method, 0, True) for method in METHODS]
    config = PairedBenchmarkConfig(
        episodes=1, bootstrap_resamples=20, workers=1, output_dir=str(tmp_path)
    )
    summary = summarize_paired_rows(rows, config)
    audit = summary["fixed_cbf_static_g015"]["formal_scope_audit"]
    assert audit["initial_safe_episodes"] == 1
    assert audit["raw_discrete_cbf_satisfied_episodes"] == 1
    assert audit["stepwise_certificate_eligible_episodes"] == 0
    assert audit["recursive_certificate_eligible_episodes"] == 0


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


def test_paper_replication_summary_excludes_extensions_and_applies_gate(tmp_path):
    manifest = PaperFidelityManifest.load("configs/paper_fidelity.json")
    config = PairedBenchmarkConfig(
        **(manifest.benchmark_kwargs() | {
            "output_dir": str(tmp_path),
        })
    )
    methods = configured_methods(config)
    rows = []
    for episode in range(config.episodes):
        for method in methods:
            rows.append(
                _row(
                    method,
                    episode,
                    method.name == "paper_async_feedback_static",
                )
            )

    summaries = summarize_paired_rows(rows, config)
    gate = evaluate_confirmatory_efficacy_gate(summaries, config)

    assert set(summaries) == {
        "fixed_cbf_static_g015",
        "paper_async_feedback_static",
    }
    assert gate["evaluated"] is True
    assert gate["passed"] is True


def test_formal_contract_gate_requires_every_episode_to_be_recursive(tmp_path):
    rows = []
    for method in METHODS:
        row = _row(method, 0, True)
        if method.experiment_profile == "formal_extension":
            row.update(
                formal_final_input_certified=True,
                formal_robust_filter_satisfied=True,
                formal_model_match_verified=True,
                formal_exact_or_bounded_observation=True,
                formal_terminal_safe_set_or_backup_certified=True,
                formal_stepwise_certificate_eligible=True,
                formal_recursive_certificate_eligible=True,
                minimum_robust_filter_residual=0.0,
                minimum_backup_authority_margin=0.01,
            )
        rows.append(row)
    config = PairedBenchmarkConfig(
        stage="development", episodes=1, bootstrap_resamples=20,
        workers=1, output_dir=str(tmp_path),
    )
    gate = evaluate_formal_contract_gate(
        summarize_paired_rows(rows, config), config
    )
    assert gate["evaluated"] is True
    assert gate["passed"] is True
    assert gate["whole_body_panda_certified"] is False


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
