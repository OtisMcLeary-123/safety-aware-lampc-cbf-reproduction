"""Large paired Safe Panda Gym benchmark for the complete safety stack."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import csv
import json
from math import comb, isfinite
import multiprocessing
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hf_llm import GammaDecision


@dataclass(frozen=True, slots=True)
class PairedBenchmarkConfig:
    episodes: int = 500
    seed: int = 20260713
    bootstrap_resamples: int = 10_000
    max_steps: int = 140
    workers: int = 4
    speed_lower: float = 0.025
    speed_upper: float = 0.20
    lateral_lower: float = 0.045
    lateral_upper: float = 0.080
    intervention_time_lower: float = 0.0
    intervention_time_upper: float = 0.40
    output_dir: str = "artifacts/paired_benchmark_protocol_v2_500"
    resume: bool = True
    feedback_schedule_mode: str = "ttc"
    feedback_ttc_threshold: float = 1.5

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.bootstrap_resamples < 1:
            raise ValueError("episode and bootstrap counts must be positive")
        if self.max_steps < 4 or self.workers < 1:
            raise ValueError("max_steps >= 4 and workers >= 1 are required")
        if not 0.0 < self.speed_lower < self.speed_upper:
            raise ValueError("invalid obstacle speed interval")
        if not 0.0 <= self.lateral_lower < self.lateral_upper:
            raise ValueError("invalid lateral-offset interval")
        if not 0.0 <= self.intervention_time_lower <= self.intervention_time_upper:
            raise ValueError("invalid intervention-time interval")
        if self.feedback_schedule_mode not in {"elapsed_time", "ttc"}:
            raise ValueError("feedback_schedule_mode must be elapsed_time or ttc")
        if self.feedback_ttc_threshold <= 0.0:
            raise ValueError("feedback_ttc_threshold must be positive")


@dataclass(frozen=True, slots=True)
class BenchmarkMethod:
    name: str
    safety_mode: str
    gamma: float
    prediction_mode: str
    safety_reflex_enabled: bool
    optimal_decay_weight: float = 0.0
    online_feedback: bool = False
    comparator: str = "fixed_cbf_static_g015"
    experiment_profile: str = "paper_fidelity"
    delta_u_weight: float = 0.5
    reference_mode: str = "direct_target"
    provisional_local_feedback: bool = False


METHODS: tuple[BenchmarkMethod, ...] = (
    BenchmarkMethod("distance_static", "distance", 0.15, "static", False),
    BenchmarkMethod("fixed_cbf_static_g015", "cbf", 0.15, "static", False),
    BenchmarkMethod("proactive_cbf_static_g002", "cbf", 0.02, "static", False),
    BenchmarkMethod(
        "robust_static_g002", "cbf", 0.02, "static", False,
        comparator="robust_static_g002",
        experiment_profile="robust_extension", delta_u_weight=2.0,
        reference_mode="straight",
    ),
    BenchmarkMethod(
        "predictive_velocity_g002", "cbf", 0.02, "velocity", False,
        comparator="robust_static_g002",
        experiment_profile="robust_extension", delta_u_weight=2.0,
        reference_mode="straight",
    ),
    BenchmarkMethod(
        "predictive_cbf_g002", "cbf", 0.02, "velocity_tube", False,
        comparator="predictive_velocity_g002",
        experiment_profile="robust_extension", delta_u_weight=2.0,
        reference_mode="straight",
    ),
    BenchmarkMethod(
        "predictive_reflex_g002", "cbf", 0.02, "velocity_tube", True,
        comparator="predictive_cbf_g002",
        experiment_profile="robust_extension", delta_u_weight=2.0,
        reference_mode="straight",
    ),
    BenchmarkMethod(
        "optimal_decay_predictive_g002", "cbf", 0.02, "velocity_tube", False, 10.0,
        comparator="predictive_cbf_g002",
        experiment_profile="robust_extension", delta_u_weight=2.0,
        reference_mode="straight",
    ),
    BenchmarkMethod(
        "robust_stack_fixed_g015", "cbf", 0.15, "velocity_tube", True, 10.0,
        comparator="robust_stack_fixed_g015",
        experiment_profile="robust_extension", delta_u_weight=2.0,
        reference_mode="straight",
    ),
    BenchmarkMethod(
        "robust_stack_async_feedback", "cbf", 0.15, "velocity_tube", True, 10.0,
        online_feedback=True, comparator="robust_stack_fixed_g015",
        experiment_profile="robust_extension", delta_u_weight=2.0,
        reference_mode="straight", provisional_local_feedback=True,
    ),
)


def exact_mcnemar_pvalue(baseline: Sequence[bool], method: Sequence[bool]) -> float:
    """Two-sided exact McNemar test using the binomial distribution."""

    if len(baseline) != len(method) or not baseline:
        raise ValueError("paired outcomes must be non-empty and equal length")
    baseline_only = sum(a and not b for a, b in zip(baseline, method))
    method_only = sum(b and not a for a, b in zip(baseline, method))
    discordant = baseline_only + method_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(baseline_only, method_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _bootstrap_interval(values: Any, resamples: int, rng: Any) -> tuple[float, float]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(resamples, len(array)), replace=True)
    lower, upper = np.quantile(np.mean(samples, axis=1), [0.025, 0.975])
    return float(lower), float(upper)


def _holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (count - rank) * pvalues[name])
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def _conditions(config: PairedBenchmarkConfig) -> list[dict[str, Any]]:
    import numpy as np

    rng = np.random.default_rng(config.seed)
    speeds = rng.uniform(config.speed_lower, config.speed_upper, config.episodes)
    lateral = rng.uniform(config.lateral_lower, config.lateral_upper, config.episodes)
    signs = rng.choice((-1.0, 1.0), config.episodes)
    interventions = rng.uniform(
        config.intervention_time_lower,
        config.intervention_time_upper,
        config.episodes,
    )
    return [
        {
            "episode": index,
            "seed": 100_000 + index,
            "obstacle_speed": float(speeds[index]),
            "lateral_offset": float(signs[index] * lateral[index]),
            "intervention_time": float(interventions[index]),
        }
        for index in range(config.episodes)
    ]


def _run_paired_condition(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    from .smooth_dynamic_demo import SmoothDynamicConfig, run_smooth_dynamic_demo

    condition = payload["condition"]
    max_steps = int(payload["max_steps"])
    feedback_gamma = float(payload["feedback_gamma"])
    feedback_latency = float(payload["feedback_latency"])
    feedback_schedule_mode = str(payload.get("feedback_schedule_mode", "ttc"))
    feedback_ttc_threshold = float(payload.get("feedback_ttc_threshold", 1.5))
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        schedule: tuple[tuple[float, float], ...] = ()
        if method.online_feedback:
            if feedback_schedule_mode == "elapsed_time":
                schedule = ((condition["intervention_time"] + feedback_latency, feedback_gamma),)
        config = SmoothDynamicConfig(
            delta_u_weight=method.delta_u_weight,
            gamma=method.gamma,
            seed=int(condition["seed"]),
            max_steps=max_steps,
            reference_speed=0.08,
            reference_mode=method.reference_mode,
            safety_mode=method.safety_mode,
            prediction_mode=method.prediction_mode,
            safety_reflex_enabled=method.safety_reflex_enabled,
            optimal_decay_weight=method.optimal_decay_weight,
            gamma_schedule=schedule,
            provisional_feedback_times=(
                (condition["intervention_time"],)
                if method.provisional_local_feedback
                and feedback_schedule_mode == "elapsed_time"
                else ()
            ),
            feedback_ttc_threshold=(
                feedback_ttc_threshold
                if method.online_feedback and feedback_schedule_mode == "ttc"
                else None
            ),
            feedback_response_latency=feedback_latency,
            feedback_gamma=(
                feedback_gamma
                if method.online_feedback and feedback_schedule_mode == "ttc"
                else None
            ),
            gamma_update_ttl=1.0,
            obstacle_start_offset=(condition["lateral_offset"], 0.44, 0.0),
            obstacle_velocity=(0.0, -condition["obstacle_speed"], 0.0),
            save_animation=False,
            save_plots=False,
            save_metrics=False,
            output_dir=f"/tmp/lampc-paired-{os.getpid()}",
        )
        result = run_smooth_dynamic_demo(config)
        rows.append(
            {
                **condition,
                "method": method.name,
                "comparator": method.comparator,
                "experiment_profile": method.experiment_profile,
                "outcome": result.outcome,
                "success": bool(result.reached_goal and not result.collision),
                "reached_goal": result.reached_goal,
                "collision": result.collision,
                "steps": result.steps,
                "minimum_true_clearance": result.minimum_true_clearance,
                "minimum_measured_clearance": result.minimum_measured_clearance,
                "final_goal_distance": result.final_goal_distance,
                "mean_solve_time": result.mean_solve_time,
                "max_solve_time": result.max_solve_time,
                "p99_solve_time": result.p99_solve_time,
                "final_gamma": result.final_gamma,
                "gamma_updates_applied": result.gamma_updates_applied,
                "gamma_updates_rejected": result.gamma_updates_rejected,
                "reflex_interventions": result.reflex_interventions,
                "reflex_backups": result.reflex_backups,
                "mean_optimal_decay": result.mean_optimal_decay,
                "minimum_optimal_decay": result.minimum_optimal_decay,
                "avoidance_onset_time": result.avoidance_onset_time,
                "minimum_predicted_ttc": result.minimum_predicted_ttc,
                "feedback_latency": feedback_latency if method.online_feedback else 0.0,
                "feedback_schedule_mode": feedback_schedule_mode,
                "feedback_trigger_time": result.feedback_trigger_time,
                "feedback_available_time": result.feedback_available_time,
                "feedback_causal_opportunity": result.feedback_causal_opportunity,
                "feedback_updates_rejected_late": result.feedback_updates_rejected_late,
                "solver_failures": result.solver_failures,
                "solver_rejections": result.solver_rejections,
                "solver_max_cpu_time_exits": result.solver_max_cpu_time_exits,
                "solver_infeasible_exits": result.solver_infeasible_exits,
                "solver_unknown_exits": result.solver_unknown_exits,
                "deadline_misses": result.deadline_misses,
                "emergency_fallbacks": result.emergency_fallbacks,
                "maximum_constraint_violation": result.maximum_constraint_violation,
                "most_infeasible_stage": result.most_infeasible_stage,
                "infeasible_stage_events": result.infeasible_stage_events,
                "mean_model_transition_error": result.mean_model_transition_error,
                "max_model_transition_error": result.max_model_transition_error,
                "mean_action_tracking_error": result.mean_action_tracking_error,
                "max_action_tracking_error": result.max_action_tracking_error,
                **result.smoothness.as_dict(),
            }
        )
    return rows


def _read_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    typed: list[dict[str, Any]] = []
    boolean_fields = {
        "success", "reached_goal", "collision", "feedback_causal_opportunity"
    }
    integer_fields = {
        "episode", "seed", "steps", "gamma_updates_applied", "gamma_updates_rejected",
        "reflex_interventions", "reflex_backups",
        "solver_failures", "solver_rejections", "deadline_misses",
        "solver_max_cpu_time_exits", "solver_infeasible_exits",
        "solver_unknown_exits",
        "emergency_fallbacks", "feedback_updates_rejected_late",
        "most_infeasible_stage", "infeasible_stage_events",
    }
    text_fields = {
        "method", "comparator", "experiment_profile", "outcome",
        "feedback_schedule_mode",
    }
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if key in text_fields:
                parsed[key] = value
            elif key in boolean_fields:
                parsed[key] = value == "True"
            elif key in integer_fields:
                parsed[key] = int(value)
            else:
                parsed[key] = None if value in {"", "None"} else float(value)
        typed.append(parsed)
    return typed


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    ordered = sorted(rows, key=lambda row: (int(row["episode"]), str(row["method"])))
    fieldnames = list(dict.fromkeys(key for row in ordered for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)


def summarize_paired_rows(
    rows: Sequence[Mapping[str, Any]], config: PairedBenchmarkConfig
) -> dict[str, Any]:
    import numpy as np

    expected = {method.name for method in METHODS}
    present = {str(row["method"]) for row in rows}
    if present != expected:
        raise ValueError("rows do not contain exactly the configured methods")
    rng = np.random.default_rng(config.seed + 1)
    summaries: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    by_method = {
        method.name: sorted(
            (row for row in rows if row["method"] == method.name),
            key=lambda row: int(row["episode"]),
        )
        for method in METHODS
    }
    for method in METHODS:
        method_rows = by_method[method.name]
        if len(method_rows) != config.episodes:
            raise ValueError(f"method {method.name} does not have {config.episodes} episodes")
        comparator_rows = by_method[method.comparator]
        outcomes = np.asarray([bool(row["success"]) for row in method_rows], dtype=float)
        comparator = np.asarray([bool(row["success"]) for row in comparator_rows], dtype=float)
        paired_difference = outcomes - comparator
        success_ci = _bootstrap_interval(outcomes, config.bootstrap_resamples, rng)
        difference_ci = _bootstrap_interval(
            paired_difference, config.bootstrap_resamples, rng
        )
        clearance = np.asarray(
            [float(row["minimum_true_clearance"]) for row in method_rows]
        )
        comparator_clearance = np.asarray(
            [float(row["minimum_true_clearance"]) for row in comparator_rows]
        )
        clearance_difference_ci = _bootstrap_interval(
            clearance - comparator_clearance, config.bootstrap_resamples, rng
        )
        pvalue = exact_mcnemar_pvalue(
            [bool(row["success"]) for row in comparator_rows],
            [bool(row["success"]) for row in method_rows],
        )
        raw_pvalues[method.name] = pvalue
        summaries[method.name] = {
            "episodes": len(method_rows),
            "successes": int(np.sum(outcomes)),
            "success_rate": float(np.mean(outcomes)),
            "success_bootstrap_95_ci": list(success_ci),
            "comparator": method.comparator,
            "paired_success_difference": float(np.mean(paired_difference)),
            "paired_success_difference_95_ci": list(difference_ci),
            "mcnemar_exact_p": pvalue,
            "collisions": sum(bool(row["collision"]) for row in method_rows),
            "outcomes": {
                outcome: sum(row.get("outcome") == outcome for row in method_rows)
                for outcome in (
                    "goal", "collision", "timeout", "environment_truncated",
                    "infeasible_abort", "emergency_fallback",
                )
            },
            "mean_minimum_true_clearance": float(np.mean(clearance)),
            "paired_clearance_difference_95_ci": list(clearance_difference_ci),
            "median_avoidance_onset_time": float(np.median([
                row["avoidance_onset_time"] for row in method_rows
                if row["avoidance_onset_time"] is not None
            ])) if any(row["avoidance_onset_time"] is not None for row in method_rows) else None,
            "median_minimum_predicted_ttc": float(np.median([
                row["minimum_predicted_ttc"] for row in method_rows
                if row["minimum_predicted_ttc"] is not None
            ])) if any(row["minimum_predicted_ttc"] is not None for row in method_rows) else None,
            "mean_path_length": float(np.mean([row["path_length"] for row in method_rows])),
            "mean_jerk_rms": float(np.mean([row["jerk_rms"] for row in method_rows])),
            "mean_solve_time": float(np.mean([row["mean_solve_time"] for row in method_rows])),
            "max_solve_time": float(np.max([row["max_solve_time"] for row in method_rows])),
            "maximum_episode_p99_solve_time": float(np.max([
                row.get("p99_solve_time", row["max_solve_time"])
                for row in method_rows
            ])),
            "gamma_updates_applied": sum(int(row["gamma_updates_applied"]) for row in method_rows),
            "gamma_updates_rejected": sum(int(row["gamma_updates_rejected"]) for row in method_rows),
            "reflex_interventions": sum(int(row["reflex_interventions"]) for row in method_rows),
            "reflex_backups": sum(int(row["reflex_backups"]) for row in method_rows),
            "mean_optimal_decay": float(np.mean([row["mean_optimal_decay"] for row in method_rows])),
            "minimum_optimal_decay": float(np.min([row["minimum_optimal_decay"] for row in method_rows])),
            "solver_failures": sum(int(row.get("solver_failures", 0)) for row in method_rows),
            "solver_rejections": sum(int(row.get("solver_rejections", 0)) for row in method_rows),
            "solver_max_cpu_time_exits": sum(
                int(row.get("solver_max_cpu_time_exits", 0))
                for row in method_rows
            ),
            "solver_infeasible_exits": sum(
                int(row.get("solver_infeasible_exits", 0))
                for row in method_rows
            ),
            "solver_unknown_exits": sum(
                int(row.get("solver_unknown_exits", 0))
                for row in method_rows
            ),
            "deadline_misses": sum(int(row.get("deadline_misses", 0)) for row in method_rows),
            "emergency_fallbacks": sum(int(row.get("emergency_fallbacks", 0)) for row in method_rows),
            "feedback_causal_opportunities": sum(
                bool(row.get("feedback_causal_opportunity", False))
                for row in method_rows
            ),
            "feedback_updates_with_causal_opportunity": sum(
                bool(row.get("feedback_causal_opportunity", False))
                and int(row.get("gamma_updates_applied", 0)) > 0
                for row in method_rows
            ),
            "feedback_updates_rejected_late": sum(
                int(row.get("feedback_updates_rejected_late", 0))
                for row in method_rows
            ),
            "infeasible_stage_events": sum(
                int(row.get("infeasible_stage_events", 0))
                for row in method_rows
            ),
            "experiment_profile": method.experiment_profile,
        }
    adjusted = _holm_adjust(raw_pvalues)
    for name, value in adjusted.items():
        summaries[name]["mcnemar_holm_adjusted_p"] = value
    return summaries


def _plot_summary(rows: Sequence[Mapping[str, Any]], summaries: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [method.name for method in METHODS]
    rates = [summaries[name]["success_rate"] for name in labels]
    lower = [rates[i] - summaries[name]["success_bootstrap_95_ci"][0] for i, name in enumerate(labels)]
    upper = [summaries[name]["success_bootstrap_95_ci"][1] - rates[i] for i, name in enumerate(labels)]
    clearances = [
        [1000.0 * float(row["minimum_true_clearance"]) for row in rows if row["method"] == name]
        for name in labels
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    axes[0].bar(range(len(labels)), rates, yerr=[lower, upper], capsize=4)
    axes[0].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("collision-free goal success")
    axes[0].set_title(f"{len(rows) // len(labels)} paired conditions; bootstrap 95% CI")
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].boxplot(clearances, tick_labels=labels, showfliers=False)
    axes[1].axhline(0.0, color="red", linestyle="--", label="collision boundary")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylabel("raw minimum true clearance [mm]")
    axes[1].set_title("Safety metric always uses raw trajectory")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_paired_benchmark(
    feedback_decision: GammaDecision,
    config: PairedBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run or resume all methods on identical Safe Panda Gym conditions."""

    cfg = config or PairedBenchmarkConfig()
    if feedback_decision.fallback_used:
        raise ValueError("benchmark requires a validated non-fallback feedback decision")
    if not isfinite(feedback_decision.latency_seconds) or feedback_decision.latency_seconds < 0.0:
        raise ValueError("feedback latency must be finite and non-negative")
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "episodes.csv"
    rows = _read_checkpoint(csv_path) if cfg.resume else []
    complete = {
        int(episode)
        for episode in {row["episode"] for row in rows}
        if sum(int(row["episode"]) == int(episode) for row in rows) == len(METHODS)
    }
    conditions = [item for item in _conditions(cfg) if item["episode"] not in complete]
    print(
        f"[paired] {len(complete)}/{cfg.episodes} conditions restored; "
        f"{len(conditions)} pending across {cfg.workers} workers",
        flush=True,
    )
    payloads = [
        {
            "condition": condition,
            "max_steps": cfg.max_steps,
            "feedback_gamma": feedback_decision.gamma,
            "feedback_latency": feedback_decision.latency_seconds,
            "feedback_schedule_mode": cfg.feedback_schedule_mode,
            "feedback_ttc_threshold": cfg.feedback_ttc_threshold,
        }
        for condition in conditions
    ]
    if cfg.workers == 1:
        for index, payload in enumerate(payloads, 1):
            rows.extend(_run_paired_condition(payload))
            _write_rows(csv_path, rows)
            if index % 10 == 0 or index == len(payloads):
                print(f"[paired] completed {len(complete) + index}/{cfg.episodes}", flush=True)
    elif payloads:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=context) as executor:
            futures = [executor.submit(_run_paired_condition, payload) for payload in payloads]
            for index, future in enumerate(as_completed(futures), 1):
                # Deliberately no automatic retry: any worker failure aborts and
                # leaves completed conditions in the resumable checkpoint.
                rows.extend(future.result())
                _write_rows(csv_path, rows)
                if index % 10 == 0 or index == len(futures):
                    print(f"[paired] completed {len(complete) + index}/{cfg.episodes}", flush=True)
    summaries = summarize_paired_rows(rows, cfg)
    summary = {
        "study_id": f"safe-panda-paired-protocol-v2-{cfg.episodes}",
        "protocol_version": 2,
        "config": asdict(cfg),
        "methods": summaries,
        "feedback_decision": feedback_decision.as_dict(),
        "paired_randomization": True,
        "common_random_numbers": True,
        "simulator": "PandaReachSafe-v3 / PyBullet Tiny / do-mpc / CasADi / IPOPT",
        "statistical_tests": {
            "success_rate": "non-parametric bootstrap 95% CI",
            "paired_success": "paired bootstrap difference + exact McNemar",
            "multiple_comparisons": "Holm adjustment",
            "safety_source": "raw end-effector and true obstacle trajectory",
        },
        "formal_scope": (
            "Monte Carlo evidence only; Gaussian noise is not deterministically bounded "
            "and operational-space reflex does not certify whole-body Panda safety."
        ),
    }
    (root / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot_summary(rows, summaries, root / "paired_success_and_clearance.png")
    print("[paired] complete", flush=True)
    return summary
