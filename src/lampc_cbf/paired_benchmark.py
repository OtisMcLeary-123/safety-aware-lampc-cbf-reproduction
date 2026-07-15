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
from .paper_manifest import PaperFidelityManifest


@dataclass(frozen=True, slots=True)
class PairedBenchmarkConfig:
    stage: str = "confirmatory"
    episodes: int = 500
    seed: int = 20260713
    bootstrap_resamples: int = 10_000
    max_steps: int = 220
    workers: int = 4
    speed_lower: float = 0.025
    speed_upper: float = 0.20
    lateral_lower: float = 0.045
    lateral_upper: float = 0.080
    intervention_time_lower: float = 0.0
    intervention_time_upper: float = 0.40
    output_dir: str = "artifacts/paired_benchmark_protocol_v5_500"
    resume: bool = True
    feedback_schedule_mode: str = "ttc"
    feedback_request_policy: str = "one_shot_per_feedback_episode"
    feedback_requests_per_episode: int = 1
    latency_trace_mode: str = "precollected_uncached_per_episode_replay"
    feedback_ttc_threshold: float = 1.5
    efficacy_method: str = "paper_async_feedback_static"
    efficacy_comparator: str = "fixed_cbf_static_g015"
    efficacy_alpha: float = 0.05
    efficacy_minimum_paired_difference: float = 0.0
    method_names: tuple[str, ...] = ()
    fixed_lateral_offset: float | None = None
    sensor_period: float = 0.67
    measurement_noise_sigma: float = 0.005
    reference_speed: float = 0.08
    cbf_transition_mode: str = "command_velocity"
    manifest_path: str | None = None
    manifest_hash: str | None = None
    manifest_profile: str | None = None
    model_substitution: bool = False
    goal_offset: tuple[float, float, float] = (0.0, 0.30, 0.0)
    obstacle_start_forward_offset: float = 0.44
    obstacle_start_vertical_offset: float = 0.0
    obstacle_radius: float = 0.10
    collision_radius: float = 0.035
    gamma_update_ttl: float = 1.0
    solver_max_constraint_violation: float = 1e-6
    solver_max_cpu_time: float = 0.035
    control_deadline: float = 0.04
    reject_deadline_miss: bool = False
    initial_query: str = "Move gripper to red cube."
    feedback_prompt: str = (
        "Watch out! I think the robot is going to crash soon. Increase clearance now."
    )

    def __post_init__(self) -> None:
        if self.stage not in {
            "smoke", "development", "paper-replication", "confirmatory",
        }:
            raise ValueError(
                "stage must be smoke, development, paper-replication, or confirmatory"
            )
        if self.episodes < 1 or self.bootstrap_resamples < 1:
            raise ValueError("episode and bootstrap counts must be positive")
        if self.max_steps < 4 or self.workers < 1:
            raise ValueError("max_steps >= 4 and workers >= 1 are required")
        if not 0.0 < self.speed_lower < self.speed_upper:
            raise ValueError("invalid obstacle speed interval")
        if not 0.0 <= self.lateral_lower < self.lateral_upper:
            raise ValueError("invalid lateral-offset interval")
        if self.fixed_lateral_offset is not None and not isfinite(
            self.fixed_lateral_offset
        ):
            raise ValueError("fixed lateral offset must be finite")
        if not 0.0 <= self.intervention_time_lower <= self.intervention_time_upper:
            raise ValueError("invalid intervention-time interval")
        if self.feedback_schedule_mode not in {"elapsed_time", "ttc"}:
            raise ValueError("feedback_schedule_mode must be elapsed_time or ttc")
        if self.feedback_requests_per_episode < 1:
            raise ValueError("feedback requests per episode must be positive")
        if self.feedback_ttc_threshold <= 0.0:
            raise ValueError("feedback_ttc_threshold must be positive")
        if not 0.0 < self.efficacy_alpha < 1.0:
            raise ValueError("efficacy_alpha must be in (0, 1)")
        if not -1.0 <= self.efficacy_minimum_paired_difference <= 1.0:
            raise ValueError("efficacy minimum paired difference must be in [-1, 1]")
        if self.sensor_period <= 0.0 or self.measurement_noise_sigma < 0.0:
            raise ValueError("sensor period must be positive and noise non-negative")
        if self.reference_speed <= 0.0:
            raise ValueError("reference speed must be positive")
        if len(self.goal_offset) != 3:
            raise ValueError("goal offset must contain three values")
        if self.obstacle_start_forward_offset <= 0.0:
            raise ValueError("obstacle forward offset must be positive")
        if self.obstacle_radius <= 0.0 or self.collision_radius <= 0.0:
            raise ValueError("obstacle and collision radii must be positive")
        if self.gamma_update_ttl <= 0.0:
            raise ValueError("gamma update TTL must be positive")
        if any(
            value <= 0.0
            for value in (
                self.solver_max_constraint_violation,
                self.solver_max_cpu_time,
                self.control_deadline,
            )
        ):
            raise ValueError("solver tolerances and deadlines must be positive")
        if not self.initial_query.strip() or not self.feedback_prompt.strip():
            raise ValueError("paper queries must be non-empty")
        if self.cbf_transition_mode not in {"paper_state", "command_velocity"}:
            raise ValueError("invalid CBF transition mode")
        if len(set(self.method_names)) != len(self.method_names):
            raise ValueError("method names must be unique")
        if self.stage == "paper-replication":
            required_methods = (
                "fixed_cbf_static_g015",
                "paper_async_feedback_static",
            )
            if self.episodes != 50:
                raise ValueError("paper-replication requires exactly 50 episodes")
            if self.feedback_schedule_mode != "elapsed_time":
                raise ValueError("paper-replication requires elapsed-time feedback")
            if self.feedback_request_policy != "one_shot_per_feedback_episode":
                raise ValueError(
                    "paper-replication requires one-shot feedback per feedback episode"
                )
            if self.feedback_requests_per_episode != 1:
                raise ValueError(
                    "paper-replication requires exactly one feedback request per episode"
                )
            if self.latency_trace_mode != (
                "precollected_uncached_per_episode_replay"
            ):
                raise ValueError(
                    "paper-replication requires the frozen per-episode latency trace mode"
                )
            if self.method_names != required_methods:
                raise ValueError(
                    "paper-replication requires the frozen paper-fidelity method pair"
                )
            if self.fixed_lateral_offset is None:
                raise ValueError("paper-replication requires a fixed scene offset")
            if self.manifest_path is None or self.manifest_hash is None:
                raise ValueError("paper-replication requires a validated manifest")
            if self.manifest_profile not in {
                "paper_fidelity", "paper_fidelity_model_substitution",
            }:
                raise ValueError("paper-replication requires a known manifest profile")
            if len(self.manifest_hash) != 64 or any(
                character not in "0123456789abcdef"
                for character in self.manifest_hash
            ):
                raise ValueError("manifest hash must be a lowercase SHA-256 digest")


def validate_feedback_decision_trace(
    feedback_decisions: Sequence[GammaDecision],
    config: PairedBenchmarkConfig,
) -> tuple[GammaDecision, ...]:
    """Validate that every episode has its own usable feedback request record."""

    decisions = tuple(feedback_decisions)
    expected_count = config.episodes * config.feedback_requests_per_episode
    if len(decisions) != expected_count:
        raise ValueError(
            "benchmark requires exactly one feedback decision per episode"
            if config.feedback_requests_per_episode == 1
            else f"benchmark requires exactly {expected_count} feedback decisions"
        )
    if any(decision.fallback_used for decision in decisions):
        raise ValueError("benchmark requires validated non-fallback feedback decisions")
    if any(
        not isfinite(decision.latency_seconds) or decision.latency_seconds < 0.0
        for decision in decisions
    ):
        raise ValueError("feedback latencies must be finite and non-negative")
    if config.stage == "paper-replication":
        if any(decision.cache_hit for decision in decisions):
            raise ValueError(
                "paper-replication requires one uncached feedback request per episode"
            )
        request_times = [decision.requested_at_unix for decision in decisions]
        if any(not isfinite(timestamp) for timestamp in request_times):
            raise ValueError("feedback request timestamps must be finite")
        if len(set(request_times)) != config.episodes:
            raise ValueError(
                "paper-replication rejects a single frozen latency sample; "
                "provide one distinct request record per episode"
            )
    return decisions


def load_feedback_checkpoint(
    path: str | Path,
    *,
    episodes: int,
    paper_manifest: PaperFidelityManifest | None,
) -> list[GammaDecision]:
    """Load a validated provider-decision prefix without repeating requests."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("feedback checkpoint must contain a JSON list")
    if len(payload) > episodes:
        raise ValueError("feedback checkpoint contains more decisions than episodes")
    decisions: list[GammaDecision] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("feedback decision entries must be JSON objects")
        decisions.append(
            GammaDecision(raw_response=None, **item)
            if "raw_response" not in item
            else GammaDecision(**item)
        )
    if any(decision.fallback_used or decision.cache_hit for decision in decisions):
        raise ValueError("feedback checkpoint contains fallback or cached decisions")
    request_times = [decision.requested_at_unix for decision in decisions]
    if len(set(request_times)) != len(request_times):
        raise ValueError("feedback checkpoint request timestamps must be distinct")
    if paper_manifest is not None and any(
        not paper_manifest.accepts_feedback_decision(
            model=decision.model,
            provider=decision.provider,
            cache_hit=decision.cache_hit,
        )
        for decision in decisions
    ):
        raise ValueError("feedback checkpoint does not match the paper manifest")
    return decisions


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
    feedback_trigger_mode: str = "configured"
    formal_safety_filter_enabled: bool = False
    bounded_measurement_noise: bool = False
    known_obstacle_velocity: bool = False


METHODS: tuple[BenchmarkMethod, ...] = (
    BenchmarkMethod("distance_static", "distance", 0.15, "static", False),
    BenchmarkMethod("fixed_cbf_static_g015", "cbf", 0.15, "static", False),
    BenchmarkMethod("proactive_cbf_static_g002", "cbf", 0.02, "static", False),
    BenchmarkMethod(
        "paper_async_feedback_static", "cbf", 0.15, "static", False,
        online_feedback=True, comparator="fixed_cbf_static_g015",
        feedback_trigger_mode="elapsed_time",
    ),
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
    BenchmarkMethod(
        "formal_stack_fixed_g015", "cbf", 0.15, "velocity_tube", False, 10.0,
        comparator="formal_stack_fixed_g015",
        experiment_profile="formal_extension", delta_u_weight=2.0,
        reference_mode="straight", formal_safety_filter_enabled=True,
        bounded_measurement_noise=True, known_obstacle_velocity=True,
    ),
    BenchmarkMethod(
        "formal_stack_async_feedback", "cbf", 0.15, "velocity_tube", False, 10.0,
        online_feedback=True, comparator="formal_stack_fixed_g015",
        experiment_profile="formal_extension", delta_u_weight=2.0,
        reference_mode="straight", provisional_local_feedback=True,
        feedback_trigger_mode="elapsed_time", formal_safety_filter_enabled=True,
        bounded_measurement_noise=True, known_obstacle_velocity=True,
    ),
)


def configured_methods(config: PairedBenchmarkConfig) -> tuple[BenchmarkMethod, ...]:
    """Return the frozen method set selected for this benchmark stage."""

    by_name = {method.name: method for method in METHODS}
    names = config.method_names or tuple(by_name)
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"unknown benchmark methods: {unknown}")
    selected = tuple(by_name[name] for name in names)
    selected_names = {method.name for method in selected}
    missing_comparators = sorted({
        method.comparator
        for method in selected
        if method.comparator not in selected_names
    })
    if missing_comparators:
        raise ValueError(
            "selected methods omit required comparators: "
            + ", ".join(missing_comparators)
        )
    return selected


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


def _joint_success(row: Mapping[str, Any]) -> bool:
    """Paper endpoint: reach the goal without collision; every timeout is failure."""

    outcome = str(row.get("outcome", ""))
    if outcome != "goal":
        return False
    return bool(row.get("reached_goal", False)) and not bool(row.get("collision", False))


def _pairing_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return every available common-random-number condition identifier."""

    fields = (
        "episode", "seed", "obstacle_speed", "lateral_offset", "intervention_time"
    )
    return tuple((field, row[field]) for field in fields if field in row)


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
    if config.fixed_lateral_offset is None:
        lateral = rng.uniform(
            config.lateral_lower, config.lateral_upper, config.episodes
        )
        signs = rng.choice((-1.0, 1.0), config.episodes)
        lateral_offsets = signs * lateral
    else:
        lateral_offsets = np.full(config.episodes, config.fixed_lateral_offset)
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
            "lateral_offset": float(lateral_offsets[index]),
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
    protocol_version = int(payload.get("protocol_version", 5))
    benchmark_stage = str(payload.get("benchmark_stage", "confirmatory"))
    selected_names = tuple(str(name) for name in payload.get("method_names", ()))
    if selected_names:
        methods_by_name = {method.name: method for method in METHODS}
        try:
            selected_methods = tuple(methods_by_name[name] for name in selected_names)
        except KeyError as exc:
            raise ValueError(f"unknown payload method: {exc.args[0]}") from exc
    else:
        selected_methods = METHODS
    rows: list[dict[str, Any]] = []
    for method in selected_methods:
        method_feedback_mode = (
            feedback_schedule_mode
            if method.feedback_trigger_mode == "configured"
            else method.feedback_trigger_mode
        )
        schedule: tuple[tuple[float, float], ...] = ()
        if method.online_feedback:
            if method_feedback_mode == "elapsed_time":
                schedule = ((condition["intervention_time"] + feedback_latency, feedback_gamma),)
        config = SmoothDynamicConfig(
            delta_u_weight=method.delta_u_weight,
            gamma=method.gamma,
            seed=int(condition["seed"]),
            max_steps=max_steps,
            reference_speed=float(payload.get("reference_speed", 0.08)),
            goal_offset=tuple(payload.get("goal_offset", (0.0, 0.30, 0.0))),
            reference_mode=method.reference_mode,
            safety_mode=method.safety_mode,
            prediction_mode=method.prediction_mode,
            safety_reflex_enabled=method.safety_reflex_enabled,
            measurement_noise_mode=(
                "bounded_ball" if method.bounded_measurement_noise else "gaussian"
            ),
            measurement_noise_sigma=float(
                payload.get("measurement_noise_sigma", 0.005)
            ),
            measurement_error_bound=0.005,
            sensor_period=(
                0.04
                if method.bounded_measurement_noise
                else float(payload.get("sensor_period", 0.67))
            ),
            known_obstacle_velocity=method.known_obstacle_velocity,
            formal_safety_filter_enabled=method.formal_safety_filter_enabled,
            formal_robot_transition_error_bound=0.008,
            formal_filter_speed_limit=0.4,
            formal_obstacle_speed_bound=float(payload.get("speed_upper", 0.20)),
            optimal_decay_weight=method.optimal_decay_weight,
            gamma_schedule=schedule,
            gamma_schedule_request_times=(
                (condition["intervention_time"],)
                if method.online_feedback and method_feedback_mode == "elapsed_time"
                else ()
            ),
            provisional_feedback_times=(
                (condition["intervention_time"],)
                if method.provisional_local_feedback
                and method_feedback_mode == "elapsed_time"
                else ()
            ),
            feedback_ttc_threshold=(
                feedback_ttc_threshold
                if method.online_feedback and method_feedback_mode == "ttc"
                else None
            ),
            feedback_response_latency=feedback_latency,
            feedback_gamma=(
                feedback_gamma
                if method.online_feedback and method_feedback_mode == "ttc"
                else None
            ),
            gamma_update_ttl=float(payload.get("gamma_update_ttl", 1.0)),
            obstacle_start_offset=(
                condition["lateral_offset"],
                float(payload.get("obstacle_start_forward_offset", 0.44)),
                float(payload.get("obstacle_start_vertical_offset", 0.0)),
            ),
            obstacle_velocity=(0.0, -condition["obstacle_speed"], 0.0),
            obstacle_radius=float(payload.get("obstacle_radius", 0.10)),
            collision_radius=float(payload.get("collision_radius", 0.035)),
            solver_max_constraint_violation=float(
                payload.get("solver_max_constraint_violation", 1e-6)
            ),
            solver_max_cpu_time=float(payload.get("solver_max_cpu_time", 0.035)),
            control_deadline=float(payload.get("control_deadline", 0.04)),
            reject_deadline_miss=bool(payload.get("reject_deadline_miss", False)),
            save_animation=False,
            save_plots=False,
            save_metrics=False,
            output_dir=f"/tmp/lampc-paired-{os.getpid()}",
            cbf_transition_mode=str(
                payload.get("cbf_transition_mode", "command_velocity")
            ),
        )
        result = run_smooth_dynamic_demo(config)
        formal_exact_or_bounded_observation = bool(
            config.measurement_noise_mode == "bounded_ball"
            and config.measurement_error_bound >= 0.0
        )
        formal_applied_input_matches_mpc = bool(
            result.reflex_interventions == 0
            and result.solver_rejections == 0
            and result.emergency_fallbacks == 0
        )
        formal_model_match_verified = bool(
            result.max_action_tracking_error
            <= (
                config.formal_robot_transition_error_bound
                if config.formal_safety_filter_enabled
                else 1e-6
            )
        )
        formal_initial_safe = result.initial_true_barrier >= 0.0
        formal_raw_discrete_cbf_satisfied = bool(
            method.safety_mode == "cbf" and result.true_cbf_violation_steps == 0
        )
        formal_final_input_certified = bool(
            config.formal_safety_filter_enabled
            and result.formal_filter_uncertified_steps == 0
        )
        formal_robust_filter_satisfied = bool(
            config.formal_safety_filter_enabled
            and result.minimum_robust_filter_residual is not None
            and result.minimum_robust_filter_residual >= -1e-9
        )
        formal_terminal_certified = bool(
            config.formal_safety_filter_enabled
            and result.formal_terminal_backup_uncertified_steps == 0
            and result.minimum_backup_authority_margin is not None
            and result.minimum_backup_authority_margin >= -1e-9
        )
        formal_stepwise_certificate_eligible = all(
            (
                method.safety_mode == "cbf",
                formal_initial_safe,
                formal_exact_or_bounded_observation,
                formal_final_input_certified,
                formal_robust_filter_satisfied,
                formal_model_match_verified,
            )
        )
        formal_recursive_certificate_eligible = bool(
            formal_stepwise_certificate_eligible and formal_terminal_certified
        )
        elapsed_episode_time = result.steps * 0.04
        feedback_local_causal_opportunity = bool(
            method.provisional_local_feedback
            and condition["intervention_time"] < elapsed_episode_time
        )
        feedback_llm_causal_opportunity = bool(
            method.online_feedback and result.gamma_updates_applied > 0
        )
        feedback_available_time = (
            condition["intervention_time"] + feedback_latency
            if method.online_feedback and method_feedback_mode == "elapsed_time"
            else result.feedback_available_time
        )
        feedback_response_arrived_before_termination = bool(
            method.online_feedback
            and feedback_available_time is not None
            and feedback_available_time < elapsed_episode_time
        )
        rows.append(
            {
                **condition,
                "protocol_version": protocol_version,
                "benchmark_stage": benchmark_stage,
                "max_steps_budget": max_steps,
                "manifest_hash": payload.get("manifest_hash"),
                "method": method.name,
                "comparator": method.comparator,
                "experiment_profile": method.experiment_profile,
                "outcome": result.outcome,
                "joint_success": bool(
                    result.outcome == "goal" and result.reached_goal and not result.collision
                ),
                # Compatibility alias. Summaries recompute the endpoint from the
                # terminal outcome instead of trusting this stored value.
                "success": bool(
                    result.outcome == "goal" and result.reached_goal and not result.collision
                ),
                "reached_goal": result.reached_goal,
                "collision": result.collision,
                "steps": result.steps,
                "minimum_true_clearance": result.minimum_true_clearance,
                "minimum_measured_clearance": result.minimum_measured_clearance,
                "initial_true_barrier": result.initial_true_barrier,
                "minimum_true_barrier": result.minimum_true_barrier,
                "minimum_true_cbf_residual": result.minimum_true_cbf_residual,
                "true_cbf_violation_steps": result.true_cbf_violation_steps,
                "final_goal_distance": result.final_goal_distance,
                "initial_goal_distance": result.initial_goal_distance,
                "net_goal_progress": result.net_goal_progress,
                "mean_goal_progress_rate": result.mean_goal_progress_rate,
                "final_speed_scale": result.final_speed_scale,
                "final_clearance_margin": result.final_clearance_margin,
                "safety_profile_transitions": result.safety_profile_transitions,
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
                "feedback_schedule_mode": method_feedback_mode,
                "feedback_trigger_time": (
                    condition["intervention_time"]
                    if method.online_feedback and method_feedback_mode == "elapsed_time"
                    else result.feedback_trigger_time
                ),
                "feedback_available_time": feedback_available_time,
                "feedback_causal_opportunity": (
                    feedback_llm_causal_opportunity
                    if method_feedback_mode == "elapsed_time"
                    else result.feedback_causal_opportunity
                ),
                "feedback_local_causal_opportunity": (
                    feedback_local_causal_opportunity
                ),
                "feedback_llm_causal_opportunity": (
                    feedback_llm_causal_opportunity
                ),
                "feedback_response_arrived_before_termination": (
                    feedback_response_arrived_before_termination
                ),
                "feedback_unavailable_by_termination": bool(
                    method.online_feedback
                    and not feedback_response_arrived_before_termination
                ),
                "feedback_updates_rejected_late": result.feedback_updates_rejected_late,
                "formal_filter_interventions": result.formal_filter_interventions,
                "formal_filter_uncertified_steps": (
                    result.formal_filter_uncertified_steps
                ),
                "formal_terminal_backup_uncertified_steps": (
                    result.formal_terminal_backup_uncertified_steps
                ),
                "minimum_robust_filter_residual": (
                    result.minimum_robust_filter_residual
                ),
                "minimum_backup_authority_margin": (
                    result.minimum_backup_authority_margin
                ),
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
                "formal_initial_safe": formal_initial_safe,
                "formal_raw_discrete_cbf_satisfied": (
                    formal_raw_discrete_cbf_satisfied
                ),
                "formal_exact_or_bounded_observation": (
                    formal_exact_or_bounded_observation
                ),
                "formal_applied_input_matches_mpc": formal_applied_input_matches_mpc,
                "formal_final_input_certified": formal_final_input_certified,
                "formal_robust_filter_satisfied": formal_robust_filter_satisfied,
                "formal_model_match_verified": formal_model_match_verified,
                "formal_terminal_safe_set_or_backup_certified": (
                    formal_terminal_certified
                ),
                "formal_stepwise_certificate_eligible": (
                    formal_stepwise_certificate_eligible
                ),
                "formal_recursive_certificate_eligible": (
                    formal_recursive_certificate_eligible
                ),
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
        "joint_success", "success", "reached_goal", "collision",
        "feedback_causal_opportunity", "feedback_local_causal_opportunity",
        "feedback_llm_causal_opportunity",
        "feedback_response_arrived_before_termination",
        "feedback_unavailable_by_termination", "formal_initial_safe",
        "formal_raw_discrete_cbf_satisfied",
        "formal_exact_or_bounded_observation",
        "formal_applied_input_matches_mpc", "formal_model_match_verified",
        "formal_final_input_certified", "formal_robust_filter_satisfied",
        "formal_terminal_safe_set_or_backup_certified",
        "formal_stepwise_certificate_eligible",
        "formal_recursive_certificate_eligible",
    }
    integer_fields = {
        "protocol_version", "max_steps_budget", "episode", "seed", "steps",
        "gamma_updates_applied", "gamma_updates_rejected",
        "reflex_interventions", "reflex_backups",
        "solver_failures", "solver_rejections", "deadline_misses",
        "solver_max_cpu_time_exits", "solver_infeasible_exits",
        "solver_unknown_exits",
        "emergency_fallbacks", "feedback_updates_rejected_late",
        "most_infeasible_stage", "infeasible_stage_events",
        "safety_profile_transitions",
        "true_cbf_violation_steps",
        "formal_filter_interventions", "formal_filter_uncertified_steps",
        "formal_terminal_backup_uncertified_steps",
    }
    text_fields = {
        "benchmark_stage", "method", "comparator", "experiment_profile", "outcome",
        "feedback_schedule_mode", "manifest_hash",
    }
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if key in text_fields:
                parsed[key] = None if value in {"", "None"} else value
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

    methods = configured_methods(config)
    expected = {method.name for method in methods}
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
        for method in methods
    }
    for method in methods:
        method_rows = by_method[method.name]
        if len(method_rows) != config.episodes:
            raise ValueError(f"method {method.name} does not have {config.episodes} episodes")
        comparator_rows = by_method[method.comparator]
        method_keys = [_pairing_key(row) for row in method_rows]
        comparator_keys = [_pairing_key(row) for row in comparator_rows]
        if len(set(method_keys)) != len(method_keys):
            raise ValueError(f"method {method.name} contains duplicate paired conditions")
        if method_keys != comparator_keys:
            raise ValueError(
                f"method {method.name} is not paired with comparator {method.comparator}"
            )
        method_joint = [_joint_success(row) for row in method_rows]
        comparator_joint = [_joint_success(row) for row in comparator_rows]
        outcomes = np.asarray(method_joint, dtype=float)
        comparator = np.asarray(comparator_joint, dtype=float)
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
            comparator_joint,
            method_joint,
        )
        raw_pvalues[method.name] = pvalue
        summaries[method.name] = {
            "episodes": len(method_rows),
            "paired_conditions_verified": True,
            "endpoint": "joint_success = outcome=='goal' AND reached_goal AND NOT collision",
            "successes": int(np.sum(outcomes)),
            "success_rate": float(np.mean(outcomes)),
            "success_bootstrap_95_ci": list(success_ci),
            "comparator": method.comparator,
            "paired_success_difference": float(np.mean(paired_difference)),
            "paired_success_difference_95_ci": list(difference_ci),
            "mcnemar_exact_p": pvalue,
            "collisions": sum(bool(row["collision"]) for row in method_rows),
            "timeout_failures": sum(
                row.get("outcome") in {"timeout", "safety_timeout", "environment_truncated"}
                for row in method_rows
            ),
            "stored_success_mismatches": sum(
                bool(row.get("success", False)) != joint
                or (
                    "joint_success" in row
                    and bool(row.get("joint_success", False)) != joint
                )
                for row, joint in zip(method_rows, method_joint)
            ),
            "outcomes": {
                outcome: sum(row.get("outcome") == outcome for row in method_rows)
                for outcome in (
                    "goal", "collision", "timeout", "safety_timeout",
                    "controller_stall", "solver_failure", "emergency_fallback",
                    "environment_truncated",
                )
            },
            "mean_minimum_true_clearance": float(np.mean(clearance)),
            "minimum_raw_true_barrier": (
                float(np.min([
                    row["minimum_true_barrier"] for row in method_rows
                    if row.get("minimum_true_barrier") is not None
                ]))
                if any(
                    row.get("minimum_true_barrier") is not None
                    for row in method_rows
                )
                else None
            ),
            "minimum_raw_true_cbf_residual": (
                float(np.min([
                    row["minimum_true_cbf_residual"] for row in method_rows
                    if row.get("minimum_true_cbf_residual") is not None
                ]))
                if any(
                    row.get("minimum_true_cbf_residual") is not None
                    for row in method_rows
                )
                else None
            ),
            "raw_true_cbf_violation_steps": sum(
                int(row.get("true_cbf_violation_steps", 0))
                for row in method_rows
            ),
            "formal_filter_interventions": sum(
                int(row.get("formal_filter_interventions", 0))
                for row in method_rows
            ),
            "formal_filter_uncertified_steps": sum(
                int(row.get("formal_filter_uncertified_steps", 0))
                for row in method_rows
            ),
            "formal_terminal_backup_uncertified_steps": sum(
                int(row.get("formal_terminal_backup_uncertified_steps", 0))
                for row in method_rows
            ),
            "minimum_robust_filter_residual": (
                float(np.min([
                    row["minimum_robust_filter_residual"] for row in method_rows
                    if row.get("minimum_robust_filter_residual") is not None
                ]))
                if any(
                    row.get("minimum_robust_filter_residual") is not None
                    for row in method_rows
                )
                else None
            ),
            "minimum_backup_authority_margin": (
                float(np.min([
                    row["minimum_backup_authority_margin"] for row in method_rows
                    if row.get("minimum_backup_authority_margin") is not None
                ]))
                if any(
                    row.get("minimum_backup_authority_margin") is not None
                    for row in method_rows
                )
                else None
            ),
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
            "feedback_local_causal_opportunities": sum(
                bool(row.get("feedback_local_causal_opportunity", False))
                for row in method_rows
            ),
            "feedback_llm_causal_opportunities": sum(
                bool(row.get("feedback_llm_causal_opportunity", False))
                for row in method_rows
            ),
            "feedback_responses_arrived_before_termination": sum(
                bool(row.get("feedback_response_arrived_before_termination", False))
                for row in method_rows
            ),
            "feedback_unavailable_by_termination": sum(
                bool(row.get("feedback_unavailable_by_termination", False))
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
            "formal_scope_audit": {
                "stepwise_certificate_eligible_episodes": sum(
                    bool(row.get("formal_stepwise_certificate_eligible", False))
                    for row in method_rows
                ),
                "recursive_certificate_eligible_episodes": sum(
                    bool(row.get("formal_recursive_certificate_eligible", False))
                    for row in method_rows
                ),
                "initial_safe_episodes": sum(
                    bool(row.get("formal_initial_safe", False))
                    for row in method_rows
                ),
                "raw_discrete_cbf_satisfied_episodes": sum(
                    bool(row.get("formal_raw_discrete_cbf_satisfied", False))
                    for row in method_rows
                ),
                "exact_or_bounded_observation_episodes": sum(
                    bool(row.get("formal_exact_or_bounded_observation", False))
                    for row in method_rows
                ),
                "applied_input_matches_mpc_episodes": sum(
                    bool(row.get("formal_applied_input_matches_mpc", False))
                    for row in method_rows
                ),
                "final_input_certified_episodes": sum(
                    bool(row.get("formal_final_input_certified", False))
                    for row in method_rows
                ),
                "robust_filter_satisfied_episodes": sum(
                    bool(row.get("formal_robust_filter_satisfied", False))
                    for row in method_rows
                ),
                "model_match_verified_episodes": sum(
                    bool(row.get("formal_model_match_verified", False))
                    for row in method_rows
                ),
                "terminal_safe_set_or_backup_certified_episodes": sum(
                    bool(row.get(
                        "formal_terminal_safe_set_or_backup_certified", False
                    ))
                    for row in method_rows
                ),
            },
        }
    adjusted = _holm_adjust(raw_pvalues)
    for name, value in adjusted.items():
        summaries[name]["mcnemar_holm_adjusted_p"] = value
    return summaries


def evaluate_confirmatory_efficacy_gate(
    summaries: Mapping[str, Mapping[str, Any]],
    config: PairedBenchmarkConfig,
) -> dict[str, Any]:
    """Evaluate the single preregistered online-feedback efficacy contrast."""

    if config.efficacy_method not in summaries:
        raise ValueError(f"unknown efficacy method: {config.efficacy_method}")
    method = summaries[config.efficacy_method]
    if method.get("comparator") != config.efficacy_comparator:
        raise ValueError("efficacy method does not use the configured comparator")

    difference = float(method["paired_success_difference"])
    lower, upper = (float(value) for value in method["paired_success_difference_95_ci"])
    pvalue = float(method["mcnemar_exact_p"])
    margin = config.efficacy_minimum_paired_difference
    checks = {
        "paired_joint_success_difference_above_margin": difference > margin,
        "bootstrap_95_ci_lower_above_margin": lower > margin,
        "exact_two_sided_mcnemar_p_at_most_alpha": pvalue <= config.efficacy_alpha,
    }
    evaluated = config.stage in {"paper-replication", "confirmatory"}
    return {
        "evaluated": evaluated,
        "passed": all(checks.values()) if evaluated else None,
        "primary_method": config.efficacy_method,
        "comparator": config.efficacy_comparator,
        "endpoint": "paired joint success; all timeout outcomes count as failure",
        "minimum_paired_difference": margin,
        "alpha": config.efficacy_alpha,
        "paired_joint_success_difference": difference,
        "paired_joint_success_difference_95_ci": [lower, upper],
        "exact_two_sided_mcnemar_p": pvalue,
        "paper_reported_absolute_improvement": 0.34,
        "paper_effect_size_reached": difference >= 0.34,
        "checks": checks,
    }


def evaluate_secondary_robust_contrast(
    summaries: Mapping[str, Mapping[str, Any]],
    config: PairedBenchmarkConfig,
) -> dict[str, Any]:
    """Report the robust-stack feedback contrast without making it the paper gate."""

    method_name = "robust_stack_async_feedback"
    comparator_name = "robust_stack_fixed_g015"
    if method_name not in summaries or comparator_name not in summaries:
        return {
            "evaluated_as_gate": False,
            "method": method_name,
            "comparator": comparator_name,
            "reason": "robust extension methods were not selected for this stage",
        }
    method = summaries[method_name]
    if method.get("comparator") != comparator_name:
        raise ValueError("robust extension comparator is not isolated")
    difference = float(method["paired_success_difference"])
    lower, upper = (float(value) for value in method["paired_success_difference_95_ci"])
    return {
        "evaluated_as_gate": False,
        "method": method_name,
        "comparator": comparator_name,
        "paired_joint_success_difference": difference,
        "paired_joint_success_difference_95_ci": [lower, upper],
        "exact_two_sided_mcnemar_p": float(method["mcnemar_exact_p"]),
        "reason": "robust extension is secondary and cannot replace the paper-fidelity claim",
    }


def evaluate_formal_contract_gate(
    summaries: Mapping[str, Mapping[str, Any]],
    config: PairedBenchmarkConfig,
) -> dict[str, Any]:
    """Check the bounded-error, final-input, and terminal-backup contract.

    Passing this gate validates every sampled episode against declared bounds;
    it is not a whole-body Panda proof and cannot validate an incorrect bound.
    """

    method_names = ("formal_stack_fixed_g015", "formal_stack_async_feedback")
    if not any(name in summaries for name in method_names):
        return {
            "evaluated": False,
            "passed": None,
            "scope": (
                "formal extension methods were not selected for this stage"
            ),
            "whole_body_panda_certified": False,
            "checks": {},
        }
    if not all(name in summaries for name in method_names):
        raise ValueError("formal contract gate requires both formal methods")
    checks: dict[str, bool] = {}
    for name in method_names:
        summary = summaries[name]
        audit = summary["formal_scope_audit"]
        episodes = int(summary["episodes"])
        checks[f"{name}:zero_collision"] = int(summary["collisions"]) == 0
        checks[f"{name}:all_recursive_eligible"] = (
            int(audit["recursive_certificate_eligible_episodes"]) == episodes
        )
        checks[f"{name}:zero_filter_uncertified_steps"] = (
            int(summary["formal_filter_uncertified_steps"]) == 0
        )
        checks[f"{name}:zero_terminal_uncertified_steps"] = (
            int(summary["formal_terminal_backup_uncertified_steps"]) == 0
        )
        checks[f"{name}:positive_backup_authority_margin"] = (
            summary.get("minimum_backup_authority_margin") is not None
            and float(summary["minimum_backup_authority_margin"]) > 0.0
        )
    evaluated = config.stage in {"development", "confirmatory"}
    return {
        "evaluated": evaluated,
        "passed": all(checks.values()) if evaluated else None,
        "scope": (
            "spherical end-effector versus one spherical obstacle under the "
            "declared bounded measurement, obstacle-motion, and transition errors"
        ),
        "whole_body_panda_certified": False,
        "checks": checks,
    }


def _plot_summary(
    rows: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Any],
    methods: Sequence[BenchmarkMethod],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [method.name for method in methods]
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
    feedback_decisions: Sequence[GammaDecision],
    config: PairedBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run or resume all methods on identical Safe Panda Gym conditions."""

    cfg = config or PairedBenchmarkConfig()
    methods = configured_methods(cfg)
    decisions = validate_feedback_decision_trace(feedback_decisions, cfg)
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "episodes.csv"
    rows = _read_checkpoint(csv_path) if cfg.resume else []
    if rows and any(
        int(row.get("protocol_version", -1)) != 5
        or row.get("benchmark_stage") != cfg.stage
        or int(row.get("max_steps_budget", -1)) != cfg.max_steps
        or row.get("manifest_hash") != cfg.manifest_hash
        for row in rows
    ):
        raise ValueError(
            "checkpoint protocol, stage, or max-step budget differs from this run; "
            "use a new output directory"
        )
    complete = {
        int(episode)
        for episode in {row["episode"] for row in rows}
        if sum(int(row["episode"]) == int(episode) for row in rows) == len(methods)
    }
    conditions = [item for item in _conditions(cfg) if item["episode"] not in complete]
    if rows:
        expected_latency = {
            episode: decisions[episode].latency_seconds
            for episode in range(cfg.episodes)
        }
        if any(
            bool(row.get("method") in {
                "paper_async_feedback_static", "robust_stack_async_feedback",
                "formal_stack_async_feedback",
            })
            and abs(
                float(row.get("feedback_latency", -1.0))
                - expected_latency[int(row["episode"])]
            ) > 1e-12
            for row in rows
        ):
            raise ValueError(
                "checkpoint feedback latency trace differs from this run; "
                "use a new output directory"
            )
    print(
        f"[paired] {len(complete)}/{cfg.episodes} conditions restored; "
        f"{len(conditions)} pending across {cfg.workers} workers",
        flush=True,
    )
    payloads = [
        {
            "condition": condition,
            "max_steps": cfg.max_steps,
            "feedback_gamma": decisions[int(condition["episode"])].gamma,
            "feedback_latency": decisions[int(condition["episode"])].latency_seconds,
            "feedback_schedule_mode": cfg.feedback_schedule_mode,
            "feedback_ttc_threshold": cfg.feedback_ttc_threshold,
            "protocol_version": 5,
            "benchmark_stage": cfg.stage,
            "speed_upper": cfg.speed_upper,
            "method_names": tuple(method.name for method in methods),
            "sensor_period": cfg.sensor_period,
            "measurement_noise_sigma": cfg.measurement_noise_sigma,
            "reference_speed": cfg.reference_speed,
            "cbf_transition_mode": cfg.cbf_transition_mode,
            "manifest_hash": cfg.manifest_hash,
            "goal_offset": cfg.goal_offset,
            "obstacle_start_forward_offset": cfg.obstacle_start_forward_offset,
            "obstacle_start_vertical_offset": cfg.obstacle_start_vertical_offset,
            "obstacle_radius": cfg.obstacle_radius,
            "collision_radius": cfg.collision_radius,
            "gamma_update_ttl": cfg.gamma_update_ttl,
            "solver_max_constraint_violation": cfg.solver_max_constraint_violation,
            "solver_max_cpu_time": cfg.solver_max_cpu_time,
            "control_deadline": cfg.control_deadline,
            "reject_deadline_miss": cfg.reject_deadline_miss,
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
    efficacy_gate = evaluate_confirmatory_efficacy_gate(summaries, cfg)
    secondary_robust_contrast = evaluate_secondary_robust_contrast(summaries, cfg)
    formal_contract_gate = evaluate_formal_contract_gate(summaries, cfg)
    summary = {
        "study_id": f"safe-panda-paired-protocol-v5-{cfg.stage}-{cfg.episodes}",
        "protocol_version": 5,
        "config": asdict(cfg),
        "selected_methods": [method.name for method in methods],
        "paper_fidelity_manifest": (
            {
                "path": cfg.manifest_path,
                "sha256": cfg.manifest_hash,
                "profile": cfg.manifest_profile,
                "model_substitution": cfg.model_substitution,
            }
            if cfg.manifest_hash is not None
            else None
        ),
        "methods": summaries,
        "efficacy_gate": efficacy_gate,
        "secondary_robust_contrast": secondary_robust_contrast,
        "formal_contract_gate": formal_contract_gate,
        "feedback_decisions": [
            {
                key: value
                for key, value in decision.as_dict().items()
                if key != "raw_response"
            }
            for decision in decisions
        ],
        "feedback_latency": {
            "mode": cfg.latency_trace_mode,
            "request_policy": cfg.feedback_request_policy,
            "requests_per_episode": cfg.feedback_requests_per_episode,
            "request_count": len(decisions),
            "minimum_seconds": min(decision.latency_seconds for decision in decisions),
            "mean_seconds": sum(
                decision.latency_seconds for decision in decisions
            ) / len(decisions),
            "maximum_seconds": max(decision.latency_seconds for decision in decisions),
            "unique_request_hashes": len({
                decision.request_hash for decision in decisions
            }),
            "unique_request_timestamps": len({
                decision.requested_at_unix for decision in decisions
            }),
            "all_latency_samples_identical": len({
                decision.latency_seconds for decision in decisions
            }) == 1,
        },
        "paired_randomization": True,
        "common_random_numbers": True,
        "simulator": "PandaReachSafe-v3 / PyBullet Tiny / do-mpc / CasADi / IPOPT",
        "statistical_tests": {
            "success_rate": "non-parametric bootstrap 95% CI",
            "paired_success": "paired bootstrap difference + exact McNemar",
            "multiple_comparisons": "Holm adjustment",
            "safety_source": "raw end-effector and true obstacle trajectory",
            "confirmatory_primary_endpoint": (
                "paired joint success; timeout, stall, solver failure, emergency fallback, "
                "and environment truncation are failures"
            ),
        },
        "formal_scope": (
            "Monte Carlo is not a proof. Paper-fidelity and robust-extension profiles "
            "remain formally ineligible under Gaussian observation noise or an "
            "uncertified post-MPC reflex. The formal-extension profile uses bounded-ball "
            "measurement error, a declared additive transition-error bound, a final "
            "discrete robust safety filter, and a radial escape backup set. Its contract "
            "covers only the spherical end-effector/obstacle model, not Panda links, "
            "joints, contact dynamics, or correctness of the declared bounds."
        ),
    }
    (root / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot_summary(
        rows, summaries, methods, root / "paired_success_and_clearance.png"
    )
    print("[paired] complete", flush=True)
    return summary
