"""Continuous-reference MPC-CBF experiment and smoothness ablation primitive."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from .smoothness import SmoothnessMetrics, make_reference_bspline


@dataclass(frozen=True, slots=True)
class SmoothDynamicConfig:
    delta_u_weight: float = 0.5
    jerk_weight: float = 0.0
    optimal_decay_weight: float = 0.0
    optimal_decay_lower: float = 0.1
    gamma: float = 0.10
    seed: int = 11
    max_steps: int = 260
    render_stride: int = 2
    save_animation: bool = False
    sensor_period: float = 0.67
    measurement_noise_sigma: float = 0.005
    measurement_noise_mode: str = "gaussian"
    measurement_error_bound: float = 0.005
    reference_speed: float = 0.12
    goal_offset: tuple[float, float, float] = (0.0, 0.30, 0.0)
    position_q_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    obstacle_start_offset: tuple[float, float, float] = (-0.12, 0.15, 0.0)
    obstacle_velocity: tuple[float, float, float] = (0.06, 0.0, 0.0)
    obstacle_radius: float = 0.10
    collision_radius: float = 0.035
    prediction_mode: str = "velocity_tube"
    confidence_multiplier: float = 3.0
    velocity_error_bound: float = 0.03
    initial_velocity_error_bound: float = 0.20
    model_error_growth: float = 0.005
    max_relative_speed: float = 0.4
    total_latency: float = 0.04
    obstacle_acceleration_bound: float = 0.0
    sampled_data_margin_enabled: bool = True
    velocity_filter: float = 1.0
    robot_velocity_filter: float = 0.5
    robot_velocity_maximum: float = 0.4
    robot_velocity_feedback_enabled: bool = True
    known_obstacle_velocity: bool = False
    formal_safety_filter_enabled: bool = False
    formal_robot_transition_error_bound: float = 0.008
    formal_filter_speed_limit: float = 0.4
    formal_obstacle_speed_bound: float = 0.20
    formal_obstacle_velocity_error_bound: float = 0.0
    abort_on_formal_certificate_failure: bool = True
    safety_reflex_enabled: bool = True
    reflex_lookahead_steps: int = 8
    reflex_alpha: float = 4.0
    reflex_backup_selection: str = "task_consistent"
    reflex_committed_backup_enabled: bool = True
    reflex_committed_backup_steps: int = 8
    reflex_barrier_mode: str = "radial_cbf"
    reflex_side_latch_enabled: bool = False
    reflex_side_latch_steps: int = 15
    reflex_side_release_clearance: float = 0.05
    reflex_side_switch_hysteresis: float = 0.01
    reflex_policy_library_enabled: bool = False
    reflex_policy_speed_scales: tuple[float, ...] = (0.50, 0.75, 1.0)
    reflex_recovery_clearance_slack: float = 0.005
    reflex_tangential_subgoal_enabled: bool = False
    reflex_tangential_subgoal_distance: float = 0.10
    reflex_dpcbf_safety_scale: float = 1.05
    reflex_dpcbf_lambda_gain: float = 0.10
    reflex_dpcbf_mu_gain: float = 0.50
    avoidance_onset_threshold: float = 0.005
    route_margin: float = 0.08
    reference_mode: str = "behind_spline"
    reference_route_profile: str = "legacy_lateral"
    avoidance_waypoint_offsets: tuple[tuple[float, float, float], ...] = ()
    safety_mode: str = "cbf"
    # ``command_velocity`` matches the immediate Safe Panda command.
    # ``paper_state`` retains the replacement-state transition.
    # ``double_integrator`` uses acceleration input and exact discretization.
    cbf_transition_mode: str = "command_velocity"
    # ``horizon`` keeps the frozen-baseline CBF constraint on every stage.
    # ``first_step`` is the opt-in hard one-step D-GCBF remedy profile.
    cbf_constraint_scope: str = "horizon"
    # ``zero`` keeps the frozen fail-closed zero command on solver rejection.
    # ``brake`` is the opt-in braking-fallback remedy profile: decay the last
    # velocity and accept it only if it passes a one-step CBF screen.
    solver_fallback_mode: str = "zero"
    fallback_braking_decay: float = 0.5
    # Paper eq. (3e) declares a terminal set X_f without publishing it.
    # This opt-in reconstruction enforces the terminal safe set h(p_N) >= 0.
    # Deviation registry entry 1.2.
    terminal_safe_set_enabled: bool = False
    # ``repository`` keeps the frozen IpoptConfig; ``reference_defaults``
    # uses library-default IPOPT as verified in the reference repositories
    # (deviation registry entry 1.3).
    ipopt_profile: str = "repository"
    # Dead-time-robust CBF margin for the static prediction mode (ZOCBF-style
    # remedy, arXiv:2005.06418 / 2411.17079): inflate the constraint radius by
    # ``speed_bound * (time since last measurement + stage offset)`` so the
    # barrier covers any obstacle motion between sensor updates. ``off``
    # preserves the frozen baseline. Planning-side only — true-clearance
    # metrics keep the raw combined radius.
    dead_time_margin_mode: str = "off"
    dead_time_obstacle_speed_bound: float = 0.20
    # Soft-slack remedy (skill remedy profile 1): per-stage CBF slack with an
    # L1 exact penalty. 0 keeps the frozen hard constraint.
    cbf_slack_weight: float = 0.0
    # ``paper_experiment`` keeps the frozen (0, 0.15] gamma interval from the
    # paper's trajectory experiment. ``paper_continuous`` opts into the full
    # published Table-1/2 range (0, 1] for scheduled/feedback gammas
    # (registry 3.3 paper-fidelity path).
    gamma_range_mode: str = "paper_experiment"
    # Channel-2 feedback arm: language switches the obstacle PREDICTION
    # mode mid-episode instead of (or alongside) gamma. Each entry is
    # (time, mode); mode must be a valid prediction_mode. gamma is left
    # untouched by this schedule -- it isolates the prediction channel.
    prediction_mode_schedule: tuple[tuple[float, str], ...] = ()
    prediction_mode_schedule_request_times: tuple[float, ...] = ()
    gamma_schedule: tuple[tuple[float, float], ...] = ()
    gamma_schedule_request_times: tuple[float, ...] = ()
    gamma_update_ttl: float = 1.0
    context_safety_enabled: bool = False
    requested_safety_level: int = 3
    safety_exit_ttc_hysteresis: float = 0.50
    safety_clear_hold_time: float = 0.40
    safety_recovery_duration: float = 0.80
    safety_profile_recovery_enabled: bool = True
    stall_progress_threshold: float = 0.01
    provisional_feedback_times: tuple[float, ...] = ()
    feedback_ttc_threshold: float | None = None
    feedback_response_latency: float = 0.0
    feedback_gamma: float | None = None
    feedback_reaction_margin: float = 0.20
    reject_feedback_without_causal_opportunity: bool = True
    solver_max_constraint_violation: float = 1e-6
    solver_max_cpu_time: float = 0.035
    control_deadline: float = 0.04
    reject_deadline_miss: bool = False
    save_plots: bool = True
    save_metrics: bool = True
    output_dir: str = "artifacts/smoothness_ablation/spline_du_0.5"

    @property
    def gamma_upper(self) -> float:
        return 1.0 if self.gamma_range_mode == "paper_continuous" else 0.15

    def __post_init__(self) -> None:
        if self.delta_u_weight < 0.0 or self.jerk_weight < 0.0 or self.optimal_decay_weight < 0.0:
            raise ValueError("smoothness weights must be non-negative")
        if not 0.0 < self.optimal_decay_lower <= 1.0:
            raise ValueError("optimal_decay_lower must be in (0, 1]")
        if self.gamma_range_mode not in {"paper_experiment", "paper_continuous"}:
            raise ValueError(
                "gamma_range_mode must be paper_experiment or paper_continuous"
            )
        if not 0.0 < self.gamma <= self.gamma_upper:
            raise ValueError(
                "gamma must be in (0, %.2f] for gamma_range_mode %s"
                % (self.gamma_upper, self.gamma_range_mode)
            )
        if self.max_steps < 1 or self.render_stride < 1:
            raise ValueError("step counts must be positive")
        if self.sensor_period <= 0.0 or self.reference_speed <= 0.0:
            raise ValueError("period and reference speed must be positive")
        if self.measurement_noise_sigma < 0.0:
            raise ValueError("measurement noise must be non-negative")
        if self.measurement_noise_mode not in {"gaussian", "bounded_ball"}:
            raise ValueError("measurement_noise_mode must be gaussian or bounded_ball")
        if self.measurement_error_bound < 0.0:
            raise ValueError("measurement_error_bound must be non-negative")
        if len(self.position_q_weights) != 3 or any(
            value < 0.0 or not isfinite(value)
            for value in self.position_q_weights
        ):
            raise ValueError(
                "position_q_weights must contain three finite non-negative values"
            )
        if self.obstacle_acceleration_bound < 0.0:
            raise ValueError("obstacle_acceleration_bound must be non-negative")
        if self.gamma_update_ttl <= 0.0:
            raise ValueError("gamma_update_ttl must be positive")
        if self.reference_mode not in {"behind_spline", "straight", "direct_target"}:
            raise ValueError(
                "reference_mode must be behind_spline, straight, or direct_target"
            )
        if self.reference_route_profile not in {"legacy_lateral", "3d_waypoints"}:
            raise ValueError(
                "reference_route_profile must be legacy_lateral or 3d_waypoints"
            )
        if any(
            len(offset) != 3
            or any(not isfinite(value) for value in offset)
            for offset in self.avoidance_waypoint_offsets
        ):
            raise ValueError("avoidance_waypoint_offsets must contain finite 3-vectors")
        if (
            self.reference_route_profile == "3d_waypoints"
            and len(self.avoidance_waypoint_offsets) < 2
        ):
            raise ValueError("3d_waypoints profile requires at least two waypoint offsets")
        if self.prediction_mode not in {"static", "velocity", "velocity_tube"}:
            raise ValueError("prediction_mode must be static, velocity, or velocity_tube")
        if not 0.0 <= self.velocity_filter <= 1.0:
            raise ValueError("velocity_filter must be in [0, 1]")
        if not self.initial_velocity_error_bound >= 0.0:
            raise ValueError("initial_velocity_error_bound must be non-negative")
        if not 0.0 <= self.robot_velocity_filter <= 1.0:
            raise ValueError("robot_velocity_filter must be in [0, 1]")
        if self.robot_velocity_maximum <= 0.0:
            raise ValueError("robot_velocity_maximum must be positive")
        if any(
            value < 0.0
            for value in (
                self.formal_robot_transition_error_bound,
                self.formal_obstacle_speed_bound,
                self.formal_obstacle_velocity_error_bound,
            )
        ) or self.formal_filter_speed_limit <= 0.0:
            raise ValueError("formal safety bounds must be non-negative")
        if self.formal_safety_filter_enabled and self.measurement_noise_mode != "bounded_ball":
            raise ValueError("formal safety filter requires deterministically bounded noise")
        if self.reflex_lookahead_steps < 1 or self.reflex_alpha <= 0.0:
            raise ValueError("reflex lookahead and alpha must be positive")
        if self.reflex_backup_selection not in {"max_clearance", "task_consistent"}:
            raise ValueError("invalid reflex backup selection")
        if self.reflex_committed_backup_steps < 1:
            raise ValueError("reflex committed backup steps must be positive")
        if self.reflex_barrier_mode not in {
            "radial_cbf",
            "collision_cone",
            "dynamic_parabolic",
        }:
            raise ValueError("invalid reflex barrier mode")
        if self.reflex_side_latch_steps < 1:
            raise ValueError("reflex side latch steps must be positive")
        if self.reflex_side_release_clearance < 0.0:
            raise ValueError("reflex side release clearance must be non-negative")
        if self.reflex_side_switch_hysteresis < 0.0:
            raise ValueError("reflex side switch hysteresis must be non-negative")
        if not self.reflex_policy_speed_scales or any(
            not 0.0 < value <= 1.0 for value in self.reflex_policy_speed_scales
        ):
            raise ValueError("reflex policy speed scales must be in (0, 1]")
        if self.reflex_policy_library_enabled and self.reflex_barrier_mode not in {
            "collision_cone",
            "dynamic_parabolic",
        }:
            raise ValueError(
                "reflex policy library requires collision cone or DPCBF mode"
            )
        if (
            self.reflex_tangential_subgoal_enabled
            and not self.reflex_policy_library_enabled
        ):
            raise ValueError("tangential subgoal requires reflex policy library")
        if self.reflex_tangential_subgoal_distance < 0.0:
            raise ValueError("tangential subgoal distance must be non-negative")
        if self.reflex_recovery_clearance_slack < 0.0:
            raise ValueError("reflex recovery clearance slack must be non-negative")
        if self.reflex_dpcbf_safety_scale <= 1.0:
            raise ValueError("reflex DPCBF safety scale must exceed one")
        if self.reflex_dpcbf_lambda_gain < 0.0 or self.reflex_dpcbf_mu_gain < 0.0:
            raise ValueError("reflex DPCBF gains must be non-negative")
        if self.avoidance_onset_threshold <= 0.0:
            raise ValueError("avoidance_onset_threshold must be positive")
        if self.safety_mode not in {"cbf", "distance", "none"}:
            raise ValueError("safety_mode must be cbf, distance, or none")
        if self.cbf_transition_mode not in {
            "paper_state",
            "command_velocity",
            "double_integrator",
            "paper_increment",
        }:
            raise ValueError(
                "cbf_transition_mode must be paper_state, command_velocity, "
                "double_integrator, or paper_increment"
            )
        if self.cbf_transition_mode in {"double_integrator", "paper_increment"} and (
            self.safety_reflex_enabled or self.formal_safety_filter_enabled
        ):
            raise ValueError(
                "double_integrator and paper_increment modes require reflex "
                "and formal filters disabled"
            )
        if self.cbf_constraint_scope not in {"horizon", "first_step"}:
            raise ValueError(
                "cbf_constraint_scope must be horizon or first_step"
            )
        if self.solver_fallback_mode not in {"zero", "brake"}:
            raise ValueError("solver_fallback_mode must be zero or brake")
        if self.ipopt_profile not in {"repository", "reference_defaults"}:
            raise ValueError(
                "ipopt_profile must be repository or reference_defaults"
            )
        if not 0.0 <= self.fallback_braking_decay < 1.0:
            raise ValueError("fallback_braking_decay must be in [0, 1)")
        if not 1 <= self.requested_safety_level <= 5:
            raise ValueError("requested_safety_level must be in [1, 5]")
        if self.safety_exit_ttc_hysteresis < 0.0:
            raise ValueError("safety_exit_ttc_hysteresis must be non-negative")
        if self.safety_clear_hold_time < 0.0:
            raise ValueError("safety_clear_hold_time must be non-negative")
        if self.safety_recovery_duration <= 0.0:
            raise ValueError("safety_recovery_duration must be positive")
        if self.stall_progress_threshold < 0.0:
            raise ValueError("stall_progress_threshold must be non-negative")
        if self.solver_max_constraint_violation < 0.0:
            raise ValueError("solver_max_constraint_violation must be non-negative")
        if self.solver_max_cpu_time <= 0.0:
            raise ValueError("solver_max_cpu_time must be positive")
        if self.control_deadline <= 0.0:
            raise ValueError("control_deadline must be positive")
        if self.dead_time_margin_mode not in {"off", "speed_bound"}:
            raise ValueError("dead_time_margin_mode must be off or speed_bound")
        if self.dead_time_margin_mode == "speed_bound":
            if self.prediction_mode != "static":
                raise ValueError(
                    "dead_time_margin_mode requires prediction_mode static; "
                    "velocity modes already carry the uncertainty tube"
                )
            if self.dead_time_obstacle_speed_bound <= 0.0:
                raise ValueError(
                    "dead_time_obstacle_speed_bound must be positive"
                )
        if self.cbf_slack_weight < 0.0:
            raise ValueError("cbf_slack_weight must be non-negative")
        if any(time < 0.0 for time in self.provisional_feedback_times):
            raise ValueError("provisional feedback times must be non-negative")
        if tuple(sorted(self.provisional_feedback_times)) != self.provisional_feedback_times:
            raise ValueError("provisional feedback times must be sorted")
        previous_prediction_time = -1.0
        for switch_time, mode in self.prediction_mode_schedule:
            if switch_time < 0.0 or switch_time < previous_prediction_time:
                raise ValueError(
                    "prediction_mode_schedule times must be non-negative and sorted"
                )
            if mode not in {"static", "velocity", "velocity_tube"}:
                raise ValueError("scheduled prediction mode must be a valid mode")
            previous_prediction_time = switch_time
        if self.prediction_mode_schedule_request_times and (
            len(self.prediction_mode_schedule_request_times)
            != len(self.prediction_mode_schedule)
            or any(
                time < 0.0
                for time in self.prediction_mode_schedule_request_times
            )
        ):
            raise ValueError(
                "prediction_mode_schedule request times must be non-negative "
                "and match schedule"
            )
        if self.gamma_schedule_request_times and (
            len(self.gamma_schedule_request_times) != len(self.gamma_schedule)
            or any(time < 0.0 for time in self.gamma_schedule_request_times)
        ):
            raise ValueError(
                "gamma schedule request times must be non-negative and match schedule"
            )
        if self.feedback_ttc_threshold is not None and self.feedback_ttc_threshold <= 0.0:
            raise ValueError("feedback_ttc_threshold must be positive when provided")
        if self.feedback_response_latency < 0.0:
            raise ValueError("feedback_response_latency must be non-negative")
        if self.feedback_reaction_margin < 0.0:
            raise ValueError("feedback_reaction_margin must be non-negative")
        if self.feedback_gamma is not None and not 0.0 < self.feedback_gamma <= self.gamma_upper:
            raise ValueError(
                "feedback_gamma must be within the configured gamma range"
            )
        if (self.feedback_ttc_threshold is None) != (self.feedback_gamma is None):
            raise ValueError("TTC feedback requires both threshold and feedback_gamma")
        previous_time = -1.0
        for update_time, gamma in self.gamma_schedule:
            if update_time < 0.0 or update_time < previous_time:
                raise ValueError("gamma_schedule times must be non-negative and sorted")
            if not 0.0 < gamma <= self.gamma_upper:
                raise ValueError(
                    "scheduled gamma must be within the configured gamma range"
                )
            previous_time = update_time


@dataclass(frozen=True, slots=True)
class SmoothDynamicResult:
    outcome: str
    reached_goal: bool
    collision: bool
    steps: int
    sensor_updates: int
    minimum_true_clearance: float
    minimum_measured_clearance: float
    initial_true_barrier: float
    minimum_true_barrier: float
    minimum_true_cbf_residual: float | None
    true_cbf_violation_steps: int
    formal_filter_interventions: int
    formal_filter_uncertified_steps: int
    formal_terminal_backup_uncertified_steps: int
    minimum_robust_filter_residual: float | None
    minimum_backup_authority_margin: float | None
    final_goal_distance: float
    initial_goal_distance: float
    net_goal_progress: float
    mean_goal_progress_rate: float
    final_speed_scale: float
    final_clearance_margin: float
    safety_profile_transitions: int
    mean_solve_time: float
    max_solve_time: float
    p99_solve_time: float
    final_gamma: float
    gamma_updates_applied: int
    gamma_updates_rejected: int
    prediction_mode_updates_applied: int
    final_prediction_mode: str
    reflex_interventions: int
    reflex_backups: int
    reflex_side_switches: int
    reflex_policy_selections: int
    reflex_robust_recoveries: int
    mean_optimal_decay: float
    minimum_optimal_decay: float
    avoidance_onset_time: float | None
    minimum_predicted_ttc: float | None
    solver_failures: int
    solver_rejections: int
    solver_max_cpu_time_exits: int
    solver_infeasible_exits: int
    solver_unknown_exits: int
    deadline_misses: int
    emergency_fallbacks: int
    braking_fallbacks: int
    maximum_constraint_violation: float | None
    mean_model_transition_error: float
    max_model_transition_error: float
    mean_action_tracking_error: float
    max_action_tracking_error: float
    feedback_trigger_time: float | None
    feedback_available_time: float | None
    feedback_causal_opportunity: bool
    feedback_updates_rejected_late: int
    most_infeasible_stage: int | None
    infeasible_stage_events: int
    smoothness: SmoothnessMetrics
    output_dir: str


def sample_obstacle_measurement_noise(
    rng: Any,
    *,
    mode: str,
    sigma: float,
    error_bound: float,
) -> Any:
    """Sample Gaussian evidence noise or a deterministically bounded 3-D ball."""

    import numpy as np

    if mode == "gaussian":
        return rng.normal(0.0, sigma, size=3)
    if mode != "bounded_ball":
        raise ValueError("noise mode must be gaussian or bounded_ball")
    value = np.asarray(rng.uniform(-error_bound, error_bound, size=3), dtype=float)
    magnitude = float(np.linalg.norm(value))
    if magnitude > error_bound > 0.0:
        value *= error_bound / magnitude
    return value


def build_reference_route(
    start: Sequence[float],
    goal: Sequence[float],
    *,
    obstacle_velocity: Sequence[float],
    combined_radius: float,
    route_margin: float,
    profile: str = "legacy_lateral",
    waypoint_offsets: Sequence[Sequence[float]] = (),
) -> tuple[Any, Any]:
    """Build an opt-in 3-D route while retaining the legacy lateral route."""

    import numpy as np

    start_array = np.asarray(start, dtype=float)
    goal_array = np.asarray(goal, dtype=float)
    if start_array.shape != (3,) or goal_array.shape != (3,):
        raise ValueError("start and goal must be three-dimensional")
    if profile == "legacy_lateral":
        route_direction = -1.0 if float(obstacle_velocity[0]) >= 0.0 else 1.0
        route_points = np.asarray(
            [
                start_array,
                start_array + np.array(
                    [route_direction * (combined_radius + route_margin), 0.075, 0.0]
                ),
                start_array + np.array(
                    [route_direction * (combined_radius + route_margin), 0.225, 0.0]
                ),
                goal_array,
            ]
        )
    elif profile == "3d_waypoints":
        if len(waypoint_offsets) < 2:
            raise ValueError("3d_waypoints requires at least two waypoint offsets")
        offsets = np.asarray(waypoint_offsets, dtype=float)
        if offsets.ndim != 2 or offsets.shape[1] != 3:
            raise ValueError("waypoint_offsets must have shape (N, 3)")
        route_points = np.vstack(
            [start_array, *(start_array + offset for offset in offsets), goal_array]
        )
    else:
        raise ValueError("unknown reference route profile")
    return route_points, make_reference_bspline(route_points)


def classify_episode_outcome(
    *,
    reached_goal: bool,
    collision: bool,
    truncated: bool,
    initial_goal_distance: float,
    final_goal_distance: float,
    solver_rejections: int,
    steps: int,
    stall_progress_threshold: float = 0.01,
) -> str:
    """Classify terminal cause without treating any progress as success."""

    if collision:
        return "collision"
    if reached_goal:
        return "goal"
    if truncated:
        return "environment_truncated"
    if steps > 0 and solver_rejections >= steps:
        return "solver_failure"
    progress = float(initial_goal_distance) - float(final_goal_distance)
    if progress < stall_progress_threshold:
        return "controller_stall"
    return "safety_timeout"


@dataclass(frozen=True, slots=True)
class ReferenceObstacleStage:
    """Auditable reference and obstacle TVPs for one horizon stage."""

    reference_state: tuple[float, ...]
    obstacle_position: tuple[float, float, float]
    obstacle_next_position: tuple[float, float, float]
    robust_radius: float
    robust_radius_next: float


def braking_fallback_command(
    position: Sequence[float],
    velocity: Sequence[float],
    stage: ReferenceObstacleStage,
    *,
    gamma: float,
    dt: float,
    braking_decay: float,
    cbf_transition_mode: str,
    input_limit: float,
) -> tuple[tuple[float, float, float, float], bool]:
    """Bounded braking command for a solver-rejection step, CBF-screened.

    The braked command targets ``v_next = braking_decay * v`` within the
    input limits and is accepted only when its one-step CBF residual against
    the held obstacle measurement is non-negative; otherwise the frozen
    fail-closed zero command is returned with ``engaged=False``.
    """

    import numpy as np

    from .symbolic import barrier_value, discrete_cbf_value

    if not 0.0 <= braking_decay < 1.0:
        raise ValueError("braking_decay must be in [0, 1)")
    if dt <= 0.0 or input_limit <= 0.0:
        raise ValueError("dt and input_limit must be positive")
    current_position = np.asarray(position, dtype=float).reshape(3)
    current_velocity = np.asarray(velocity, dtype=float).reshape(3)
    braked_velocity = braking_decay * current_velocity
    if cbf_transition_mode == "double_integrator":
        command = np.clip(
            (braked_velocity - current_velocity) / dt, -input_limit, input_limit
        )
        next_position = (
            current_position + dt * current_velocity + 0.5 * dt**2 * command
        )
    elif cbf_transition_mode == "command_velocity":
        command = np.clip(braked_velocity, -input_limit, input_limit)
        next_position = current_position + dt * command
    elif cbf_transition_mode == "paper_state":
        command = np.clip(braked_velocity, -input_limit, input_limit)
        next_position = current_position + dt * current_velocity
    elif cbf_transition_mode == "paper_increment":
        # Non-paper velocity-increment variant (registry 1.1); the one-step
        # position successor does not depend on the input.
        command = np.clip(
            braked_velocity - current_velocity, -input_limit, input_limit
        )
        next_position = current_position + dt * current_velocity
    else:
        raise ValueError(
            "cbf_transition_mode must be paper_state, command_velocity, "
            "double_integrator, or paper_increment"
        )
    h_current = barrier_value(
        tuple(float(value) for value in current_position),
        stage.obstacle_position,
        stage.robust_radius,
        0.0,
    )
    h_next = barrier_value(
        tuple(float(value) for value in next_position),
        stage.obstacle_next_position,
        stage.robust_radius_next,
        0.0,
    )
    if discrete_cbf_value(h_current, h_next, gamma) < 0.0:
        return (0.0, 0.0, 0.0, 0.0), False
    return (
        float(command[0]),
        float(command[1]),
        float(command[2]),
        0.0,
    ), True


class ReferenceObstacleTVP:
    """One TVP provider for the B-spline reference and held obstacle sensor."""

    def __init__(
        self,
        reference_path: Any,
        initial_obstacle_measurement: Sequence[float],
        *,
        reference_speed: float,
        obstacle_radius: float,
        collision_radius: float,
        gamma: float,
        dt: float,
        horizon: int,
        safety_mode: str = "cbf",
        prediction_mode: str = "velocity_tube",
        tube_config: Any | None = None,
        velocity_filter: float = 1.0,
        direct_target: bool = False,
        cbf_transition_mode: str = "paper_state",
        cbf_constraint_scope: str = "horizon",
        terminal_safe_set: bool = False,
        dead_time_margin_mode: str = "off",
        dead_time_speed_bound: float = 0.0,
        gamma_upper: float = 0.15,
    ) -> None:
        import numpy as np

        from .obstacle_prediction import ConstantVelocityObserver, UncertaintyTubeConfig

        self.reference_path = np.asarray(reference_path, dtype=float)
        if self.reference_path.ndim != 2 or self.reference_path.shape[1] != 3:
            raise ValueError("reference_path must have shape (samples, 3)")
        self.measurement = np.asarray(
            initial_obstacle_measurement, dtype=float
        ).reshape(3)
        self.observer = ConstantVelocityObserver(
            tuple(float(value) for value in self.measurement),
            velocity_filter=velocity_filter,
        )
        self.reference_speed = reference_speed
        self.base_combined_radius = obstacle_radius + collision_radius
        self.clearance_margin = 0.0
        self.speed_scale = 1.0
        self.gamma = gamma
        self.dt = dt
        self.horizon = horizon
        if safety_mode not in {"cbf", "distance", "none"}:
            raise ValueError("invalid safety_mode")
        self.safety_mode = safety_mode
        if prediction_mode not in {"static", "velocity", "velocity_tube"}:
            raise ValueError("invalid prediction_mode")
        self.prediction_mode = prediction_mode
        self.tube = tube_config or UncertaintyTubeConfig()
        self.control_time = 0.0
        self.direct_target = direct_target
        if cbf_transition_mode not in {
            "paper_state",
            "command_velocity",
            "double_integrator",
            "paper_increment",
        }:
            raise ValueError(
                "cbf_transition_mode must be paper_state, command_velocity, "
                "double_integrator, or paper_increment"
            )
        self.cbf_transition_mode = cbf_transition_mode
        if cbf_constraint_scope not in {"horizon", "first_step"}:
            raise ValueError(
                "cbf_constraint_scope must be horizon or first_step"
            )
        self.cbf_constraint_scope = cbf_constraint_scope
        self.terminal_safe_set = bool(terminal_safe_set)
        if dead_time_margin_mode not in {"off", "speed_bound"}:
            raise ValueError("dead_time_margin_mode must be off or speed_bound")
        self.dead_time_margin_mode = dead_time_margin_mode
        self.dead_time_speed_bound = float(dead_time_speed_bound)
        if not 0.0 < gamma_upper <= 1.0:
            raise ValueError("gamma_upper must be in (0, 1]")
        self.gamma_upper = float(gamma_upper)
        segment_lengths = np.linalg.norm(
            np.diff(self.reference_path, axis=0), axis=1
        )
        self.arc_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        self.progress_index = 0
        self.robot_position = self.reference_path[0].copy()
        self.temporary_subgoal: Any | None = None
        self._obstacle_tvp = None
        self._obstacle_next_tvp = None
        self._robust_radius_tvp = None
        self._robust_radius_next_tvp = None
        self._reference_tvp = None
        self._gamma_tvp = None
        self._cbf_active_tvp = None
        self._terminal_safe_active_tvp = None

    def cbf_active_at_stage(self, stage: int) -> float:
        """Gate value for the CBF constraint at horizon stage ``stage``.

        Under ``first_step`` (hard one-step D-GCBF, Zeng et al. CDC 2021
        precedent) only stage 0 carries the barrier constraint; later stages
        receive a trivially satisfied inequality.
        """

        if stage < 0:
            raise ValueError("stage must be non-negative")
        if self.cbf_constraint_scope == "first_step" and stage > 0:
            return 0.0
        return 1.0

    def terminal_safe_active_at_stage(self, stage: int) -> float:
        """Gate for the reconstructed terminal set ``h(p_N) >= 0``.

        The constraint is attached at stage ``N-1``, where the successor
        expression evaluates the terminal position ``p_N``.
        """

        if stage < 0:
            raise ValueError("stage must be non-negative")
        return 1.0 if stage == self.horizon - 1 else 0.0

    def update_gamma(self, gamma: float) -> None:
        """Hot-swap an LLM-selected safety parameter without rebuilding MPC."""

        if not 0.0 < gamma <= self.gamma_upper:
            raise ValueError("gamma must be within the configured gamma range")
        self.gamma = float(gamma)

    def update_prediction_mode(self, mode: str) -> None:
        """Channel-2 feedback: hot-swap the obstacle prediction mode.

        The constant-velocity observer (``self.observer``) is updated on
        every sensor measurement regardless of the active prediction
        mode, so a switch into ``velocity_tube`` is never cold-started --
        it inherits whatever velocity estimate has accumulated so far.
        """

        if mode not in {"static", "velocity", "velocity_tube"}:
            raise ValueError("prediction mode must be static, velocity, or velocity_tube")
        self.prediction_mode = mode

    def update_safety_profile(
        self, *, gamma: float, clearance_margin: float, speed_scale: float
    ) -> None:
        """Apply a bounded context profile to all horizon stages atomically."""

        if not 0.0 < gamma <= 0.15:
            raise ValueError("gamma must be in (0, 0.15]")
        if clearance_margin < 0.0:
            raise ValueError("clearance_margin must be non-negative")
        if not 0.0 < speed_scale <= 1.0:
            raise ValueError("speed_scale must be in (0, 1]")
        self.gamma = float(gamma)
        self.clearance_margin = float(clearance_margin)
        self.speed_scale = float(speed_scale)

    def set_temporary_subgoal(
        self, subgoal: Sequence[float] | None
    ) -> None:
        """Set or release a short-lived liveness waypoint for the next MPC step."""

        import numpy as np

        if subgoal is None:
            self.temporary_subgoal = None
            return
        point = np.asarray(subgoal, dtype=float).reshape(3)
        if not np.all(np.isfinite(point)):
            raise ValueError("temporary subgoal must be a finite 3-vector")
        self.temporary_subgoal = point.copy()

    def update(
        self,
        robot_position: Sequence[float],
        obstacle_measurement: Sequence[float],
        *,
        control_time: float = 0.0,
        measurement_time: float = 0.0,
    ) -> None:
        import numpy as np

        position = np.asarray(robot_position, dtype=float).reshape(3)
        measurement = np.asarray(obstacle_measurement, dtype=float).reshape(3)
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(measurement)):
            raise ValueError("TVP updates must be finite 3-vectors")
        if control_time < 0.0 or measurement_time < 0.0 or measurement_time > control_time + 1e-12:
            raise ValueError("TVP times must satisfy 0 <= measurement_time <= control_time")
        # A short backward allowance prevents numerical tracking noise from
        # permanently skipping the closest point while progress remains monotone.
        start = max(0, self.progress_index - 4)
        distances = np.linalg.norm(self.reference_path[start:] - position, axis=1)
        nearest = start + int(np.argmin(distances))
        self.progress_index = max(self.progress_index, nearest)
        self.robot_position = position.copy()
        self.measurement = measurement.copy()
        self.control_time = float(control_time)
        self.observer.observe(measurement, float(measurement_time))

    def declare(self, model: Any, x: Any, u: Any, ca: Any) -> None:
        del x, u, ca
        self._obstacle_tvp = model.set_variable(
            "_tvp", "obstacle_position", shape=(3, 1)
        )
        self._obstacle_next_tvp = model.set_variable(
            "_tvp", "obstacle_next_position", shape=(3, 1)
        )
        self._robust_radius_tvp = model.set_variable(
            "_tvp", "obstacle_robust_radius", shape=(1, 1)
        )
        self._robust_radius_next_tvp = model.set_variable(
            "_tvp", "obstacle_next_robust_radius", shape=(1, 1)
        )
        self._reference_tvp = model.set_variable(
            "_tvp", "reference_state", shape=(8, 1)
        )
        self._gamma_tvp = model.set_variable("_tvp", "cbf_gamma", shape=(1, 1))
        # The gate variable is declared only for the opt-in one-step profile
        # so the frozen full-horizon baseline model stays byte-identical.
        if self.safety_mode == "cbf" and self.cbf_constraint_scope == "first_step":
            self._cbf_active_tvp = model.set_variable(
                "_tvp", "cbf_active", shape=(1, 1)
            )
        if self.safety_mode == "cbf" and self.terminal_safe_set:
            self._terminal_safe_active_tvp = model.set_variable(
                "_tvp", "terminal_safe_active", shape=(1, 1)
            )

    def prediction_at_stage(self, stage: int) -> ReferenceObstacleStage:
        """Return the exact TVPs assigned to ``stage`` by :meth:`configure`.

        Keeping this calculation independent of do-mpc makes the obstacle
        current/next indexing directly testable. Stage ``k`` represents times
        ``t+k*dt`` and ``t+(k+1)*dt`` respectively.
        """

        import numpy as np

        if not isinstance(stage, int) or not 0 <= stage <= self.horizon:
            raise ValueError("stage must be an integer in [0, horizon]")
        base_distance = self.arc_length[self.progress_index]
        target_distance = (
            self.arc_length[-1]
            if self.direct_target
            else min(
                self.arc_length[-1],
                base_distance
                + stage * self.dt * self.reference_speed * self.speed_scale,
            )
        )
        index = int(
            min(
                len(self.reference_path) - 1,
                np.searchsorted(self.arc_length, target_distance, side="left"),
            )
        )
        if self.temporary_subgoal is not None:
            displacement = self.temporary_subgoal - self.robot_position
            distance = float(np.linalg.norm(displacement))
            direction = displacement / distance if distance > 1e-9 else np.zeros(3)
            travel = min(
                distance,
                (stage + 1) * self.dt * self.reference_speed * self.speed_scale,
            )
            point = self.robot_position + travel * direction
            desired_velocity = (
                direction * self.reference_speed * self.speed_scale
                if distance > 1e-9
                else np.zeros(3)
            )
        else:
            point = self.reference_path[index]
            next_index = min(index + 1, len(self.reference_path) - 1)
            tangent = self.reference_path[next_index] - point
            tangent_norm = float(np.linalg.norm(tangent))
            desired_velocity = (
                tangent / tangent_norm * self.reference_speed * self.speed_scale
                if (
                    not self.direct_target
                    and tangent_norm > 1e-9
                    and index < len(self.reference_path) - 1
                )
                else np.zeros(3)
            )
        reference_state = np.concatenate([point, [0.0], desired_velocity, [0.0]])
        stage_age = (
            max(0.0, self.control_time - self.observer.timestamp) + stage * self.dt
        )
        next_age = stage_age + self.dt
        if self.prediction_mode in {"velocity", "velocity_tube"}:
            obstacle_position = self.observer.predict(
                self.observer.timestamp + stage_age
            )
            obstacle_next_position = self.observer.predict(
                self.observer.timestamp + next_age
            )
            tube_now = (
                self.uncertainty_at_age(stage_age)
                if self.prediction_mode == "velocity_tube"
                else 0.0
            )
            tube_next = (
                self.uncertainty_at_age(next_age)
                if self.prediction_mode == "velocity_tube"
                else 0.0
            )
            robust_radius = (
                self.base_combined_radius + self.clearance_margin + tube_now
            )
            robust_radius_next = (
                self.base_combined_radius + self.clearance_margin + tube_next
            )
        else:
            obstacle_position = tuple(float(value) for value in self.measurement)
            obstacle_next_position = obstacle_position
            robust_radius = self.base_combined_radius + self.clearance_margin
            robust_radius_next = robust_radius
            if self.dead_time_margin_mode == "speed_bound":
                # ZOCBF-style dead-time robustness: the held measurement can
                # be stale by ``stage_age``; any bounded obstacle motion stays
                # inside the inflated ball, so safety holds between samples.
                robust_radius += self.dead_time_speed_bound * stage_age
                robust_radius_next += self.dead_time_speed_bound * next_age
        return ReferenceObstacleStage(
            reference_state=tuple(float(value) for value in reference_state),
            obstacle_position=tuple(float(value) for value in obstacle_position),
            obstacle_next_position=tuple(
                float(value) for value in obstacle_next_position
            ),
            robust_radius=float(robust_radius),
            robust_radius_next=float(robust_radius_next),
        )

    def uncertainty_at_age(self, prediction_age: float) -> float:
        """Inflate unknown velocity before the first distinct sensor update."""

        velocity_bound = (
            self.tube.initial_velocity_error_bound
            if self.observer.updates < 2
            else self.tube.velocity_error_bound
        )
        return self.tube.inflation(
            prediction_age, velocity_error_bound=velocity_bound
        )

    def configure(self, model: Any, mpc: Any, x: Any, u: Any, ca: Any) -> None:
        import numpy as np

        if (
            self._obstacle_tvp is None
            or self._obstacle_next_tvp is None
            or self._robust_radius_tvp is None
            or self._robust_radius_next_tvp is None
            or self._reference_tvp is None
            or self._gamma_tvp is None
        ):
            raise RuntimeError("declare must run before configure")
        position = x[:3]
        if self.cbf_transition_mode == "command_velocity":
            next_position = position + self.dt * u[:3]
        elif self.cbf_transition_mode == "double_integrator":
            next_position = (
                position
                + self.dt * x[4:7]
                + 0.5 * self.dt**2 * u[:3]
            )
        else:
            next_position = position + self.dt * x[4:7]
        h_current = (
            ca.sumsqr(position - self._obstacle_tvp) - self._robust_radius_tvp**2
        )
        h_next = (
            ca.sumsqr(next_position - self._obstacle_next_tvp)
            - self._robust_radius_next_tvp**2
        )
        if self.safety_mode == "cbf":
            decay = (
                model.u["cbf_decay"]
                if "cbf_decay" in model.u.keys()
                else 1.0
            )
            cbf_expression = decay * (1.0 - self._gamma_tvp) * h_current - h_next
            if "cbf_slack" in model.u.keys():
                # Soft constraint: violation is bounded by the slack input,
                # which the L1 exact penalty keeps at zero when feasible.
                cbf_expression = cbf_expression - model.u["cbf_slack"]
            if self._cbf_active_tvp is not None:
                cbf_expression = self._cbf_active_tvp * cbf_expression
            mpc.set_nl_cons(
                "dynamic_obstacle_cbf",
                cbf_expression,
                ub=0.0,
            )
            if self._terminal_safe_active_tvp is not None:
                mpc.set_nl_cons(
                    "terminal_safe_set",
                    self._terminal_safe_active_tvp * (-h_next),
                    ub=0.0,
                )
        elif self.safety_mode == "distance":
            mpc.set_nl_cons("dynamic_obstacle_distance", -h_next, ub=0.0)
        template = mpc.get_tvp_template()

        def tvp_fun(t_now: float) -> Any:
            del t_now
            for stage in range(self.horizon + 1):
                prediction = self.prediction_at_stage(stage)
                template["_tvp", stage, "obstacle_position"] = (
                    np.asarray(prediction.obstacle_position).reshape(3, 1)
                )
                template["_tvp", stage, "obstacle_next_position"] = (
                    np.asarray(prediction.obstacle_next_position).reshape(3, 1)
                )
                template["_tvp", stage, "obstacle_robust_radius"] = (
                    prediction.robust_radius
                )
                template["_tvp", stage, "obstacle_next_robust_radius"] = (
                    prediction.robust_radius_next
                )
                template["_tvp", stage, "reference_state"] = (
                    np.asarray(prediction.reference_state).reshape(8, 1)
                )
                template["_tvp", stage, "cbf_gamma"] = self.gamma
                if self._cbf_active_tvp is not None:
                    template["_tvp", stage, "cbf_active"] = (
                        self.cbf_active_at_stage(stage)
                    )
                if self._terminal_safe_active_tvp is not None:
                    template["_tvp", stage, "terminal_safe_active"] = (
                        self.terminal_safe_active_at_stage(stage)
                    )
            return template

        mpc.set_tvp_fun(tvp_fun)


def run_smooth_dynamic_demo(
    config: SmoothDynamicConfig | None = None,
) -> SmoothDynamicResult:
    """Run one continuous-reference MPC-CBF variant and write raw evidence."""

    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    import panda_gym  # noqa: F401

    from .controller import (
        PaperMPCConfig,
        build_mpc_controller,
        discrete_state_transition,
    )
    from .async_gamma import AtomicGammaStore, GammaUpdate, GammaUpdateQueue
    from .demo import paper_control_to_safe_panda_action
    from .formal_safety import (
        FormalDiscreteSafetyFilter,
        FormalObstacle,
        FormalSafetyConfig,
    )
    from .obstacle_prediction import UncertaintyTubeConfig
    from .safety_reflex import (
        OperationalSpaceSafetyReflex,
        ReflexObstacle,
        SafetyReflexConfig,
    )
    from .safety_scheduler import (
        ContextAwareSafetyConfig,
        ContextAwareSafetyScheduler,
        SafetyProfile,
        SafetyProfileLifecycle,
        constant_velocity_ttc,
        feedback_has_causal_opportunity,
        feedback_update_deadline,
    )
    from .solver import (
        FeasibilityPolicy,
        IpoptConfig,
        constraint_violation_profile_from_mpc,
        diagnostics_from_do_mpc,
        safe_control_or_none,
    )
    from .safe_panda import CartesianVelocityEstimator, simulator_calibration_sample
    from .smoothness import (
        calculate_smoothness_metrics,
        make_reference_bspline,
        make_visual_smoothing_spline,
    )

    cfg = config or SmoothDynamicConfig()
    rng = np.random.default_rng(cfg.seed)
    wrapped_env = gym.make(
        "PandaReachSafe-v3", render_mode="rgb_array", renderer="Tiny"
    )
    env = wrapped_env.unwrapped
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Any] = []
    positions: list[Any] = []
    true_obstacles: list[Any] = []
    measured_obstacles: list[Any] = []
    controls: list[Any] = []
    nominal_controls: list[Any] = []
    true_clearances: list[float] = []
    measured_clearances: list[float] = []
    true_barriers: list[float] = []
    true_cbf_residuals: list[float] = []
    goal_distances: list[float] = []
    solve_times: list[float] = []
    predicted_ttc_values: list[float] = []
    context_profile_trace: list[dict[str, Any]] = []
    runtime_trace: list[dict[str, Any]] = []
    model_transition_errors: list[float] = []
    action_tracking_errors: list[float] = []
    solver_audits: list[dict[str, Any]] = []
    solver_failures = 0
    braking_fallbacks = 0
    solver_rejections = 0
    solver_termination_counts: dict[str, int] = {}
    emergency_fallbacks = 0
    constraint_violations: list[float] = []
    sensor_update_steps: list[int] = [0]
    formal_filter_audits: list[dict[str, Any]] = []

    try:
        observation, _ = env.reset(seed=cfg.seed)
        start = np.asarray(observation["achieved_goal"], dtype=float)
        goal = start + np.asarray(cfg.goal_offset, dtype=float)
        initial_goal_distance = float(np.linalg.norm(start - goal))
        true_obstacle = start + np.asarray(cfg.obstacle_start_offset, dtype=float)
        obstacle_velocity = np.asarray(cfg.obstacle_velocity, dtype=float)
        hidden_obstacle = np.array([2.0, 2.0, -1.0])
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        task = env.task
        task.goal = goal.copy()
        task.unsafe_region_radius = cfg.obstacle_radius
        task.unsafe_state_1_pos = true_obstacle.copy()
        task.unsafe_state_2_pos = hidden_obstacle.copy()
        if cfg.save_animation:
            # The panda-gym scene bakes the unsafe-region spheres at a fixed
            # default radius; setting the attribute above only feeds task
            # logic, not the visual mesh. Recreate the visible spheres at the
            # true obstacle radius so the animation is not misleading. Guarded
            # to the rendering path — the headless benchmark never touches it.
            for body_name, body_position in (
                ("unsafe_region_1", true_obstacle),
                ("unsafe_region_2", hidden_obstacle),
            ):
                body_index = env.sim._bodies_idx.get(body_name)
                if body_index is not None:
                    env.sim.physics_client.removeBody(body_index)
                env.sim.create_sphere(
                    body_name=body_name,
                    radius=float(cfg.obstacle_radius),
                    mass=0.0,
                    ghost=True,
                    position=np.asarray(body_position, dtype=float),
                    rgba_color=np.array([0.9, 0.1, 0.1, 0.3]),
                )
        env.sim.set_base_pose("target", goal, quaternion)
        env.sim.set_base_pose("unsafe_region_1", true_obstacle, quaternion)
        env.sim.set_base_pose("unsafe_region_2", hidden_obstacle, quaternion)
        observation = env._get_obs()

        combined_radius = cfg.obstacle_radius + cfg.collision_radius
        initial_true_barrier = float(
            np.dot(start - true_obstacle, start - true_obstacle)
            - combined_radius**2
        )
        true_barriers.append(initial_true_barrier)
        if cfg.reference_mode in {"straight", "direct_target"}:
            route_points = np.asarray([start, goal])
            reference_path = np.linspace(start, goal, 600)
        else:
            route_points, reference_path = build_reference_route(
                start,
                goal,
                obstacle_velocity=obstacle_velocity,
                combined_radius=combined_radius,
                route_margin=cfg.route_margin,
                profile=cfg.reference_route_profile,
                waypoint_offsets=cfg.avoidance_waypoint_offsets,
            )
        measurement = true_obstacle + sample_obstacle_measurement_noise(
            rng,
            mode=cfg.measurement_noise_mode,
            sigma=cfg.measurement_noise_sigma,
            error_bound=cfg.measurement_error_bound,
        )
        dynamics_mode = (
            cfg.cbf_transition_mode
            if cfg.cbf_transition_mode in {"double_integrator", "paper_increment"}
            else "paper_state"
        )
        mpc_config = PaperMPCConfig(
            linear_delta_u_weight=cfg.delta_u_weight,
            position_q_weights=cfg.position_q_weights,
            linear_jerk_weight=cfg.jerk_weight,
            optimal_decay_weight=cfg.optimal_decay_weight,
            optimal_decay_lower=cfg.optimal_decay_lower,
            cbf_slack_weight=cfg.cbf_slack_weight,
            target_tvp_name="reference_state",
            dynamics_mode=dynamics_mode,
        )
        tvp = ReferenceObstacleTVP(
            reference_path,
            measurement,
            reference_speed=cfg.reference_speed,
            obstacle_radius=cfg.obstacle_radius,
            collision_radius=cfg.collision_radius,
            gamma=cfg.gamma,
            dt=mpc_config.dt,
            horizon=mpc_config.horizon,
            safety_mode=cfg.safety_mode,
            prediction_mode=cfg.prediction_mode,
            tube_config=UncertaintyTubeConfig(
                measurement_sigma=cfg.measurement_noise_sigma,
                confidence_multiplier=cfg.confidence_multiplier,
                measurement_error_bound=(
                    cfg.measurement_error_bound
                    if cfg.measurement_noise_mode == "bounded_ball"
                    else None
                ),
                velocity_error_bound=cfg.velocity_error_bound,
                initial_velocity_error_bound=cfg.initial_velocity_error_bound,
                model_error_growth=cfg.model_error_growth,
                max_relative_speed=cfg.max_relative_speed,
                total_latency=cfg.total_latency,
                sensor_period=cfg.sensor_period,
                control_period=mpc_config.dt,
                obstacle_acceleration_bound=cfg.obstacle_acceleration_bound,
                robot_transition_error_bound=(
                    cfg.formal_robot_transition_error_bound
                    if cfg.formal_safety_filter_enabled
                    else 0.0
                ),
                sampled_data_margin_enabled=cfg.sampled_data_margin_enabled,
            ),
            velocity_filter=cfg.velocity_filter,
            direct_target=cfg.reference_mode == "direct_target",
            cbf_transition_mode=cfg.cbf_transition_mode,
            cbf_constraint_scope=cfg.cbf_constraint_scope,
            terminal_safe_set=cfg.terminal_safe_set_enabled,
            dead_time_margin_mode=cfg.dead_time_margin_mode,
            dead_time_speed_bound=cfg.dead_time_obstacle_speed_bound,
            gamma_upper=cfg.gamma_upper,
        )
        if cfg.known_obstacle_velocity:
            tvp.observer.velocity = tuple(float(value) for value in obstacle_velocity)
        formal_filter = (
            FormalDiscreteSafetyFilter(
                FormalSafetyConfig(
                    dt=mpc_config.dt,
                    gamma=cfg.gamma,
                    speed_limit=cfg.formal_filter_speed_limit,
                    measurement_error_bound=cfg.measurement_error_bound,
                    obstacle_velocity_error_bound=(
                        cfg.formal_obstacle_velocity_error_bound
                    ),
                    obstacle_acceleration_bound=cfg.obstacle_acceleration_bound,
                    robot_transition_error_bound=(
                        cfg.formal_robot_transition_error_bound
                    ),
                    obstacle_speed_bound=cfg.formal_obstacle_speed_bound,
                )
            )
            if cfg.formal_safety_filter_enabled
            else None
        )
        _, mpc = build_mpc_controller(
            mpc_config,
            model_builders=(tvp.declare,),
            constraint_builders=(tvp.configure,),
            nlpsol_options=(
                IpoptConfig.reference_defaults()
                if cfg.ipopt_profile == "reference_defaults"
                else IpoptConfig(
                    constraint_violation_tolerance=(
                        cfg.solver_max_constraint_violation
                        if cfg.solver_max_constraint_violation > 0.0
                        else 1e-12
                    ),
                    max_cpu_time=cfg.solver_max_cpu_time,
                )
            ).casadi_options(),
        )
        reflex = OperationalSpaceSafetyReflex(
            SafetyReflexConfig(
                dt=mpc_config.dt,
                lookahead_steps=cfg.reflex_lookahead_steps,
                cbf_alpha=cfg.reflex_alpha,
                speed_limit=mpc_config.linear_input_limit,
                uncertainty_growth_per_second=(
                    cfg.velocity_error_bound + cfg.model_error_growth
                    if cfg.prediction_mode == "velocity_tube"
                    else 0.0
                ),
                uncertainty_acceleration_bound=(
                    cfg.obstacle_acceleration_bound
                    if cfg.prediction_mode == "velocity_tube"
                    else 0.0
                ),
                backup_selection=cfg.reflex_backup_selection,
                committed_backup_enabled=cfg.reflex_committed_backup_enabled,
                committed_backup_steps=cfg.reflex_committed_backup_steps,
                barrier_mode=cfg.reflex_barrier_mode,
                side_latch_enabled=cfg.reflex_side_latch_enabled,
                side_latch_steps=cfg.reflex_side_latch_steps,
                side_release_clearance=cfg.reflex_side_release_clearance,
                side_switch_hysteresis=cfg.reflex_side_switch_hysteresis,
                policy_library_enabled=cfg.reflex_policy_library_enabled,
                policy_speed_scales=cfg.reflex_policy_speed_scales,
                recovery_clearance_slack=cfg.reflex_recovery_clearance_slack,
                tangential_subgoal_enabled=(
                    cfg.reflex_tangential_subgoal_enabled
                ),
                tangential_subgoal_distance=(
                    cfg.reflex_tangential_subgoal_distance
                ),
                dpcbf_safety_scale=cfg.reflex_dpcbf_safety_scale,
                dpcbf_lambda_gain=cfg.reflex_dpcbf_lambda_gain,
                dpcbf_mu_gain=cfg.reflex_dpcbf_mu_gain,
            )
        )
        previous_control = np.zeros(4)
        previous_increment = np.zeros(4)
        robot_velocity_estimator = CartesianVelocityEstimator(
            filter_weight=cfg.robot_velocity_filter,
            maximum_speed=cfg.robot_velocity_maximum,
        )
        robot_velocity_estimator.reset(start)
        observed_velocity = np.zeros(3)
        double_integrator_velocity = np.zeros(4)
        initial_state = np.concatenate([start, [0.0], previous_control])
        if mpc_config.uses_jerk_state:
            initial_state = np.concatenate([initial_state, previous_increment])
        mpc.x0 = initial_state.reshape(-1, 1)
        mpc.set_initial_guess()
        reached_goal = False
        collision = False
        truncated = False
        next_sensor_time = cfg.sensor_period
        last_measurement_time = 0.0
        schedule_index = 0
        prediction_schedule_index = 0
        applied_prediction_mode_updates: list[dict[str, Any]] = []
        gamma_queue = GammaUpdateQueue()
        gamma_store = AtomicGammaStore(
            cfg.gamma, clock_skew_tolerance=0.0, gamma_upper=cfg.gamma_upper
        )
        applied_gamma_updates: list[dict[str, float]] = []
        rejected_gamma_updates = 0
        gamma_trace: list[float] = []
        reflex_audits: list[dict[str, Any]] = []
        optimal_decay_trace: list[float] = []
        avoidance_onset_time: float | None = None
        safety_config = ContextAwareSafetyConfig(
            exit_ttc_hysteresis=cfg.safety_exit_ttc_hysteresis,
            clear_hold_time=cfg.safety_clear_hold_time,
            recovery_duration=cfg.safety_recovery_duration,
            recovery_enabled=cfg.safety_profile_recovery_enabled,
        )
        safety_scheduler = ContextAwareSafetyScheduler(safety_config)
        safety_lifecycle = SafetyProfileLifecycle(cfg.gamma, safety_config)
        last_lifecycle_state = safety_lifecycle.state
        safety_profile_transitions = 0
        hazard_clear_elapsed = 0.0
        feasibility_policy = FeasibilityPolicy(
            max_constraint_violation=cfg.solver_max_constraint_violation
        )
        provisional_index = 0
        last_solver_feasible = True
        ttc_feedback_trigger_time: float | None = None
        ttc_feedback_available_time: float | None = None
        ttc_feedback_published = False
        ttc_feedback_causal_opportunity = False
        ttc_feedback_valid_until: float | None = None
        feedback_updates_rejected_late = 0
        infeasible_stage_counts: dict[int, int] = {}

        for step_index in range(cfg.max_steps):
            elapsed = step_index * mpc_config.dt
            while (
                provisional_index < len(cfg.provisional_feedback_times)
                and elapsed + 1e-12 >= cfg.provisional_feedback_times[provisional_index]
            ):
                safety_lifecycle.activate_provisional()
                provisional_index += 1
            if (
                ttc_feedback_available_time is not None
                and not ttc_feedback_published
                and elapsed + 1e-12 >= ttc_feedback_available_time
            ):
                if (
                    ttc_feedback_causal_opportunity
                    or not cfg.reject_feedback_without_causal_opportunity
                ):
                    update_created_at = float(ttc_feedback_trigger_time)
                    update_valid_until = float(ttc_feedback_valid_until)
                    if not ttc_feedback_causal_opportunity:
                        # Explicit legacy ablation only: reproduce the former
                        # arrival-time TTL bug so its effect can be measured.
                        update_created_at = elapsed
                        update_valid_until = elapsed + cfg.gamma_update_ttl
                    gamma_queue.publish(
                        GammaUpdate(
                            gamma=float(cfg.feedback_gamma),
                            version=len(cfg.gamma_schedule) + 1,
                            created_at=update_created_at,
                            valid_until=update_valid_until,
                            reason="ttc_triggered_online_feedback",
                            source="benchmark_replay",
                        )
                    )
                else:
                    feedback_updates_rejected_late += 1
                ttc_feedback_published = True
            while (
                prediction_schedule_index < len(cfg.prediction_mode_schedule)
                and elapsed + 1e-12
                >= cfg.prediction_mode_schedule[prediction_schedule_index][0]
            ):
                switch_time, new_mode = cfg.prediction_mode_schedule[
                    prediction_schedule_index
                ]
                request_time = (
                    cfg.prediction_mode_schedule_request_times[
                        prediction_schedule_index
                    ]
                    if cfg.prediction_mode_schedule_request_times
                    else switch_time
                )
                tvp.update_prediction_mode(new_mode)
                applied_prediction_mode_updates.append(
                    {
                        "requested_time": request_time,
                        "applied_time": elapsed,
                        "mode": new_mode,
                    }
                )
                prediction_schedule_index += 1
            while (
                schedule_index < len(cfg.gamma_schedule)
                and elapsed + 1e-12 >= cfg.gamma_schedule[schedule_index][0]
            ):
                scheduled_time, scheduled_gamma = cfg.gamma_schedule[schedule_index]
                request_time = (
                    cfg.gamma_schedule_request_times[schedule_index]
                    if cfg.gamma_schedule_request_times
                    else scheduled_time
                )
                gamma_queue.publish(
                    GammaUpdate(
                        gamma=scheduled_gamma,
                        version=schedule_index + 1,
                        created_at=request_time,
                        valid_until=request_time + cfg.gamma_update_ttl,
                        reason="scheduled_online_feedback",
                        source="benchmark_replay",
                    )
                )
                schedule_index += 1
            gamma_audit = gamma_store.apply_pending(gamma_queue, now=elapsed)
            rejected_gamma_updates += (
                gamma_audit.rejected_expired
                + gamma_audit.rejected_old_version
                + gamma_audit.rejected_future
            )
            feedback_updates_rejected_late += gamma_audit.rejected_expired
            if gamma_audit.applied is not None:
                tvp.update_gamma(gamma_audit.applied.gamma)
                safety_lifecycle.accept_validated_update(
                    gamma_audit.applied.gamma
                )
                applied_gamma_updates.append(
                    {
                        "scheduled_time": gamma_audit.applied.created_at,
                        "applied_time": elapsed,
                        "gamma": gamma_audit.applied.gamma,
                        "version": gamma_audit.applied.version,
                    }
                )
            if elapsed + 1e-12 >= next_sensor_time:
                measurement = true_obstacle + sample_obstacle_measurement_noise(
                    rng,
                    mode=cfg.measurement_noise_mode,
                    sigma=cfg.measurement_noise_sigma,
                    error_bound=cfg.measurement_error_bound,
                )
                sensor_update_steps.append(step_index)
                last_measurement_time = elapsed
                next_sensor_time += cfg.sensor_period

            position = np.asarray(observation["achieved_goal"], dtype=float)
            tvp.update(
                position,
                measurement,
                control_time=elapsed,
                measurement_time=last_measurement_time,
            )
            if cfg.known_obstacle_velocity:
                # Preserve the declared exact-velocity contract after the
                # observer consumes each noisy position sample.
                tvp.observer.velocity = tuple(
                    float(value) for value in obstacle_velocity
                )
            estimated_position = np.asarray(tvp.observer.predict(elapsed), dtype=float)
            estimated_velocity = np.asarray(
                obstacle_velocity
                if cfg.known_obstacle_velocity
                else tvp.observer.velocity
                if cfg.prediction_mode in {"velocity", "velocity_tube"}
                else (0.0, 0.0, 0.0),
                dtype=float,
            )
            prediction_age = max(0.0, elapsed - tvp.observer.timestamp)
            tube_uncertainty = (
                tvp.uncertainty_at_age(prediction_age)
                if cfg.prediction_mode == "velocity_tube"
                else 0.0
            )
            robust_predicted_clearance = float(
                np.linalg.norm(position - estimated_position)
                - combined_radius
                - tube_uncertainty
            )
            predicted_ttc = constant_velocity_ttc(
                position - estimated_position,
                (
                    observed_velocity
                    if cfg.robot_velocity_feedback_enabled
                    else previous_control[:3]
                )
                - estimated_velocity,
                combined_radius,
            )
            if predicted_ttc is not None:
                predicted_ttc_values.append(predicted_ttc)
            if (
                cfg.feedback_ttc_threshold is not None
                and ttc_feedback_trigger_time is None
                and predicted_ttc is not None
                and predicted_ttc <= cfg.feedback_ttc_threshold
            ):
                ttc_feedback_trigger_time = elapsed
                ttc_feedback_available_time = elapsed + cfg.feedback_response_latency
                ttc_feedback_causal_opportunity = feedback_has_causal_opportunity(
                    predicted_ttc,
                    cfg.feedback_response_latency,
                    reaction_margin=cfg.feedback_reaction_margin,
                )
                ttc_feedback_valid_until = feedback_update_deadline(
                    elapsed,
                    predicted_ttc,
                    cfg.gamma_update_ttl,
                    reaction_margin=cfg.feedback_reaction_margin,
                )
                safety_lifecycle.activate_provisional()

            lifecycle_was_active = safety_lifecycle.active
            lifecycle_profile = safety_lifecycle.step(
                predicted_ttc=predicted_ttc,
                dt=mpc_config.dt,
                solver_feasible=last_solver_feasible,
                robust_clearance=robust_predicted_clearance,
            )
            if safety_lifecycle.state is not last_lifecycle_state:
                safety_profile_transitions += 1
                last_lifecycle_state = safety_lifecycle.state
            hazard_is_clear = (
                last_solver_feasible
                and (
                    predicted_ttc is None
                    or predicted_ttc
                    > safety_config.cautious_ttc
                    + safety_config.exit_ttc_hysteresis
                )
                and robust_predicted_clearance
                > safety_config.cautious_clearance_margin
                + safety_config.emergency_clearance_margin
            )
            hazard_clear_elapsed = (
                hazard_clear_elapsed + mpc_config.dt
                if safety_lifecycle.active and hazard_is_clear
                else 0.0
            )
            if cfg.context_safety_enabled:
                context_profile = safety_scheduler.select(
                    predicted_ttc=predicted_ttc,
                    requested_safety_level=cfg.requested_safety_level,
                    solver_feasible=last_solver_feasible,
                )
                profile = SafetyProfile(
                    gamma=min(context_profile.gamma, lifecycle_profile.gamma),
                    clearance_margin=max(
                        context_profile.clearance_margin,
                        lifecycle_profile.clearance_margin,
                    ),
                    speed_scale=min(
                        context_profile.speed_scale,
                        lifecycle_profile.speed_scale,
                    ),
                    emergency=context_profile.emergency
                    or lifecycle_profile.emergency,
                    reason=(
                        f"context:{context_profile.reason};"
                        f"lifecycle:{lifecycle_profile.reason}"
                    ),
                )
            else:
                profile = lifecycle_profile
            if cfg.context_safety_enabled or lifecycle_was_active:
                tvp.update_safety_profile(
                    gamma=profile.gamma,
                    clearance_margin=profile.clearance_margin,
                    speed_scale=profile.speed_scale,
                )
                context_profile_trace.append(
                    {
                        "step": step_index,
                        "time": elapsed,
                        "predicted_ttc": predicted_ttc,
                        "lifecycle_state": safety_lifecycle.state.value,
                        **asdict(profile),
                    }
                )
            if cfg.cbf_transition_mode in {"double_integrator", "paper_increment"}:
                feedback_velocity = double_integrator_velocity[:3].copy()
                feedback_yaw_velocity = float(double_integrator_velocity[3])
            else:
                feedback_velocity = (
                    observed_velocity
                    if cfg.robot_velocity_feedback_enabled
                    else previous_control[:3]
                )
                feedback_yaw_velocity = float(previous_control[3])
            measured_state = np.concatenate(
                [position, [0.0], feedback_velocity, [feedback_yaw_velocity]]
            )
            if mpc_config.uses_jerk_state:
                measured_state = np.concatenate([measured_state, previous_increment])
            started = perf_counter()
            raw_candidate = np.asarray(
                mpc.make_step(measured_state.reshape(-1, 1)), dtype=float
            ).reshape(-1)
            measured_solve_time = perf_counter() - started
            solve_times.append(measured_solve_time)
            expected_dimension = (
                4
                + int(mpc_config.uses_optimal_decay)
                + int(mpc_config.uses_cbf_slack)
            )
            diagnostics = diagnostics_from_do_mpc(
                mpc, measured_solve_time=measured_solve_time
            )
            violation_profile = constraint_violation_profile_from_mpc(
                mpc, tolerance=cfg.solver_max_constraint_violation
            )
            deadline_missed = measured_solve_time > cfg.control_deadline
            accepted = safe_control_or_none(
                raw_candidate,
                diagnostics,
                policy=feasibility_policy,
                expected_dimension=expected_dimension,
            )
            if cfg.reject_deadline_miss and deadline_missed:
                accepted = None
            if not diagnostics.solver_success:
                solver_failures += 1
            termination_name = diagnostics.termination.value
            solver_termination_counts[termination_name] = (
                solver_termination_counts.get(termination_name, 0) + 1
            )
            if diagnostics.constraint_violation is not None:
                constraint_violations.append(diagnostics.constraint_violation)
            rejected_candidate = accepted is None
            last_solver_feasible = not rejected_candidate
            if rejected_candidate:
                solver_rejections += 1
                emergency_fallbacks += 1
                if cfg.solver_fallback_mode == "brake":
                    fallback_command, braking_engaged = braking_fallback_command(
                        position,
                        feedback_velocity,
                        tvp.prediction_at_stage(0),
                        gamma=tvp.gamma,
                        dt=mpc_config.dt,
                        braking_decay=cfg.fallback_braking_decay,
                        cbf_transition_mode=cfg.cbf_transition_mode,
                        input_limit=mpc_config.linear_input_limit,
                    )
                    braking_fallbacks += int(braking_engaged)
                    candidate = np.asarray(fallback_command, dtype=float)
                else:
                    candidate = np.zeros(4, dtype=float)
            else:
                candidate = np.asarray(accepted[:4], dtype=float)
            for violation in violation_profile.violations:
                if violation.stage is not None:
                    infeasible_stage_counts[violation.stage] = (
                        infeasible_stage_counts.get(violation.stage, 0) + 1
                    )
            solver_audits.append(
                {
                    "step": step_index,
                    "time": elapsed,
                    "termination": diagnostics.termination.value,
                    "return_status": diagnostics.raw_status,
                    "solver_success": diagnostics.solver_success,
                    "constraint_violation": diagnostics.constraint_violation,
                    "iterations": diagnostics.iterations,
                    "solve_time": measured_solve_time,
                    "deadline_missed": deadline_missed,
                    "candidate_rejected": rejected_candidate,
                    "constraint_stage_layout_supported": (
                        violation_profile.stage_layout_supported
                    ),
                    "constraint_violations": [
                        asdict(item) for item in violation_profile.violations
                    ],
                }
            )
            optimal_decay_trace.append(
                float(raw_candidate[4])
                if mpc_config.uses_optimal_decay
                and raw_candidate.shape == (expected_dimension,)
                and np.isfinite(raw_candidate[4])
                and not rejected_candidate
                else 1.0
            )
            nominal_candidate = candidate.copy()
            reflex_intervened = False
            reflex_backup_used = False
            if (
                cfg.cbf_transition_mode
                not in {"double_integrator", "paper_increment"}
                and (
                    cfg.safety_reflex_enabled
                    or safety_lifecycle.requires_reflex
                    or rejected_candidate
                )
                and cfg.safety_mode != "none"
            ):
                reflex_result = reflex.gate(
                    position,
                    candidate[:3],
                    (
                        ReflexObstacle(
                            tuple(float(value) for value in estimated_position),
                            tuple(float(value) for value in estimated_velocity),
                            combined_radius + tvp.clearance_margin,
                            tube_uncertainty,
                            "moving_obstacle",
                        ),
                    ),
                    goal_position=goal,
                )
                if cfg.reflex_tangential_subgoal_enabled:
                    tvp.set_temporary_subgoal(reflex_result.temporary_subgoal)
                candidate[:3] = reflex_result.velocity
                reflex_intervened = reflex_result.intervened
                reflex_backup_used = reflex_result.backup_used
                if reflex_result.intervened:
                    reflex_audits.append(
                        {
                            "step": step_index,
                            "time": elapsed,
                            "backup_used": reflex_result.backup_used,
                            "reason": reflex_result.reason,
                            "nominal_minimum_clearance": reflex_result.nominal_minimum_clearance,
                            "filtered_minimum_clearance": reflex_result.filtered_minimum_clearance,
                            "maximum_cbf_violation": reflex_result.maximum_cbf_violation,
                            "selected_policy": reflex_result.selected_policy,
                            "avoidance_side": reflex_result.avoidance_side,
                            "side_switched": reflex_result.side_switched,
                            "temporary_subgoal": reflex_result.temporary_subgoal,
                            "robust_recovery": reflex_result.robust_recovery,
                        }
                    )
            formal_filter_result = None
            if formal_filter is not None:
                formal_filter_result = formal_filter.filter(
                    position,
                    candidate[:3],
                    FormalObstacle(
                        tuple(float(value) for value in estimated_position),
                        tuple(float(value) for value in estimated_velocity),
                        combined_radius,
                    ),
                )
                candidate[:3] = formal_filter_result.velocity
                formal_filter_audits.append(
                    {
                        "step": step_index,
                        "time": elapsed,
                        "intervened": formal_filter_result.intervened,
                        "one_step_certified": (
                            formal_filter_result.one_step_certified
                        ),
                        "terminal_backup_certified": (
                            formal_filter_result.terminal_backup_certified
                        ),
                        "robust_cbf_residual": (
                            formal_filter_result.robust_cbf_residual
                        ),
                        "backup_authority_margin": (
                            formal_filter_result.backup_authority_margin
                        ),
                    }
                )
                if (
                    cfg.abort_on_formal_certificate_failure
                    and (
                        not formal_filter_result.one_step_certified
                        or not formal_filter_result.terminal_backup_certified
                    )
                ):
                    raise RuntimeError(
                        "formal safety certificate failed; refusing to send action"
                    )
            pre_step_position = position.copy()
            pre_step_obstacle = true_obstacle.copy()
            applied_gamma = float(tvp.gamma)
            applied_decay = float(optimal_decay_trace[-1])
            if cfg.cbf_transition_mode in {"double_integrator", "paper_increment"}:
                model_state = np.concatenate(
                    [pre_step_position, [0.0], double_integrator_velocity]
                )
                predicted_state = np.asarray(
                    discrete_state_transition(
                        model_state,
                        candidate,
                        dt=mpc_config.dt,
                        mode=cfg.cbf_transition_mode,
                    ),
                    dtype=float,
                )
                action_velocity = (
                    predicted_state[:3] - pre_step_position
                ) / mpc_config.dt
                model_velocity = action_velocity.copy()
                action_control = np.concatenate(
                    [action_velocity, [candidate[3]]]
                )
            else:
                predicted_state = None
                model_velocity = feedback_velocity.copy()
                action_control = candidate
            action = paper_control_to_safe_panda_action(
                action_control, env.action_space.shape[0],
                linear_input_limit=(
                    cfg.formal_filter_speed_limit
                    if formal_filter is not None
                    else (
                        cfg.robot_velocity_maximum
                        if cfg.cbf_transition_mode
                        in {"double_integrator", "paper_increment"}
                        else mpc_config.linear_input_limit
                    )
                ),
                dt=mpc_config.dt,
            )
            observation, _, terminated, truncated, _ = env.step(action)
            previous_increment = candidate - previous_control
            previous_control = candidate
            if predicted_state is not None:
                double_integrator_velocity = predicted_state[4:8].copy()

            true_obstacle = true_obstacle + obstacle_velocity * mpc_config.dt
            env.sim.set_base_pose("unsafe_region_1", true_obstacle, quaternion)
            task.unsafe_state_1_pos = true_obstacle.copy()
            position = np.asarray(observation["achieved_goal"], dtype=float)
            observed_velocity = np.asarray(
                robot_velocity_estimator.update(position, mpc_config.dt),
                dtype=float,
            )
            calibration = simulator_calibration_sample(
                pre_step_position,
                position,
                model_velocity,
                action_control[:3],
                mpc_config.dt,
            )
            model_transition_errors.append(calibration.model_transition_error)
            action_tracking_errors.append(calibration.action_tracking_error)
            true_clearance = float(
                np.linalg.norm(position - true_obstacle) - combined_radius
            )
            current_true_barrier = float(
                np.dot(
                    pre_step_position - pre_step_obstacle,
                    pre_step_position - pre_step_obstacle,
                )
                - combined_radius**2
            )
            next_true_barrier = float(
                np.dot(position - true_obstacle, position - true_obstacle)
                - combined_radius**2
            )
            true_cbf_residual = float(
                next_true_barrier
                - applied_decay * (1.0 - applied_gamma) * current_true_barrier
            )
            measured_clearance = float(
                np.linalg.norm(position - measurement) - combined_radius
            )
            goal_distance = float(np.linalg.norm(position - goal))
            elapsed_after_step = (step_index + 1) * mpc_config.dt
            net_progress = initial_goal_distance - goal_distance
            if (
                avoidance_onset_time is None
                and abs(float(position[0] - start[0])) >= cfg.avoidance_onset_threshold
            ):
                avoidance_onset_time = (step_index + 1) * mpc_config.dt
            positions.append(position.copy())
            true_obstacles.append(true_obstacle.copy())
            measured_obstacles.append(measurement.copy())
            controls.append(candidate.copy())
            nominal_controls.append(nominal_candidate)
            true_clearances.append(true_clearance)
            measured_clearances.append(measured_clearance)
            true_barriers.append(next_true_barrier)
            if cfg.safety_mode == "cbf":
                true_cbf_residuals.append(true_cbf_residual)
            goal_distances.append(goal_distance)
            gamma_trace.append(tvp.gamma)
            runtime_trace.append(
                {
                    "step": step_index,
                    "time": elapsed,
                    "lifecycle_state": safety_lifecycle.state.value,
                    "gamma": tvp.gamma,
                    "clearance_margin": tvp.clearance_margin,
                    "speed_scale": tvp.speed_scale,
                    "predicted_ttc": predicted_ttc,
                    "measurement_age": prediction_age,
                    "tube_uncertainty": tube_uncertainty,
                    "robust_predicted_clearance": robust_predicted_clearance,
                    "estimated_obstacle_position": estimated_position.tolist(),
                    "estimated_obstacle_velocity": estimated_velocity.tolist(),
                    "hazard_clear_elapsed": hazard_clear_elapsed,
                    "observed_velocity": observed_velocity.tolist(),
                    "commanded_velocity": action_control[:3].tolist(),
                    "mpc_input": candidate[:4].tolist(),
                    "true_barrier": next_true_barrier,
                    "true_cbf_residual": (
                        true_cbf_residual if cfg.safety_mode == "cbf" else None
                    ),
                    "applied_cbf_decay": applied_decay,
                    "goal_distance": goal_distance,
                    "net_goal_progress": net_progress,
                    "mean_goal_progress_rate": net_progress / elapsed_after_step,
                    "reflex_intervened": reflex_intervened,
                    "reflex_backup_used": reflex_backup_used,
                    "formal_filter_intervened": bool(
                        formal_filter_result is not None
                        and formal_filter_result.intervened
                    ),
                    "formal_filter_one_step_certified": (
                        formal_filter_result.one_step_certified
                        if formal_filter_result is not None
                        else None
                    ),
                    "formal_terminal_backup_certified": (
                        formal_filter_result.terminal_backup_certified
                        if formal_filter_result is not None
                        else None
                    ),
                    "reflex_avoidance_side": reflex.latched_side,
                    "reflex_side_switches": reflex.side_switches,
                    "temporary_subgoal": (
                        tvp.temporary_subgoal.tolist()
                        if tvp.temporary_subgoal is not None
                        else None
                    ),
                    "solver_feasible": last_solver_feasible,
                }
            )
            if cfg.save_animation and step_index % cfg.render_stride == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))
            collision = collision or true_clearance < 0.0
            reached_goal = bool(terminated) or goal_distance < 0.05
            if reached_goal or collision or truncated:
                break

        if len(positions) < 4:
            raise RuntimeError("experiment produced too few samples")
        raw_positions = np.asarray(positions)
        true_array = np.asarray(true_obstacles)
        measured_array = np.asarray(measured_obstacles)
        controls_array = np.asarray(controls)
        smoothness = calculate_smoothness_metrics(raw_positions, mpc_config.dt)
        visual_spline = (
            make_visual_smoothing_spline(raw_positions)
            if cfg.save_plots
            else np.empty((0, 3))
        )
        time = np.arange(1, len(raw_positions) + 1) * mpc_config.dt

        if cfg.save_plots:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
            axes[0].plot(
                raw_positions[:, 0], raw_positions[:, 1], color="steelblue",
                alpha=0.4, linewidth=1.0, label="raw trajectory (safety truth)",
            )
            axes[0].plot(
                visual_spline[:, 0], visual_spline[:, 1], color="navy",
                linewidth=2.0, label="visual smoothing spline",
            )
            axes[0].plot(
                reference_path[:, 0], reference_path[:, 1], "--", color="green",
                linewidth=1.2, label="continuous B-spline reference",
            )
            axes[0].plot(true_array[:, 0], true_array[:, 1], "r--", label="obstacle")
            axes[0].scatter(start[0], start[1], c="black", label="start")
            axes[0].scatter(goal[0], goal[1], c="green", marker="*", s=140, label="goal")
            axes[0].set_aspect("equal", adjustable="box")
            axes[0].set_xlabel("x [m]")
            axes[0].set_ylabel("y [m]")
            axes[0].set_title("Raw, visual spline, and MPC reference")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=7)

            axes[1].plot(time, true_clearances, label="raw true clearance")
            axes[1].plot(time, measured_clearances, label="raw measured clearance")
            axes[1].axhline(0.0, color="red", linestyle="--", label="collision boundary")
            axes[1].set_xlabel("time [s]")
            axes[1].set_ylabel("clearance [m]")
            axes[1].set_title("Safety metrics use raw trajectory only")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=8)
            fig.savefig(output_dir / "raw_smoothed_and_safety.png", dpi=160)
            plt.close(fig)

        if frames:
            gif_frames = [Image.fromarray(frame) for frame in frames]
            gif_frames[0].save(
                output_dir / "robot_motion.gif", save_all=True,
                append_images=gif_frames[1:], duration=80, loop=0, optimize=True,
            )

        final_goal_distance = float(goal_distances[-1])
        net_goal_progress = initial_goal_distance - final_goal_distance
        result = SmoothDynamicResult(
            outcome=classify_episode_outcome(
                reached_goal=reached_goal,
                collision=collision,
                truncated=truncated,
                initial_goal_distance=initial_goal_distance,
                final_goal_distance=final_goal_distance,
                solver_rejections=solver_rejections,
                steps=len(raw_positions),
                stall_progress_threshold=cfg.stall_progress_threshold,
            ),
            reached_goal=reached_goal,
            collision=collision,
            steps=len(raw_positions),
            sensor_updates=len(sensor_update_steps),
            minimum_true_clearance=float(min(true_clearances)),
            minimum_measured_clearance=float(min(measured_clearances)),
            initial_true_barrier=initial_true_barrier,
            minimum_true_barrier=float(min(true_barriers)),
            minimum_true_cbf_residual=(
                float(min(true_cbf_residuals)) if true_cbf_residuals else None
            ),
            true_cbf_violation_steps=sum(
                residual < -cfg.solver_max_constraint_violation
                for residual in true_cbf_residuals
            ),
            formal_filter_interventions=sum(
                bool(item["intervened"]) for item in formal_filter_audits
            ),
            formal_filter_uncertified_steps=sum(
                not bool(item["one_step_certified"])
                for item in formal_filter_audits
            ),
            formal_terminal_backup_uncertified_steps=sum(
                not bool(item["terminal_backup_certified"])
                for item in formal_filter_audits
            ),
            minimum_robust_filter_residual=(
                min(float(item["robust_cbf_residual"]) for item in formal_filter_audits)
                if formal_filter_audits
                else None
            ),
            minimum_backup_authority_margin=(
                min(float(item["backup_authority_margin"]) for item in formal_filter_audits)
                if formal_filter_audits
                else None
            ),
            final_goal_distance=final_goal_distance,
            initial_goal_distance=initial_goal_distance,
            net_goal_progress=net_goal_progress,
            mean_goal_progress_rate=(
                net_goal_progress / (len(raw_positions) * mpc_config.dt)
            ),
            final_speed_scale=tvp.speed_scale,
            final_clearance_margin=tvp.clearance_margin,
            safety_profile_transitions=safety_profile_transitions,
            mean_solve_time=float(np.mean(solve_times)),
            max_solve_time=float(np.max(solve_times)),
            p99_solve_time=float(np.quantile(solve_times, 0.99)),
            final_gamma=tvp.gamma,
            gamma_updates_applied=len(applied_gamma_updates),
            gamma_updates_rejected=(
                rejected_gamma_updates + feedback_updates_rejected_late
            ),
            prediction_mode_updates_applied=len(applied_prediction_mode_updates),
            final_prediction_mode=tvp.prediction_mode,
            reflex_interventions=len(reflex_audits),
            reflex_backups=sum(int(item["backup_used"]) for item in reflex_audits),
            reflex_side_switches=reflex.side_switches,
            reflex_policy_selections=sum(
                item["selected_policy"] != "nominal" for item in reflex_audits
            ),
            reflex_robust_recoveries=sum(
                bool(item["robust_recovery"]) for item in reflex_audits
            ),
            mean_optimal_decay=float(np.mean(optimal_decay_trace)),
            minimum_optimal_decay=float(np.min(optimal_decay_trace)),
            avoidance_onset_time=avoidance_onset_time,
            minimum_predicted_ttc=(
                float(min(predicted_ttc_values)) if predicted_ttc_values else None
            ),
            solver_failures=solver_failures,
            solver_rejections=solver_rejections,
            solver_max_cpu_time_exits=solver_termination_counts.get(
                "max_cpu_time", 0
            ),
            solver_infeasible_exits=solver_termination_counts.get(
                "infeasible", 0
            ),
            solver_unknown_exits=solver_termination_counts.get("unknown", 0),
            deadline_misses=sum(
                solve_time > cfg.control_deadline for solve_time in solve_times
            ),
            emergency_fallbacks=emergency_fallbacks,
            braking_fallbacks=braking_fallbacks,
            maximum_constraint_violation=(
                float(max(constraint_violations)) if constraint_violations else None
            ),
            mean_model_transition_error=float(np.mean(model_transition_errors)),
            max_model_transition_error=float(np.max(model_transition_errors)),
            mean_action_tracking_error=float(np.mean(action_tracking_errors)),
            max_action_tracking_error=float(np.max(action_tracking_errors)),
            feedback_trigger_time=ttc_feedback_trigger_time,
            feedback_available_time=ttc_feedback_available_time,
            feedback_causal_opportunity=ttc_feedback_causal_opportunity,
            feedback_updates_rejected_late=feedback_updates_rejected_late,
            most_infeasible_stage=(
                max(infeasible_stage_counts, key=infeasible_stage_counts.get)
                if infeasible_stage_counts
                else None
            ),
            infeasible_stage_events=sum(infeasible_stage_counts.values()),
            smoothness=smoothness,
            output_dir=str(output_dir),
        )
        payload = {
            "config": asdict(cfg),
            "result": {**asdict(result), "smoothness": smoothness.as_dict()},
            "safety_metric_source": "raw simulated end-effector positions only",
            "visual_spline_is_safety_evidence": False,
            "start": start.tolist(), "goal": goal.tolist(),
            "route_points": route_points.tolist(),
            "reference_path": reference_path.tolist(),
            "sensor_update_steps": sensor_update_steps,
            "estimated_obstacle_velocity": list(tvp.observer.velocity),
            "uncertainty_assumption": {
                "mode": cfg.prediction_mode,
                "bounded_measurement_error_m": tvp.tube.measurement_bound,
                "latency_inflation_m": tvp.tube.latency_bound,
                "control_zoh_intersample_inflation_m": (
                    tvp.tube.intersample_bound
                ),
                "sensor_zoh_period_s": tvp.tube.sensor_period,
                "control_zoh_period_s": tvp.tube.control_period,
                "identified_sensor_hold_growth_m": (
                    tvp.tube.sensor_hold_growth()
                ),
                "bootstrap_sensor_hold_growth_m": (
                    tvp.tube.sensor_hold_growth(
                        velocity_error_bound=(
                            tvp.tube.initial_velocity_error_bound
                        )
                    )
                ),
                "obstacle_acceleration_bound_mps2": (
                    tvp.tube.obstacle_acceleration_bound
                ),
                "initial_velocity_error_bound_mps": (
                    tvp.tube.initial_velocity_error_bound
                ),
                "identified_velocity_error_bound_mps": (
                    tvp.tube.velocity_error_bound
                ),
                "velocity_observer_filter": cfg.velocity_filter,
                "measurement_noise_is_deterministically_bounded": (
                    cfg.measurement_noise_mode == "bounded_ball"
                ),
                "measurement_noise_mode": cfg.measurement_noise_mode,
                "declared_measurement_error_bound_m": (
                    cfg.measurement_error_bound
                    if cfg.measurement_noise_mode == "bounded_ball"
                    else None
                ),
                "robot_transition_error_bound_m": (
                    cfg.formal_robot_transition_error_bound
                    if cfg.formal_safety_filter_enabled
                    else None
                ),
            },
            "applied_gamma_updates": applied_gamma_updates,
            "applied_prediction_mode_updates": applied_prediction_mode_updates,
            "gamma_trace": gamma_trace,
            "context_profile_trace": context_profile_trace,
            "runtime_trace": runtime_trace,
            "solver_audits": solver_audits,
            "positions": raw_positions.tolist(),
            "visual_spline": visual_spline.tolist(),
            "true_obstacles": true_array.tolist(),
            "measured_obstacles": measured_array.tolist(),
            "controls": controls_array.tolist(),
            "nominal_controls": np.asarray(nominal_controls).tolist(),
            "reflex_audits": reflex_audits,
            "formal_filter_audits": formal_filter_audits,
            "optimal_decay_trace": optimal_decay_trace,
            "true_clearances": true_clearances,
            "measured_clearances": measured_clearances,
            "true_barriers": true_barriers,
            "true_cbf_residuals": true_cbf_residuals,
            "goal_distances": goal_distances,
            "predicted_ttc_values": predicted_ttc_values,
            "solve_times": solve_times,
        }
        if cfg.save_metrics:
            (output_dir / "metrics.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        return result
    finally:
        wrapped_env.close()
