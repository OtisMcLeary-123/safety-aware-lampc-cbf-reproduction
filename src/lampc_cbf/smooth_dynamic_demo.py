"""Continuous-reference MPC-CBF experiment and smoothness ablation primitive."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from .smoothness import SmoothnessMetrics


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
    reference_speed: float = 0.12
    goal_offset: tuple[float, float, float] = (0.0, 0.30, 0.0)
    obstacle_start_offset: tuple[float, float, float] = (-0.12, 0.15, 0.0)
    obstacle_velocity: tuple[float, float, float] = (0.06, 0.0, 0.0)
    obstacle_radius: float = 0.10
    collision_radius: float = 0.035
    prediction_mode: str = "velocity_tube"
    confidence_multiplier: float = 3.0
    velocity_error_bound: float = 0.03
    model_error_growth: float = 0.005
    max_relative_speed: float = 0.4
    total_latency: float = 0.04
    velocity_filter: float = 0.5
    safety_reflex_enabled: bool = True
    reflex_lookahead_steps: int = 8
    reflex_alpha: float = 4.0
    avoidance_onset_threshold: float = 0.005
    route_margin: float = 0.08
    reference_mode: str = "behind_spline"
    safety_mode: str = "cbf"
    # ``command_velocity`` matches the closed-loop simulator interface: the
    # command selected at this control cycle determines the position used by
    # the one-step CBF. ``paper_state`` retains the published double-integrator
    # transition, where p[k+1] depends on the stored velocity x[4:7].
    cbf_transition_mode: str = "command_velocity"
    gamma_schedule: tuple[tuple[float, float], ...] = ()
    gamma_update_ttl: float = 1.0
    context_safety_enabled: bool = False
    requested_safety_level: int = 3
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

    def __post_init__(self) -> None:
        if self.delta_u_weight < 0.0 or self.jerk_weight < 0.0 or self.optimal_decay_weight < 0.0:
            raise ValueError("smoothness weights must be non-negative")
        if not 0.0 < self.optimal_decay_lower <= 1.0:
            raise ValueError("optimal_decay_lower must be in (0, 1]")
        if not 0.0 < self.gamma <= 0.15:
            raise ValueError("paper experiment requires 0 < gamma <= 0.15")
        if self.max_steps < 1 or self.render_stride < 1:
            raise ValueError("step counts must be positive")
        if self.sensor_period <= 0.0 or self.reference_speed <= 0.0:
            raise ValueError("period and reference speed must be positive")
        if self.measurement_noise_sigma < 0.0:
            raise ValueError("measurement noise must be non-negative")
        if self.gamma_update_ttl <= 0.0:
            raise ValueError("gamma_update_ttl must be positive")
        if self.reference_mode not in {"behind_spline", "straight", "direct_target"}:
            raise ValueError(
                "reference_mode must be behind_spline, straight, or direct_target"
            )
        if self.prediction_mode not in {"static", "velocity", "velocity_tube"}:
            raise ValueError("prediction_mode must be static, velocity, or velocity_tube")
        if not 0.0 <= self.velocity_filter <= 1.0:
            raise ValueError("velocity_filter must be in [0, 1]")
        if self.reflex_lookahead_steps < 1 or self.reflex_alpha <= 0.0:
            raise ValueError("reflex lookahead and alpha must be positive")
        if self.avoidance_onset_threshold <= 0.0:
            raise ValueError("avoidance_onset_threshold must be positive")
        if self.safety_mode not in {"cbf", "distance", "none"}:
            raise ValueError("safety_mode must be cbf, distance, or none")
        if self.cbf_transition_mode not in {"paper_state", "command_velocity"}:
            raise ValueError(
                "cbf_transition_mode must be paper_state or command_velocity"
            )
        if not 1 <= self.requested_safety_level <= 5:
            raise ValueError("requested_safety_level must be in [1, 5]")
        if self.solver_max_constraint_violation < 0.0:
            raise ValueError("solver_max_constraint_violation must be non-negative")
        if self.solver_max_cpu_time <= 0.0:
            raise ValueError("solver_max_cpu_time must be positive")
        if self.control_deadline <= 0.0:
            raise ValueError("control_deadline must be positive")
        if any(time < 0.0 for time in self.provisional_feedback_times):
            raise ValueError("provisional feedback times must be non-negative")
        if tuple(sorted(self.provisional_feedback_times)) != self.provisional_feedback_times:
            raise ValueError("provisional feedback times must be sorted")
        if self.feedback_ttc_threshold is not None and self.feedback_ttc_threshold <= 0.0:
            raise ValueError("feedback_ttc_threshold must be positive when provided")
        if self.feedback_response_latency < 0.0:
            raise ValueError("feedback_response_latency must be non-negative")
        if self.feedback_reaction_margin < 0.0:
            raise ValueError("feedback_reaction_margin must be non-negative")
        if self.feedback_gamma is not None and not 0.0 < self.feedback_gamma <= 0.15:
            raise ValueError("feedback_gamma must be in (0, 0.15] when provided")
        if (self.feedback_ttc_threshold is None) != (self.feedback_gamma is None):
            raise ValueError("TTC feedback requires both threshold and feedback_gamma")
        previous_time = -1.0
        for update_time, gamma in self.gamma_schedule:
            if update_time < 0.0 or update_time < previous_time:
                raise ValueError("gamma_schedule times must be non-negative and sorted")
            if not 0.0 < gamma <= 0.15:
                raise ValueError("scheduled gamma must be in (0, 0.15]")
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
    final_goal_distance: float
    mean_solve_time: float
    max_solve_time: float
    p99_solve_time: float
    final_gamma: float
    gamma_updates_applied: int
    gamma_updates_rejected: int
    reflex_interventions: int
    reflex_backups: int
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


@dataclass(frozen=True, slots=True)
class ReferenceObstacleStage:
    """Auditable reference and obstacle TVPs for one horizon stage."""

    reference_state: tuple[float, ...]
    obstacle_position: tuple[float, float, float]
    obstacle_next_position: tuple[float, float, float]
    robust_radius: float
    robust_radius_next: float


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
        velocity_filter: float = 0.5,
        direct_target: bool = False,
        cbf_transition_mode: str = "paper_state",
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
        if cbf_transition_mode not in {"paper_state", "command_velocity"}:
            raise ValueError(
                "cbf_transition_mode must be paper_state or command_velocity"
            )
        self.cbf_transition_mode = cbf_transition_mode
        segment_lengths = np.linalg.norm(
            np.diff(self.reference_path, axis=0), axis=1
        )
        self.arc_length = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        self.progress_index = 0
        self._obstacle_tvp = None
        self._obstacle_next_tvp = None
        self._robust_radius_tvp = None
        self._robust_radius_next_tvp = None
        self._reference_tvp = None
        self._gamma_tvp = None

    def update_gamma(self, gamma: float) -> None:
        """Hot-swap an LLM-selected safety parameter without rebuilding MPC."""

        if not 0.0 < gamma <= 0.15:
            raise ValueError("gamma must be in the experimental interval (0, 0.15]")
        self.gamma = float(gamma)

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
                self.tube.inflation(stage_age)
                if self.prediction_mode == "velocity_tube"
                else 0.0
            )
            tube_next = (
                self.tube.inflation(next_age)
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
        return ReferenceObstacleStage(
            reference_state=tuple(float(value) for value in reference_state),
            obstacle_position=tuple(float(value) for value in obstacle_position),
            obstacle_next_position=tuple(
                float(value) for value in obstacle_next_position
            ),
            robust_radius=float(robust_radius),
            robust_radius_next=float(robust_radius_next),
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
        transition_velocity = (
            u[:3]
            if self.cbf_transition_mode == "command_velocity"
            else x[4:7]
        )
        next_position = position + self.dt * transition_velocity
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
            mpc.set_nl_cons(
                "dynamic_obstacle_cbf",
                decay * (1.0 - self._gamma_tvp) * h_current - h_next,
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

    from .controller import PaperMPCConfig, build_mpc_controller
    from .async_gamma import AtomicGammaStore, GammaUpdate, GammaUpdateQueue
    from .demo import paper_control_to_safe_panda_action
    from .obstacle_prediction import UncertaintyTubeConfig
    from .safety_reflex import (
        OperationalSpaceSafetyReflex,
        ReflexObstacle,
        SafetyReflexConfig,
    )
    from .safety_scheduler import (
        ContextAwareSafetyScheduler,
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
    from .safe_panda import simulator_calibration_sample
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
    goal_distances: list[float] = []
    solve_times: list[float] = []
    predicted_ttc_values: list[float] = []
    context_profile_trace: list[dict[str, Any]] = []
    model_transition_errors: list[float] = []
    action_tracking_errors: list[float] = []
    solver_audits: list[dict[str, Any]] = []
    solver_failures = 0
    solver_rejections = 0
    solver_termination_counts: dict[str, int] = {}
    emergency_fallbacks = 0
    constraint_violations: list[float] = []
    sensor_update_steps: list[int] = [0]

    try:
        observation, _ = env.reset(seed=cfg.seed)
        start = np.asarray(observation["achieved_goal"], dtype=float)
        goal = start + np.asarray(cfg.goal_offset, dtype=float)
        true_obstacle = start + np.asarray(cfg.obstacle_start_offset, dtype=float)
        obstacle_velocity = np.asarray(cfg.obstacle_velocity, dtype=float)
        hidden_obstacle = np.array([2.0, 2.0, -1.0])
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        task = env.task
        task.goal = goal.copy()
        task.unsafe_region_radius = cfg.obstacle_radius
        task.unsafe_state_1_pos = true_obstacle.copy()
        task.unsafe_state_2_pos = hidden_obstacle.copy()
        env.sim.set_base_pose("target", goal, quaternion)
        env.sim.set_base_pose("unsafe_region_1", true_obstacle, quaternion)
        env.sim.set_base_pose("unsafe_region_2", hidden_obstacle, quaternion)
        observation = env._get_obs()

        combined_radius = cfg.obstacle_radius + cfg.collision_radius
        if cfg.reference_mode in {"straight", "direct_target"}:
            route_points = np.asarray([start, goal])
            reference_path = np.linspace(start, goal, 600)
        else:
            route_offset = combined_radius + cfg.route_margin
            route_direction = -1.0 if obstacle_velocity[0] >= 0.0 else 1.0
            route_points = np.asarray(
                [
                    start,
                    start + np.array([route_direction * route_offset, 0.075, 0.0]),
                    start + np.array([route_direction * route_offset, 0.225, 0.0]),
                    goal,
                ]
            )
            reference_path = make_reference_bspline(route_points)
        measurement = true_obstacle + rng.normal(
            0.0, cfg.measurement_noise_sigma, size=3
        )
        mpc_config = PaperMPCConfig(
            linear_delta_u_weight=cfg.delta_u_weight,
            linear_jerk_weight=cfg.jerk_weight,
            optimal_decay_weight=cfg.optimal_decay_weight,
            optimal_decay_lower=cfg.optimal_decay_lower,
            target_tvp_name="reference_state",
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
                velocity_error_bound=cfg.velocity_error_bound,
                model_error_growth=cfg.model_error_growth,
                max_relative_speed=cfg.max_relative_speed,
                total_latency=cfg.total_latency,
            ),
            velocity_filter=cfg.velocity_filter,
            direct_target=cfg.reference_mode == "direct_target",
            cbf_transition_mode=cfg.cbf_transition_mode,
        )
        _, mpc = build_mpc_controller(
            mpc_config,
            model_builders=(tvp.declare,),
            constraint_builders=(tvp.configure,),
            nlpsol_options=IpoptConfig(
                constraint_violation_tolerance=(
                    cfg.solver_max_constraint_violation
                    if cfg.solver_max_constraint_violation > 0.0
                    else 1e-12
                ),
                max_cpu_time=cfg.solver_max_cpu_time,
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
            )
        )
        previous_control = np.zeros(4)
        previous_increment = np.zeros(4)
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
        gamma_queue = GammaUpdateQueue()
        gamma_store = AtomicGammaStore(cfg.gamma, clock_skew_tolerance=0.0)
        applied_gamma_updates: list[dict[str, float]] = []
        rejected_gamma_updates = 0
        gamma_trace: list[float] = []
        reflex_audits: list[dict[str, Any]] = []
        optimal_decay_trace: list[float] = []
        avoidance_onset_time: float | None = None
        safety_scheduler = ContextAwareSafetyScheduler()
        feasibility_policy = FeasibilityPolicy(
            max_constraint_violation=cfg.solver_max_constraint_violation
        )
        provisional_index = 0
        provisional_safety_active = False
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
                provisional_safety_active = True
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
                schedule_index < len(cfg.gamma_schedule)
                and elapsed + 1e-12 >= cfg.gamma_schedule[schedule_index][0]
            ):
                scheduled_time, scheduled_gamma = cfg.gamma_schedule[schedule_index]
                gamma_queue.publish(
                    GammaUpdate(
                        gamma=scheduled_gamma,
                        version=schedule_index + 1,
                        created_at=elapsed,
                        valid_until=elapsed + cfg.gamma_update_ttl,
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
                # The deterministic provisional profile protects the latency
                # window only. Once a validated update arrives, hand control
                # back to the updated MPC parameter.
                provisional_safety_active = False
                applied_gamma_updates.append(
                    {
                        "scheduled_time": gamma_audit.applied.created_at,
                        "applied_time": elapsed,
                        "gamma": gamma_audit.applied.gamma,
                        "version": gamma_audit.applied.version,
                    }
                )
            if elapsed + 1e-12 >= next_sensor_time:
                measurement = true_obstacle + rng.normal(
                    0.0, cfg.measurement_noise_sigma, size=3
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
            estimated_position = np.asarray(tvp.observer.predict(elapsed), dtype=float)
            estimated_velocity = np.asarray(
                tvp.observer.velocity
                if cfg.prediction_mode in {"velocity", "velocity_tube"}
                else (0.0, 0.0, 0.0),
                dtype=float,
            )
            predicted_ttc = constant_velocity_ttc(
                position - estimated_position,
                previous_control[:3] - estimated_velocity,
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
                provisional_safety_active = True
            if cfg.context_safety_enabled or provisional_safety_active:
                profile = safety_scheduler.select(
                    predicted_ttc=predicted_ttc,
                    requested_safety_level=(
                        1 if provisional_safety_active else cfg.requested_safety_level
                    ),
                )
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
                        **asdict(profile),
                    }
                )
            measured_state = np.concatenate([position, [0.0], previous_control])
            if mpc_config.uses_jerk_state:
                measured_state = np.concatenate([measured_state, previous_increment])
            started = perf_counter()
            raw_candidate = np.asarray(
                mpc.make_step(measured_state.reshape(-1, 1)), dtype=float
            ).reshape(-1)
            measured_solve_time = perf_counter() - started
            solve_times.append(measured_solve_time)
            expected_dimension = 5 if mpc_config.uses_optimal_decay else 4
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
            if rejected_candidate:
                solver_rejections += 1
                emergency_fallbacks += 1
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
            if (
                (
                    cfg.safety_reflex_enabled
                    or provisional_safety_active
                    or rejected_candidate
                )
                and cfg.safety_mode != "none"
            ):
                prediction_age = max(0.0, elapsed - tvp.observer.timestamp)
                uncertainty = (
                    tvp.tube.inflation(prediction_age)
                    if cfg.prediction_mode == "velocity_tube"
                    else 0.0
                )
                reflex_result = reflex.gate(
                    position,
                    candidate[:3],
                    (
                        ReflexObstacle(
                            tuple(float(value) for value in estimated_position),
                            tuple(float(value) for value in estimated_velocity),
                            combined_radius,
                            uncertainty,
                            "moving_obstacle",
                        ),
                    ),
                )
                candidate[:3] = reflex_result.velocity
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
                        }
                    )
            pre_step_position = position.copy()
            model_velocity = previous_control[:3].copy()
            action = paper_control_to_safe_panda_action(
                candidate, env.action_space.shape[0],
                linear_input_limit=mpc_config.linear_input_limit,
                dt=mpc_config.dt,
            )
            observation, _, terminated, truncated, _ = env.step(action)
            previous_increment = candidate - previous_control
            previous_control = candidate

            true_obstacle = true_obstacle + obstacle_velocity * mpc_config.dt
            env.sim.set_base_pose("unsafe_region_1", true_obstacle, quaternion)
            task.unsafe_state_1_pos = true_obstacle.copy()
            position = np.asarray(observation["achieved_goal"], dtype=float)
            calibration = simulator_calibration_sample(
                pre_step_position,
                position,
                model_velocity,
                candidate[:3],
                mpc_config.dt,
            )
            model_transition_errors.append(calibration.model_transition_error)
            action_tracking_errors.append(calibration.action_tracking_error)
            true_clearance = float(
                np.linalg.norm(position - true_obstacle) - combined_radius
            )
            measured_clearance = float(
                np.linalg.norm(position - measurement) - combined_radius
            )
            goal_distance = float(np.linalg.norm(position - goal))
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
            goal_distances.append(goal_distance)
            gamma_trace.append(tvp.gamma)
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

        result = SmoothDynamicResult(
            outcome=(
                "collision"
                if collision
                else "goal"
                if reached_goal
                else "environment_truncated"
                if truncated
                else "emergency_fallback"
                if emergency_fallbacks
                else "timeout"
            ),
            reached_goal=reached_goal,
            collision=collision,
            steps=len(raw_positions),
            sensor_updates=len(sensor_update_steps),
            minimum_true_clearance=float(min(true_clearances)),
            minimum_measured_clearance=float(min(measured_clearances)),
            final_goal_distance=float(goal_distances[-1]),
            mean_solve_time=float(np.mean(solve_times)),
            max_solve_time=float(np.max(solve_times)),
            p99_solve_time=float(np.quantile(solve_times, 0.99)),
            final_gamma=tvp.gamma,
            gamma_updates_applied=len(applied_gamma_updates),
            gamma_updates_rejected=(
                rejected_gamma_updates + feedback_updates_rejected_late
            ),
            reflex_interventions=len(reflex_audits),
            reflex_backups=sum(int(item["backup_used"]) for item in reflex_audits),
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
                "gaussian_noise_is_deterministically_bounded": False,
            },
            "applied_gamma_updates": applied_gamma_updates,
            "gamma_trace": gamma_trace,
            "context_profile_trace": context_profile_trace,
            "solver_audits": solver_audits,
            "positions": raw_positions.tolist(),
            "visual_spline": visual_spline.tolist(),
            "true_obstacles": true_array.tolist(),
            "measured_obstacles": measured_array.tolist(),
            "controls": controls_array.tolist(),
            "nominal_controls": np.asarray(nominal_controls).tolist(),
            "reflex_audits": reflex_audits,
            "optimal_decay_trace": optimal_decay_trace,
            "true_clearances": true_clearances,
            "measured_clearances": measured_clearances,
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
