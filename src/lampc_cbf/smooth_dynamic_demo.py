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
    route_margin: float = 0.08
    reference_mode: str = "behind_spline"
    safety_mode: str = "cbf"
    gamma_schedule: tuple[tuple[float, float], ...] = ()
    gamma_update_ttl: float = 1.0
    save_plots: bool = True
    save_metrics: bool = True
    output_dir: str = "artifacts/smoothness_ablation/spline_du_0.5"

    def __post_init__(self) -> None:
        if self.delta_u_weight < 0.0 or self.jerk_weight < 0.0:
            raise ValueError("smoothness weights must be non-negative")
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
        if self.reference_mode not in {"behind_spline", "straight"}:
            raise ValueError("reference_mode must be behind_spline or straight")
        if self.prediction_mode not in {"static", "velocity_tube"}:
            raise ValueError("prediction_mode must be static or velocity_tube")
        if not 0.0 <= self.velocity_filter <= 1.0:
            raise ValueError("velocity_filter must be in [0, 1]")
        if self.safety_mode not in {"cbf", "distance", "none"}:
            raise ValueError("safety_mode must be cbf, distance, or none")
        previous_time = -1.0
        for update_time, gamma in self.gamma_schedule:
            if update_time < 0.0 or update_time < previous_time:
                raise ValueError("gamma_schedule times must be non-negative and sorted")
            if not 0.0 < gamma <= 0.15:
                raise ValueError("scheduled gamma must be in (0, 0.15]")
            previous_time = update_time


@dataclass(frozen=True, slots=True)
class SmoothDynamicResult:
    reached_goal: bool
    collision: bool
    steps: int
    sensor_updates: int
    minimum_true_clearance: float
    minimum_measured_clearance: float
    final_goal_distance: float
    mean_solve_time: float
    max_solve_time: float
    final_gamma: float
    gamma_updates_applied: int
    gamma_updates_rejected: int
    smoothness: SmoothnessMetrics
    output_dir: str


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
        self.combined_radius = obstacle_radius + collision_radius
        self.gamma = gamma
        self.dt = dt
        self.horizon = horizon
        if safety_mode not in {"cbf", "distance", "none"}:
            raise ValueError("invalid safety_mode")
        self.safety_mode = safety_mode
        if prediction_mode not in {"static", "velocity_tube"}:
            raise ValueError("invalid prediction_mode")
        self.prediction_mode = prediction_mode
        self.tube = tube_config or UncertaintyTubeConfig()
        self.control_time = 0.0
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

    def configure(self, model: Any, mpc: Any, x: Any, u: Any, ca: Any) -> None:
        import numpy as np

        del model, u
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
        next_position = position + self.dt * x[4:7]
        h_current = (
            ca.sumsqr(position - self._obstacle_tvp) - self._robust_radius_tvp**2
        )
        h_next = (
            ca.sumsqr(next_position - self._obstacle_next_tvp)
            - self._robust_radius_next_tvp**2
        )
        if self.safety_mode == "cbf":
            mpc.set_nl_cons(
                "dynamic_obstacle_cbf",
                (1.0 - self._gamma_tvp) * h_current - h_next,
                ub=0.0,
            )
        elif self.safety_mode == "distance":
            mpc.set_nl_cons("dynamic_obstacle_distance", -h_next, ub=0.0)
        template = mpc.get_tvp_template()

        def tvp_fun(t_now: float) -> Any:
            del t_now
            base_distance = self.arc_length[self.progress_index]
            for stage in range(self.horizon + 1):
                target_distance = min(
                    self.arc_length[-1],
                    base_distance + stage * self.dt * self.reference_speed,
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
                    tangent / tangent_norm * self.reference_speed
                    if tangent_norm > 1e-9 and index < len(self.reference_path) - 1
                    else np.zeros(3)
                )
                reference_state = np.concatenate(
                    [point, [0.0], desired_velocity, [0.0]]
                )
                stage_age = max(0.0, self.control_time - self.observer.timestamp) + stage * self.dt
                next_age = stage_age + self.dt
                if self.prediction_mode == "velocity_tube":
                    obstacle_position = np.asarray(
                        self.observer.predict(self.observer.timestamp + stage_age), dtype=float
                    )
                    obstacle_next_position = np.asarray(
                        self.observer.predict(self.observer.timestamp + next_age), dtype=float
                    )
                    robust_radius = self.combined_radius + self.tube.inflation(stage_age)
                    robust_radius_next = self.combined_radius + self.tube.inflation(next_age)
                else:
                    obstacle_position = self.measurement
                    obstacle_next_position = self.measurement
                    robust_radius = self.combined_radius
                    robust_radius_next = self.combined_radius
                template["_tvp", stage, "obstacle_position"] = (
                    obstacle_position.reshape(3, 1)
                )
                template["_tvp", stage, "obstacle_next_position"] = (
                    obstacle_next_position.reshape(3, 1)
                )
                template["_tvp", stage, "obstacle_robust_radius"] = robust_radius
                template["_tvp", stage, "obstacle_next_robust_radius"] = robust_radius_next
                template["_tvp", stage, "reference_state"] = (
                    reference_state.reshape(8, 1)
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
    true_clearances: list[float] = []
    measured_clearances: list[float] = []
    goal_distances: list[float] = []
    solve_times: list[float] = []
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
        if cfg.reference_mode == "straight":
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
        )
        _, mpc = build_mpc_controller(
            mpc_config,
            model_builders=(tvp.declare,),
            constraint_builders=(tvp.configure,),
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
        next_sensor_time = cfg.sensor_period
        last_measurement_time = 0.0
        schedule_index = 0
        gamma_queue = GammaUpdateQueue()
        gamma_store = AtomicGammaStore(cfg.gamma, clock_skew_tolerance=0.0)
        applied_gamma_updates: list[dict[str, float]] = []
        rejected_gamma_updates = 0
        gamma_trace: list[float] = []

        for step_index in range(cfg.max_steps):
            elapsed = step_index * mpc_config.dt
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
            if gamma_audit.applied is not None:
                tvp.update_gamma(gamma_audit.applied.gamma)
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
            measured_state = np.concatenate([position, [0.0], previous_control])
            if mpc_config.uses_jerk_state:
                measured_state = np.concatenate([measured_state, previous_increment])
            started = perf_counter()
            candidate = np.asarray(
                mpc.make_step(measured_state.reshape(-1, 1)), dtype=float
            ).reshape(-1)
            solve_times.append(perf_counter() - started)
            if candidate.shape != (4,) or not np.all(np.isfinite(candidate)):
                raise RuntimeError("MPC returned an invalid control vector")
            action = paper_control_to_safe_panda_action(
                candidate, env.action_space.shape[0],
                linear_input_limit=mpc_config.linear_input_limit,
            )
            observation, _, terminated, truncated, _ = env.step(action)
            previous_increment = candidate - previous_control
            previous_control = candidate

            true_obstacle = true_obstacle + obstacle_velocity * mpc_config.dt
            env.sim.set_base_pose("unsafe_region_1", true_obstacle, quaternion)
            task.unsafe_state_1_pos = true_obstacle.copy()
            position = np.asarray(observation["achieved_goal"], dtype=float)
            true_clearance = float(
                np.linalg.norm(position - true_obstacle) - combined_radius
            )
            measured_clearance = float(
                np.linalg.norm(position - measurement) - combined_radius
            )
            goal_distance = float(np.linalg.norm(position - goal))
            positions.append(position.copy())
            true_obstacles.append(true_obstacle.copy())
            measured_obstacles.append(measurement.copy())
            controls.append(candidate.copy())
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
            reached_goal=reached_goal,
            collision=collision,
            steps=len(raw_positions),
            sensor_updates=len(sensor_update_steps),
            minimum_true_clearance=float(min(true_clearances)),
            minimum_measured_clearance=float(min(measured_clearances)),
            final_goal_distance=float(goal_distances[-1]),
            mean_solve_time=float(np.mean(solve_times)),
            max_solve_time=float(np.max(solve_times)),
            final_gamma=tvp.gamma,
            gamma_updates_applied=len(applied_gamma_updates),
            gamma_updates_rejected=rejected_gamma_updates,
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
            "positions": raw_positions.tolist(),
            "visual_spline": visual_spline.tolist(),
            "true_obstacles": true_array.tolist(),
            "measured_obstacles": measured_array.tolist(),
            "controls": controls_array.tolist(),
            "true_clearances": true_clearances,
            "measured_clearances": measured_clearances,
            "goal_distances": goal_distances,
            "solve_times": solve_times,
        }
        if cfg.save_metrics:
            (output_dir / "metrics.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        return result
    finally:
        wrapped_env.close()
