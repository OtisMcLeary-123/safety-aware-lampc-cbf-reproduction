"""Paper-aligned moving-obstacle MPC-CBF experiment in Safe Panda Gym."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class DynamicObstacleConfig:
    gamma: float = 0.10
    seed: int = 11
    max_steps: int = 220
    render_stride: int = 2
    sensor_period: float = 0.67
    measurement_noise_sigma: float = 0.005
    goal_offset: tuple[float, float, float] = (0.0, 0.30, 0.0)
    obstacle_start_offset: tuple[float, float, float] = (-0.12, 0.15, 0.0)
    obstacle_velocity: tuple[float, float, float] = (0.06, 0.0, 0.0)
    obstacle_radius: float = 0.10
    collision_radius: float = 0.035
    waypoint_margin: float = 0.08
    waypoint_tolerance: float = 0.05
    output_dir: str = "artifacts/dynamic_obstacle_mpc_cbf/gamma_0.10"

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 0.15:
            raise ValueError("paper experiment requires 0 < gamma <= 0.15")
        if self.max_steps < 1 or self.render_stride < 1:
            raise ValueError("step counts must be positive")
        if self.sensor_period <= 0.0 or self.measurement_noise_sigma < 0.0:
            raise ValueError("sensor settings must be valid")
        if self.obstacle_radius <= 0.0 or self.collision_radius < 0.0:
            raise ValueError("radii must be valid")
        if self.waypoint_margin <= 0.0 or self.waypoint_tolerance <= 0.0:
            raise ValueError("waypoint settings must be positive")


@dataclass(frozen=True, slots=True)
class DynamicObstacleResult:
    reached_goal: bool
    collision: bool
    steps: int
    sensor_updates: int
    minimum_true_clearance: float
    minimum_measured_clearance: float
    final_goal_distance: float
    mean_solve_time: float
    max_solve_time: float
    output_dir: str


class OnlineObstacleCBF:
    """A zero-order-held obstacle measurement exposed to do-mpc as a TVP."""

    def __init__(
        self,
        initial_measurement: Sequence[float],
        *,
        obstacle_radius: float,
        collision_radius: float,
        gamma: float,
        dt: float,
        horizon: int,
    ) -> None:
        import numpy as np

        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must satisfy 0 < gamma <= 1")
        self.measurement = np.asarray(initial_measurement, dtype=float).reshape(3)
        self.combined_radius = obstacle_radius + collision_radius
        self.gamma = gamma
        self.dt = dt
        self.horizon = horizon
        self._tvp = None

    def update_measurement(self, measurement: Sequence[float]) -> None:
        import numpy as np

        value = np.asarray(measurement, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("obstacle measurement must be a finite 3-vector")
        self.measurement = value.copy()

    def declare_tvp(self, model: Any, x: Any, u: Any, ca: Any) -> None:
        del x, u, ca
        self._tvp = model.set_variable(
            var_type="_tvp", var_name="obstacle_position", shape=(3, 1)
        )

    def add_constraint(
        self, model: Any, mpc: Any, x: Any, u: Any, ca: Any
    ) -> None:
        del model, u
        if self._tvp is None:
            raise RuntimeError("declare_tvp must run before add_constraint")
        position = x[:3]
        next_position = position + self.dt * x[4:7]
        h_current = ca.sumsqr(position - self._tvp) - self.combined_radius**2
        h_next = ca.sumsqr(next_position - self._tvp) - self.combined_radius**2
        mpc.set_nl_cons(
            "dynamic_obstacle_cbf",
            (1.0 - self.gamma) * h_current - h_next,
            ub=0.0,
        )

        template = mpc.get_tvp_template()

        def tvp_fun(t_now: float) -> Any:
            del t_now
            for stage in range(self.horizon + 1):
                template["_tvp", stage, "obstacle_position"] = (
                    self.measurement.reshape(3, 1)
                )
            return template

        mpc.set_tvp_fun(tvp_fun)


def run_dynamic_obstacle_demo(
    config: DynamicObstacleConfig | None = None,
) -> DynamicObstacleResult:
    """Run the moving-sphere experiment and save reproducibility artifacts."""

    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    import panda_gym  # noqa: F401

    from .controller import PaperMPCConfig, build_mpc_controller
    from .demo import paper_control_to_safe_panda_action

    cfg = config or DynamicObstacleConfig()
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
    true_clearances: list[float] = []
    measured_clearances: list[float] = []
    goal_distances: list[float] = []
    controls: list[Any] = []
    solve_times: list[float] = []
    sensor_update_steps: list[int] = []
    active_subtasks: list[int] = []

    try:
        observation, _ = env.reset(seed=cfg.seed)
        start = np.asarray(observation["achieved_goal"], dtype=float)
        goal = start + np.asarray(cfg.goal_offset, dtype=float)
        true_obstacle = start + np.asarray(cfg.obstacle_start_offset, dtype=float)
        velocity = np.asarray(cfg.obstacle_velocity, dtype=float)
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

        measurement = true_obstacle + rng.normal(
            0.0, cfg.measurement_noise_sigma, size=3
        )
        sensor_update_steps.append(0)
        base_config = PaperMPCConfig()
        online_cbf = OnlineObstacleCBF(
            measurement,
            obstacle_radius=cfg.obstacle_radius,
            collision_radius=cfg.collision_radius,
            gamma=cfg.gamma,
            dt=base_config.dt,
            horizon=base_config.horizon,
        )

        combined_radius = cfg.obstacle_radius + cfg.collision_radius
        route_offset = combined_radius + cfg.waypoint_margin
        # Pass behind the crossing obstacle (opposite its x velocity). This is
        # the deterministic TP-style subtask choice; the CBF remains the
        # online safety filter for the noisy, held obstacle measurement.
        route_direction = -1.0 if velocity[0] >= 0.0 else 1.0
        route_targets = (
            start + np.array([route_direction * route_offset, 0.075, 0.0]),
            start + np.array([route_direction * route_offset, 0.225, 0.0]),
            goal,
        )

        def build_for_target(target: Any, state: Any) -> tuple[Any, PaperMPCConfig]:
            target_state = (*target, 0.0, 0.0, 0.0, 0.0, 0.0)
            active_config = PaperMPCConfig(target=target_state)
            _, active_mpc = build_mpc_controller(
                active_config,
                model_builders=(online_cbf.declare_tvp,),
                constraint_builders=(online_cbf.add_constraint,),
            )
            active_mpc.x0 = state.reshape(-1, 1)
            active_mpc.set_initial_guess()
            return active_mpc, active_config

        previous_control = np.zeros(4)
        state = np.concatenate([start, [0.0], previous_control])
        active_subtask = 0
        mpc, mpc_config = build_for_target(route_targets[0], state)
        reached_goal = False
        collision = False
        next_sensor_time = cfg.sensor_period

        for step_index in range(cfg.max_steps):
            elapsed = step_index * mpc_config.dt
            if elapsed + 1e-12 >= next_sensor_time:
                measurement = true_obstacle + rng.normal(
                    0.0, cfg.measurement_noise_sigma, size=3
                )
                online_cbf.update_measurement(measurement)
                sensor_update_steps.append(step_index)
                next_sensor_time += cfg.sensor_period

            position = np.asarray(observation["achieved_goal"], dtype=float)
            measured_state = np.concatenate([position, [0.0], previous_control])
            started = perf_counter()
            candidate = np.asarray(
                mpc.make_step(measured_state.reshape(-1, 1)), dtype=float
            ).reshape(-1)
            solve_times.append(perf_counter() - started)
            if candidate.shape != (4,) or not np.all(np.isfinite(candidate)):
                raise RuntimeError("MPC returned an invalid control vector")

            action = paper_control_to_safe_panda_action(
                candidate,
                env.action_space.shape[0],
                linear_input_limit=mpc_config.linear_input_limit,
                dt=mpc_config.dt,
            )
            observation, _, terminated, truncated, _ = env.step(action)
            previous_control = candidate

            true_obstacle = true_obstacle + velocity * mpc_config.dt
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
            active_distance = float(
                np.linalg.norm(position - route_targets[active_subtask])
            )
            positions.append(position.copy())
            true_obstacles.append(true_obstacle.copy())
            measured_obstacles.append(measurement.copy())
            true_clearances.append(true_clearance)
            measured_clearances.append(measured_clearance)
            goal_distances.append(goal_distance)
            controls.append(candidate.copy())
            active_subtasks.append(active_subtask)
            if step_index % cfg.render_stride == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))

            collision = collision or true_clearance < 0.0
            reached_goal = bool(terminated) or goal_distance < 0.05
            if (
                not reached_goal
                and active_distance < cfg.waypoint_tolerance
                and active_subtask < len(route_targets) - 1
            ):
                active_subtask += 1
                switch_state = np.concatenate([position, [0.0], previous_control])
                mpc, mpc_config = build_for_target(
                    route_targets[active_subtask], switch_state
                )
            if reached_goal or collision or truncated:
                break

        if not positions:
            raise RuntimeError("experiment produced no simulation steps")

        position_array = np.asarray(positions)
        true_array = np.asarray(true_obstacles)
        measured_array = np.asarray(measured_obstacles)
        control_array = np.asarray(controls)
        time = np.arange(1, len(positions) + 1) * mpc_config.dt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].plot(position_array[:, 0], position_array[:, 1], "b-", label="robot")
        axes[0].plot(true_array[:, 0], true_array[:, 1], "r--", label="moving obstacle")
        sample_indices = [i for i in sensor_update_steps if i < len(measured_array)]
        axes[0].scatter(
            measured_array[sample_indices, 0], measured_array[sample_indices, 1],
            c="magenta", marker="x", label="noisy sensor update",
        )
        axes[0].scatter(start[0], start[1], c="black", label="start")
        axes[0].scatter(goal[0], goal[1], c="green", marker="*", s=150, label="goal")
        axes[0].scatter(
            [target[0] for target in route_targets[:-1]],
            [target[1] for target in route_targets[:-1]],
            c="orange", marker="D", label="TP-style waypoint",
        )
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        axes[0].set_title("Robot and dynamic obstacle trajectories")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)

        axes[1].plot(time, true_clearances, label="true clearance")
        axes[1].plot(time, measured_clearances, alpha=0.75, label="CBF measured clearance")
        axes[1].axhline(0.0, color="red", linestyle="--", label="collision boundary")
        for index, update_step in enumerate(sample_indices):
            axes[1].axvline(
                (update_step + 1) * mpc_config.dt,
                color="magenta", alpha=0.15,
                label="sensor update" if index == 0 else None,
            )
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("clearance [m]")
        axes[1].set_title("CBF safety with 0.67 s zero-order hold")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=8)
        fig.savefig(output_dir / "trajectory_and_clearance.png", dpi=160)
        plt.close(fig)

        if frames:
            selected = [0, len(frames) // 2, len(frames) - 1]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            for axis, frame_index, title in zip(
                axes, selected, ("Start", "Middle", "Final")
            ):
                axis.imshow(frames[frame_index])
                axis.set_title(f"{title}: step {frame_index * cfg.render_stride + 1}")
                axis.axis("off")
            fig.savefig(output_dir / "robot_motion_montage.png", dpi=160)
            plt.close(fig)
            gif_frames = [Image.fromarray(frame) for frame in frames]
            gif_frames[0].save(
                output_dir / "robot_motion.gif", save_all=True,
                append_images=gif_frames[1:], duration=80, loop=0, optimize=True,
            )

        result = DynamicObstacleResult(
            reached_goal=reached_goal,
            collision=collision,
            steps=len(positions),
            sensor_updates=len(sensor_update_steps),
            minimum_true_clearance=float(min(true_clearances)),
            minimum_measured_clearance=float(min(measured_clearances)),
            final_goal_distance=float(goal_distances[-1]),
            mean_solve_time=float(np.mean(solve_times)),
            max_solve_time=float(np.max(solve_times)),
            output_dir=str(output_dir),
        )
        metrics = {
            "config": asdict(cfg), "result": asdict(result),
            "paper_alignment": {
                "control_period_seconds": mpc_config.dt,
                "sensor_period_seconds": cfg.sensor_period,
                "measurement_noise_sigma_meters": cfg.measurement_noise_sigma,
                "prediction_assumption": "measured obstacle position is static over each horizon",
                "obstacle_motion": "constant Cartesian velocity",
            },
            "start": start.tolist(), "goal": goal.tolist(),
            "route_targets": [target.tolist() for target in route_targets],
            "sensor_update_steps": sensor_update_steps,
            "active_subtasks": active_subtasks,
            "positions": position_array.tolist(),
            "true_obstacles": true_array.tolist(),
            "measured_obstacles": measured_array.tolist(),
            "controls": control_array.tolist(),
            "true_clearances": true_clearances,
            "measured_clearances": measured_clearances,
            "goal_distances": goal_distances,
            "solve_times": solve_times,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        return result
    finally:
        wrapped_env.close()
