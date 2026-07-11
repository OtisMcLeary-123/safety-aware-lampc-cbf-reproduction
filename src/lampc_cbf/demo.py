"""Closed-loop Safe Panda MPC-CBF demonstration and visual artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class DemoConfig:
    gamma: float = 0.10
    seed: int = 7
    max_steps: int = 400
    render_stride: int = 2
    goal_offset: tuple[float, float, float] = (0.0, 0.30, 0.0)
    obstacle_radius: float = 0.10
    collision_radius: float = 0.035
    waypoint_margin: float = 0.08
    waypoint_tolerance: float = 0.05
    output_dir: str = "artifacts/mpc_cbf_demo/gamma_0.10"

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must satisfy 0 < gamma <= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.render_stride < 1:
            raise ValueError("render_stride must be positive")
        if self.obstacle_radius <= 0.0 or self.collision_radius < 0.0:
            raise ValueError("radii must be valid")
        if self.waypoint_margin <= 0.0 or self.waypoint_tolerance <= 0.0:
            raise ValueError("waypoint parameters must be positive")


@dataclass(frozen=True, slots=True)
class DemoResult:
    reached_goal: bool
    collision: bool
    steps: int
    minimum_clearance: float
    final_goal_distance: float
    mean_solve_time: float
    max_solve_time: float
    output_dir: str


def make_static_cbf_builder(
    obstacle_positions: Sequence[Sequence[float]],
    *,
    obstacle_radius: float,
    collision_radius: float,
    gamma: float,
    dt: float = 0.04,
) -> Any:
    """Create do-mpc stage constraints for equation (14)."""

    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must satisfy 0 < gamma <= 1")
    combined_radius = obstacle_radius + collision_radius

    def builder(model: Any, mpc: Any, x: Any, u: Any, ca: Any) -> None:
        del model, u
        position = x[:3]
        next_position = position + dt * x[4:7]
        for index, obstacle_position in enumerate(obstacle_positions):
            center = ca.DM(tuple(float(value) for value in obstacle_position))
            h_current = ca.sumsqr(position - center) - combined_radius**2
            h_next = ca.sumsqr(next_position - center) - combined_radius**2
            # do-mpc expects expression <= ub. Equation (14) requires
            # h_next - (1-gamma) h_current >= 0.
            mpc.set_nl_cons(
                f"cbf_obstacle_{index}",
                (1.0 - gamma) * h_current - h_next,
                ub=0.0,
            )

    return builder


def paper_control_to_safe_panda_action(
    control: Sequence[float],
    action_dimension: int,
    *,
    linear_input_limit: float = 0.2,
) -> Any:
    """Map paper-bounded inputs to Safe Panda's normalized action interval."""

    import numpy as np

    if len(control) != 4:
        raise ValueError("paper control must have four values")
    if action_dimension not in (3, 4):
        raise ValueError("Safe Panda action dimension must be 3 or 4")
    if linear_input_limit <= 0.0:
        raise ValueError("linear_input_limit must be positive")
    xyz = np.clip(
        np.asarray(control[:3], dtype=float) / linear_input_limit, -1.0, 1.0
    )
    if action_dimension == 3:
        return xyz.astype(np.float32)
    return np.concatenate([xyz, np.zeros(1)]).astype(np.float32)


def run_safe_panda_mpc_cbf_demo(config: DemoConfig | None = None) -> DemoResult:
    """Run one deterministic closed-loop episode and write figures/GIF/metrics."""

    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    import panda_gym  # noqa: F401 - registers the environments

    from .controller import PaperMPCConfig, build_mpc_controller

    cfg = config or DemoConfig()
    np.random.seed(cfg.seed)
    wrapped_env = gym.make(
        "PandaReachSafe-v3", render_mode="rgb_array", renderer="Tiny"
    )
    env = wrapped_env.unwrapped
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Any] = []
    positions: list[Any] = []
    clearances: list[float] = []
    goal_distances: list[float] = []
    solve_times: list[float] = []
    controls: list[Any] = []
    active_subtasks: list[int] = []

    try:
        observation, _ = env.reset(seed=cfg.seed)
        start = np.asarray(observation["achieved_goal"], dtype=float)
        goal = start + np.asarray(cfg.goal_offset, dtype=float)
        obstacle_1 = start + 0.50 * np.asarray(cfg.goal_offset, dtype=float)
        obstacle_2 = start + np.array([0.20, -0.20, 0.10])

        task = env.task
        task.goal = goal.copy()
        task.unsafe_region_radius = cfg.obstacle_radius
        task.unsafe_state_1_pos = obstacle_1.copy()
        task.unsafe_state_2_pos = obstacle_2.copy()
        env.sim.set_base_pose("target", goal, np.array([0.0, 0.0, 0.0, 1.0]))
        env.sim.set_base_pose(
            "unsafe_region_1", obstacle_1, np.array([0.0, 0.0, 0.0, 1.0])
        )
        env.sim.set_base_pose(
            "unsafe_region_2", obstacle_2, np.array([0.0, 0.0, 0.0, 1.0])
        )
        observation = env._get_obs()

        combined_radius = cfg.obstacle_radius + cfg.collision_radius
        route_height = combined_radius + cfg.waypoint_margin
        planar_offset = np.asarray(cfg.goal_offset[:2], dtype=float)
        planar_norm = float(np.linalg.norm(planar_offset))
        if planar_norm <= 0.0:
            raise ValueError("goal_offset must include planar motion")
        perpendicular = np.array(
            [-planar_offset[1], planar_offset[0], 0.0]
        ) / planar_norm
        route_targets = (
            start + 0.25 * np.asarray(cfg.goal_offset) + route_height * perpendicular,
            start + 0.75 * np.asarray(cfg.goal_offset) + route_height * perpendicular,
            goal,
        )
        base_mpc_config = PaperMPCConfig()
        cbf_builder = make_static_cbf_builder(
            (obstacle_1, obstacle_2),
            obstacle_radius=cfg.obstacle_radius,
            collision_radius=cfg.collision_radius,
            gamma=cfg.gamma,
            dt=base_mpc_config.dt,
        )

        def build_for_target(target: Any, state: Any) -> tuple[Any, PaperMPCConfig]:
            target_state = (*target, 0.0, 0.0, 0.0, 0.0, 0.0)
            active_config = PaperMPCConfig(target=target_state)
            _, active_mpc = build_mpc_controller(
                active_config, constraint_builders=(cbf_builder,)
            )
            active_mpc.x0 = state.reshape(-1, 1)
            active_mpc.set_initial_guess()
            return active_mpc, active_config

        previous_control = np.zeros(4)
        initial_state = np.concatenate([start, [0.0], previous_control])
        active_subtask = 0
        mpc, mpc_config = build_for_target(
            route_targets[active_subtask], initial_state
        )

        reached_goal = False
        collision = False

        for step_index in range(cfg.max_steps):
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
            )
            observation, _, terminated, truncated, _ = env.step(action)
            previous_control = candidate

            position = np.asarray(observation["achieved_goal"], dtype=float)
            distances = [
                float(np.linalg.norm(position - center) - combined_radius)
                for center in (obstacle_1, obstacle_2)
            ]
            clearance = min(distances)
            goal_distance = float(np.linalg.norm(position - goal))
            active_distance = float(
                np.linalg.norm(position - route_targets[active_subtask])
            )
            positions.append(position.copy())
            clearances.append(clearance)
            goal_distances.append(goal_distance)
            controls.append(candidate.copy())
            active_subtasks.append(active_subtask)
            if step_index % cfg.render_stride == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))

            collision = collision or clearance < 0.0
            reached_goal = bool(terminated) or goal_distance < 0.05
            if (
                not reached_goal
                and active_distance < cfg.waypoint_tolerance
                and active_subtask < len(route_targets) - 1
            ):
                active_subtask += 1
                switch_state = np.concatenate(
                    [position, [0.0], previous_control]
                )
                mpc, mpc_config = build_for_target(
                    route_targets[active_subtask], switch_state
                )
            if reached_goal or collision or truncated:
                break

        if not positions:
            raise RuntimeError("demo produced no simulation steps")

        position_array = np.asarray(positions)
        control_array = np.asarray(controls)
        time = np.arange(1, len(positions) + 1) * mpc_config.dt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].plot(position_array[:, 0], position_array[:, 1], "b-", label="MPC-CBF path")
        axes[0].scatter(start[0], start[1], c="black", marker="o", label="start")
        axes[0].scatter(goal[0], goal[1], c="green", marker="*", s=160, label="goal")
        axes[0].scatter(
            [target[0] for target in route_targets[:-1]],
            [target[1] for target in route_targets[:-1]],
            c="orange",
            marker="D",
            label="TP-style waypoint",
        )
        for index, center in enumerate((obstacle_1, obstacle_2)):
            circle = plt.Circle(
                center[:2], combined_radius, color="red", alpha=0.25,
                label="CBF exclusion radius" if index == 0 else None,
            )
            axes[0].add_patch(circle)
            axes[0].scatter(center[0], center[1], c="red", marker="x")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        axes[0].set_title(f"Top-view trajectory, gamma={cfg.gamma:.2f}")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)

        axes[1].plot(time, clearances, label="minimum clearance")
        axes[1].axhline(0.0, color="red", linestyle="--", label="collision boundary")
        axes[1].plot(time, goal_distances, label="goal distance")
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("distance [m]")
        axes[1].set_title("Safety and convergence")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.savefig(output_dir / "trajectory_and_clearance.png", dpi=160)
        plt.close(fig)

        if frames:
            selected = [0, len(frames) // 2, len(frames) - 1]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            for axis, frame_index, title in zip(
                axes, selected, ("Start", "Middle", "Final")
            ):
                axis.imshow(frames[frame_index])
                axis.set_title(
                    f"{title}: step {frame_index * cfg.render_stride + 1}"
                )
                axis.axis("off")
            fig.savefig(output_dir / "robot_motion_montage.png", dpi=160)
            plt.close(fig)

            gif_frames = [Image.fromarray(frame) for frame in frames]
            gif_frames[0].save(
                output_dir / "robot_motion.gif",
                save_all=True,
                append_images=gif_frames[1:],
                duration=80,
                loop=0,
                optimize=True,
            )

        result = DemoResult(
            reached_goal=reached_goal,
            collision=collision,
            steps=len(positions),
            minimum_clearance=float(min(clearances)),
            final_goal_distance=float(goal_distances[-1]),
            mean_solve_time=float(np.mean(solve_times)),
            max_solve_time=float(np.max(solve_times)),
            output_dir=str(output_dir),
        )
        metrics = {
            "config": asdict(cfg),
            "result": asdict(result),
            "start": start.tolist(),
            "goal": goal.tolist(),
            "obstacles": [obstacle_1.tolist(), obstacle_2.tolist()],
            "route_targets": [target.tolist() for target in route_targets],
            "active_subtasks": active_subtasks,
            "positions": position_array.tolist(),
            "controls": control_array.tolist(),
            "clearances": clearances,
            "goal_distances": goal_distances,
            "solve_times": solve_times,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        return result
    finally:
        wrapped_env.close()
