"""Sequential MPC-CBF pick–move–place demonstration for PandaBuildL-v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class BuildLDemoConfig:
    gamma: float = 0.15
    seed: int = 7
    max_move_steps: int = 100
    hover_height: float = 0.16
    grasp_offset: float = 0.03
    place_offset: float = 0.03
    place_tolerance: float = 0.04
    waypoint_tolerance: float = 0.012
    cube_radius: float = 0.018
    gripper_collision_radius: float = 0.012
    close_steps: int = 6
    open_steps: int = 8
    settle_steps: int = 10
    render_stride: int = 5
    output_dir: str = "artifacts/build_l_mpc_cbf"

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must satisfy 0 < gamma <= 1")
        if self.max_move_steps < 1 or self.render_stride < 1:
            raise ValueError("step counts must be positive")
        if min(
            self.hover_height,
            self.grasp_offset,
            self.place_offset,
            self.place_tolerance,
            self.waypoint_tolerance,
            self.cube_radius,
            self.gripper_collision_radius,
        ) <= 0.0:
            raise ValueError("geometric parameters must be positive")


@dataclass(frozen=True, slots=True)
class BuildLDemoResult:
    success: bool
    cubes_placed: int
    total_steps: int
    minimum_clearance: float
    maximum_place_error: float
    mean_solve_time: float
    max_solve_time: float
    output_dir: str


def run_build_l_mpc_cbf_demo(
    config: BuildLDemoConfig | None = None,
) -> BuildLDemoResult:
    """Move all four cubes to the L targets using MPC-CBF motion phases.

    The legacy environment has no stable physical grasp primitive.  A fixed
    PyBullet constraint is therefore created only after the gripper reaches and
    closes around a cube, then removed during release.  All free-space robot
    motion remains closed-loop MPC-CBF.
    """

    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import pybullet as pybullet

    import panda_gym  # noqa: F401 - registers PandaBuildL-v3

    from .controller import PaperMPCConfig, build_mpc_controller
    from .demo import make_static_cbf_builder, paper_control_to_safe_panda_action

    cfg = config or BuildLDemoConfig()
    np.random.seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapped_env = gym.make(
        "PandaBuildL-v3",
        render_mode="rgb_array",
        renderer="Tiny",
        render_width=720,
        render_height=480,
        render_distance=1.1,
        render_target_position=np.array([-0.15, 0.0, 0.12]),
    )
    env = wrapped_env.unwrapped

    frames: list[Any] = []
    end_effector_positions: list[Any] = []
    clearances: list[float] = []
    solve_times: list[float] = []
    stage_events: list[dict[str, Any]] = []
    total_steps = 0
    active_stage = "reset"

    def body_positions() -> list[Any]:
        return [
            np.asarray(env.sim.get_base_position(f"object{index}"), dtype=float)
            for index in range(1, 5)
        ]

    def record_frame() -> None:
        nonlocal total_steps
        if total_steps % cfg.render_stride == 0:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))

    def record_step(obstacles: Sequence[Any]) -> None:
        position = np.asarray(env.robot.get_ee_position(), dtype=float)
        end_effector_positions.append(position)
        if obstacles:
            radius = cfg.cube_radius + cfg.gripper_collision_radius
            clearances.append(
                min(float(np.linalg.norm(position - center) - radius) for center in obstacles)
            )
        record_frame()

    def apply_gripper(command: float, steps: int, label: str) -> None:
        nonlocal total_steps, active_stage
        active_stage = label
        started_at = total_steps
        for _ in range(steps):
            env.step(np.array([0.0, 0.0, 0.0, command], dtype=np.float32))
            total_steps += 1
            record_step(())
        stage_events.append(
            {"stage": label, "start_step": started_at, "end_step": total_steps, "reached": True}
        )

    def move_to(
        target: Any,
        obstacles: Sequence[Any],
        gripper_command: float,
        label: str,
        tolerance: float | None = None,
    ) -> bool:
        nonlocal total_steps, active_stage
        active_stage = label
        target = np.asarray(target, dtype=float)
        initial_position = np.asarray(env.robot.get_ee_position(), dtype=float)
        previous_control = np.zeros(4)
        initial_state = np.concatenate([initial_position, [0.0], previous_control])
        mpc_config = PaperMPCConfig(
            target=(*target, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        builders = ()
        if obstacles:
            builders = (
                make_static_cbf_builder(
                    obstacles,
                    obstacle_radius=cfg.cube_radius,
                    collision_radius=cfg.gripper_collision_radius,
                    gamma=cfg.gamma,
                    dt=mpc_config.dt,
                ),
            )
        _, mpc = build_mpc_controller(
            mpc_config, constraint_builders=builders
        )
        mpc.x0 = initial_state.reshape(-1, 1)
        mpc.set_initial_guess()
        started_at = total_steps
        reached = False
        active_tolerance = cfg.waypoint_tolerance if tolerance is None else tolerance
        for _ in range(cfg.max_move_steps):
            position = np.asarray(env.robot.get_ee_position(), dtype=float)
            if float(np.linalg.norm(position - target)) < active_tolerance:
                reached = True
                break
            measured_state = np.concatenate([position, [0.0], previous_control])
            started = perf_counter()
            candidate = np.asarray(
                mpc.make_step(measured_state.reshape(-1, 1)), dtype=float
            ).reshape(-1)
            solve_times.append(perf_counter() - started)
            if candidate.shape != (4,) or not np.all(np.isfinite(candidate)):
                raise RuntimeError(f"invalid MPC output during {label}")
            action = paper_control_to_safe_panda_action(
                candidate,
                4,
                linear_input_limit=mpc_config.linear_input_limit,
            )
            action[3] = gripper_command
            env.step(action)
            previous_control = candidate
            total_steps += 1
            record_step(obstacles)
        stage_events.append(
            {"stage": label, "start_step": started_at, "end_step": total_steps, "reached": reached}
        )
        return reached

    def attach_object(name: str) -> int:
        client = env.sim.physics_client
        panda_id = env.sim._bodies_idx["panda"]
        object_id = env.sim._bodies_idx[name]
        ee_position, ee_orientation = client.getLinkState(panda_id, env.robot.ee_link)[:2]
        object_position, object_orientation = client.getBasePositionAndOrientation(object_id)
        inverse_position, inverse_orientation = client.invertTransform(
            ee_position, ee_orientation
        )
        relative_position, relative_orientation = client.multiplyTransforms(
            inverse_position,
            inverse_orientation,
            object_position,
            object_orientation,
        )
        return int(
            client.createConstraint(
                panda_id,
                env.robot.ee_link,
                object_id,
                -1,
                pybullet.JOINT_FIXED,
                [0.0, 0.0, 0.0],
                relative_position,
                [0.0, 0.0, 0.0],
                relative_orientation,
                [0.0, 0.0, 0.0, 1.0],
            )
        )

    try:
        observation, _ = env.reset(seed=cfg.seed)
        targets = np.asarray(observation["desired_goal"], dtype=float).reshape(4, 3)
        initial_objects = body_positions()
        record_frame()
        all_stages_reached = True

        for cube_index in range(4):
            object_name = f"object{cube_index + 1}"
            objects = body_positions()
            object_position = objects[cube_index]
            other_objects = [
                position for index, position in enumerate(objects) if index != cube_index
            ]
            target_position = targets[cube_index]
            above_object = object_position + np.array([0.0, 0.0, cfg.hover_height])
            grasp_position = object_position + np.array([0.0, 0.0, cfg.grasp_offset])
            above_target = target_position + np.array([0.0, 0.0, cfg.hover_height])
            place_position = target_position + np.array([0.0, 0.0, cfg.place_offset])

            apply_gripper(1.0, cfg.open_steps, f"cube_{cube_index + 1}_open")
            all_stages_reached &= move_to(
                above_object, other_objects, 1.0, f"cube_{cube_index + 1}_approach"
            )
            all_stages_reached &= move_to(
                grasp_position, other_objects, 1.0, f"cube_{cube_index + 1}_descend"
            )
            apply_gripper(-1.0, cfg.close_steps, f"cube_{cube_index + 1}_close")
            constraint_id = attach_object(object_name)
            all_stages_reached &= move_to(
                above_object, other_objects, -1.0, f"cube_{cube_index + 1}_lift"
            )
            all_stages_reached &= move_to(
                above_target, other_objects, -1.0, f"cube_{cube_index + 1}_transport"
            )
            end_effector_place_reached = move_to(
                place_position,
                other_objects,
                -1.0,
                f"cube_{cube_index + 1}_place",
                tolerance=cfg.place_tolerance,
            )
            env.sim.physics_client.removeConstraint(constraint_id)
            apply_gripper(1.0, cfg.open_steps, f"cube_{cube_index + 1}_release")
            for _ in range(cfg.settle_steps):
                env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
                total_steps += 1
                record_step(())
            placement_error = float(
                np.linalg.norm(
                    np.asarray(env.sim.get_base_position(object_name), dtype=float)
                    - target_position
                )
            )
            placement_reached = placement_error < 0.04
            for event in reversed(stage_events):
                if event["stage"] == f"cube_{cube_index + 1}_place":
                    event["end_effector_reached"] = end_effector_place_reached
                    event["placement_error"] = placement_error
                    event["reached"] = placement_reached
                    break
            all_stages_reached &= placement_reached
            all_stages_reached &= move_to(
                above_target, body_positions(), 1.0, f"cube_{cube_index + 1}_retreat"
            )

        final_objects = np.asarray(body_positions())
        place_errors = np.linalg.norm(final_objects - targets, axis=1)
        cubes_placed = int(np.sum(place_errors < 0.04))
        success = bool(all_stages_reached and cubes_placed == 4)
        minimum_clearance = float(min(clearances)) if clearances else float("nan")
        result = BuildLDemoResult(
            success=success,
            cubes_placed=cubes_placed,
            total_steps=total_steps,
            minimum_clearance=minimum_clearance,
            maximum_place_error=float(np.max(place_errors)),
            mean_solve_time=float(np.mean(solve_times)),
            max_solve_time=float(np.max(solve_times)),
            output_dir=str(output_dir),
        )

        trajectory = np.asarray(end_effector_positions)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].plot(trajectory[:, 0], trajectory[:, 1], color="blue", linewidth=1.2)
        axes[0].scatter(initial_objects[0][0], initial_objects[0][1], alpha=0.0)
        axes[0].scatter(
            np.asarray(initial_objects)[:, 0],
            np.asarray(initial_objects)[:, 1],
            c=["blue", "green", "green", "red"],
            marker="s",
            label="initial cubes",
        )
        axes[0].scatter(targets[:, 0], targets[:, 1], c="orange", marker="x", label="L targets")
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_title("End-effector MPC-CBF trajectory")
        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[1].bar(range(1, 5), place_errors)
        axes[1].axhline(0.04, color="red", linestyle="--", label="placement tolerance")
        axes[1].set_xticks(range(1, 5))
        axes[1].set_xlabel("cube")
        axes[1].set_ylabel("final position error [m]")
        axes[1].set_title("Build-L placement error")
        axes[1].grid(True, axis="y", alpha=0.3)
        axes[1].legend()
        fig.savefig(output_dir / "trajectory_and_placement.png", dpi=160)
        plt.close(fig)

        if frames:
            selected = [0, len(frames) // 2, len(frames) - 1]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            for axis, frame_index, title in zip(axes, selected, ("Start", "Middle", "Final")):
                axis.imshow(frames[frame_index])
                axis.set_title(title)
                axis.axis("off")
            fig.savefig(output_dir / "build_l_motion_montage.png", dpi=160)
            plt.close(fig)
            gif_frames = [Image.fromarray(frame) for frame in frames]
            gif_frames[0].save(
                output_dir / "build_l_motion.gif",
                save_all=True,
                append_images=gif_frames[1:],
                duration=80,
                loop=0,
                optimize=True,
            )

        metrics = {
            "config": asdict(cfg),
            "result": asdict(result),
            "initial_objects": [position.tolist() for position in initial_objects],
            "targets": targets.tolist(),
            "final_objects": final_objects.tolist(),
            "place_errors": place_errors.tolist(),
            "stage_events": stage_events,
            "end_effector_positions": trajectory.tolist(),
            "clearances": clearances,
            "solve_times": solve_times,
            "grasp_model": "fixed PyBullet constraint after gripper closure",
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        return result
    finally:
        wrapped_env.close()
