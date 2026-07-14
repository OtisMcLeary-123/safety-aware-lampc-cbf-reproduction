"""Sequential MPC-CBF pick–move–place demonstration for PandaBuildL-v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
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
    cube_indices: tuple[int, ...] = (0, 1, 2, 3)
    place_blue_on_red: bool = False
    dynamic_obstacle: bool = False
    obstacle_radius: float = 0.055
    obstacle_velocity: tuple[float, float, float] = (0.0, 0.025, 0.0)
    sensor_period: float = 0.67
    measurement_noise_sigma: float = 0.005
    user_instruction: str = "Build the four-cube L shape safely."
    llm_model: str = "not-used"
    llm_safety_level: int = 5
    llm_latency_seconds: float = 0.0
    llm_fallback_used: bool = False
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
            self.obstacle_radius,
            self.sensor_period,
        ) <= 0.0:
            raise ValueError("geometric parameters must be positive")
        if self.measurement_noise_sigma < 0.0:
            raise ValueError("measurement noise must be non-negative")
        if not self.cube_indices or any(index not in range(4) for index in self.cube_indices):
            raise ValueError("cube_indices must contain values in [0, 3]")
        if len(set(self.cube_indices)) != len(self.cube_indices):
            raise ValueError("cube_indices must not contain duplicates")
        if self.place_blue_on_red and self.cube_indices != (0,):
            raise ValueError("blue-on-red mode requires cube_indices=(0,)")
        if self.llm_safety_level not in range(1, 6):
            raise ValueError("llm_safety_level must be in [1, 5]")
        if self.llm_latency_seconds < 0.0 or not self.user_instruction.strip():
            raise ValueError("language metadata must be valid")


@dataclass(frozen=True, slots=True)
class BuildLDemoResult:
    success: bool
    collision_free: bool
    cubes_placed: int
    total_steps: int
    minimum_clearance: float
    minimum_dynamic_clearance: float
    minimum_dynamic_cbf_margin: float
    maximum_place_error: float
    mean_solve_time: float
    max_solve_time: float
    output_dir: str
    language_plan_used: bool = False
    language_model: str = "not-used"
    language_tp_latency_seconds: float = 0.0
    language_od_latency_seconds: float = 0.0
    language_od_fallbacks: int = 0
    language_gammas: tuple[float, ...] = ()
    language_execution_source: str = "not-used"
    language_source_metrics: str | None = None


def _project_world_points(
    points: Any,
    *,
    physics_client: Any,
    width: int,
    height: int,
    target_position: Any,
    distance: float,
    yaw: float,
    pitch: float,
    roll: float,
) -> Any:
    """Project PyBullet world coordinates into rendered-image pixels."""

    import numpy as np

    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    view = np.asarray(
        physics_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=np.asarray(target_position, dtype=float),
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=2,
        )
    ).reshape(4, 4, order="F")
    projection = np.asarray(
        physics_client.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=float(width) / height,
            nearVal=0.1,
            farVal=100.0,
        )
    ).reshape(4, 4, order="F")
    homogeneous = np.column_stack([xyz, np.ones(len(xyz))])
    clip = (projection @ view @ homogeneous.T).T
    normalized = clip[:, :3] / clip[:, 3, None]
    return np.column_stack(
        [
            (normalized[:, 0] + 1.0) * 0.5 * width,
            (1.0 - normalized[:, 1]) * 0.5 * height,
        ]
    )


def _evaluate_task_success(
    stage_events: Sequence[dict[str, Any]],
    *,
    cubes_placed: int,
    expected_cubes: int,
    clearances: Sequence[float],
    dynamic_clearances: Sequence[float],
) -> tuple[bool, bool]:
    """Return ``(success, collision_free)`` using paper-level task criteria."""

    required_stages_reached = all(
        bool(event["reached"])
        for event in stage_events
        if event.get("required_for_success", True)
    )
    collision_free = bool(
        (not clearances or min(clearances) >= 0.0)
        and (not dynamic_clearances or min(dynamic_clearances) >= 0.0)
    )
    success = bool(
        required_stages_reached
        and cubes_placed == expected_cubes
        and collision_free
    )
    return success, collision_free


def run_build_l_mpc_cbf_demo(
    config: BuildLDemoConfig | None = None,
    *,
    language_planner: Any | None = None,
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
    from .dynamic_obstacle_demo import OnlineObstacleCBF
    from .language_dsl import (
        LanguageDSLInferenceError,
        OptimizationSpec,
        SceneObject,
        controller_config_from_optimization,
    )
    from .trusted_executor import build_trusted_pick_place_macros

    cfg = config or BuildLDemoConfig()
    if language_planner is not None and (
        not cfg.place_blue_on_red or cfg.cube_indices != (0,) or not cfg.dynamic_obstacle
    ):
        raise ValueError(
            "trusted language execution requires blue-on-red mode with one dynamic obstacle"
        )
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
    frame_stages: list[str] = []
    frame_obstacles: list[Any | None] = []
    frame_cbf_radii: list[float | None] = []
    end_effector_positions: list[Any] = []
    trajectory_stages: list[str] = []
    clearances: list[float] = []
    dynamic_clearances: list[float] = []
    dynamic_cbf_margins: list[float] = []
    dynamic_cbf_margin_stages: list[str] = []
    dynamic_cbf_boundary_radii: list[float] = []
    obstacle_positions: list[Any] = []
    obstacle_measurements: list[Any] = []
    sensor_update_steps: list[int] = []
    solve_times: list[float] = []
    stage_events: list[dict[str, Any]] = []
    total_steps = 0
    active_stage = "reset"
    rng = np.random.default_rng(cfg.seed)
    true_obstacle: Any | None = None
    measured_obstacle: Any | None = None
    next_sensor_time = cfg.sensor_period
    obstacle_velocity = np.asarray(cfg.obstacle_velocity, dtype=float)
    simulation_dt = PaperMPCConfig().dt
    language_result: Any | None = None
    trusted_macros: tuple[Any, ...] = ()
    language_scene: tuple[Any, ...] = ()
    active_dynamic_cbf_radius: float | None = None

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
                frame_stages.append(active_stage)
                frame_obstacles.append(
                    None if true_obstacle is None else true_obstacle.copy()
                )
                frame_cbf_radii.append(active_dynamic_cbf_radius)

    def advance_dynamic_obstacle() -> None:
        nonlocal true_obstacle, measured_obstacle, next_sensor_time
        if true_obstacle is None:
            return
        true_obstacle = true_obstacle + obstacle_velocity * simulation_dt
        env.sim.set_base_pose(
            "language_obstacle",
            true_obstacle,
            np.array([0.0, 0.0, 0.0, 1.0]),
        )
        elapsed = total_steps * simulation_dt
        if elapsed + 1e-12 >= next_sensor_time:
            measured_obstacle = true_obstacle + rng.normal(
                0.0, cfg.measurement_noise_sigma, size=3
            )
            sensor_update_steps.append(total_steps)
            next_sensor_time += cfg.sensor_period

    def record_step(obstacles: Sequence[Any]) -> None:
        position = np.asarray(env.robot.get_ee_position(), dtype=float)
        end_effector_positions.append(position)
        trajectory_stages.append(active_stage)
        if obstacles:
            radius = cfg.cube_radius + cfg.gripper_collision_radius
            clearances.append(
                min(float(np.linalg.norm(position - center) - radius) for center in obstacles)
            )
        if true_obstacle is not None and measured_obstacle is not None:
            combined_radius = cfg.obstacle_radius + cfg.gripper_collision_radius
            center_distance = float(np.linalg.norm(position - true_obstacle))
            dynamic_clearances.append(
                center_distance - combined_radius
            )
            if active_dynamic_cbf_radius is not None:
                dynamic_cbf_margins.append(
                    center_distance - active_dynamic_cbf_radius
                )
                dynamic_cbf_margin_stages.append(active_stage)
                dynamic_cbf_boundary_radii.append(active_dynamic_cbf_radius)
            obstacle_positions.append(true_obstacle.copy())
            obstacle_measurements.append(measured_obstacle.copy())
        record_frame()

    def apply_gripper(command: float, steps: int, label: str) -> None:
        nonlocal total_steps, active_stage, active_dynamic_cbf_radius
        active_stage = label
        active_dynamic_cbf_radius = None
        started_at = total_steps
        for _ in range(steps):
            env.step(np.array([0.0, 0.0, 0.0, command], dtype=np.float32))
            total_steps += 1
            advance_dynamic_obstacle()
            record_step(())
        stage_events.append(
            {
                "stage": label,
                "start_step": started_at,
                "end_step": total_steps,
                "reached": True,
                "required_for_success": True,
            }
        )

    def move_to(
        target: Any,
        obstacles: Sequence[Any],
        gripper_command: float,
        label: str,
        tolerance: float | None = None,
        dynamic_cbf: bool = True,
        optimization_spec: OptimizationSpec | None = None,
        required_for_success: bool = True,
    ) -> bool:
        nonlocal total_steps, active_stage, active_dynamic_cbf_radius
        active_stage = label
        target = np.asarray(target, dtype=float)
        initial_position = np.asarray(env.robot.get_ee_position(), dtype=float)
        previous_control = np.zeros(4)
        initial_state = np.concatenate([initial_position, [0.0], previous_control])
        if optimization_spec is None:
            mpc_config = PaperMPCConfig(
                target=(*target, 0.0, 0.0, 0.0, 0.0, 0.0)
            )
            active_gamma = cfg.gamma
            dynamic_clearance = cfg.gripper_collision_radius
        else:
            mpc_config = replace(
                controller_config_from_optimization(
                    optimization_spec,
                    language_scene,
                    current_position=initial_position,
                ),
                target=(*target, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
            active_gamma = optimization_spec.safety.gamma
            dynamic_constraints = [
                constraint
                for constraint in optimization_spec.constraints
                if constraint.kind == "collision_clearance"
                and constraint.object_name == "moving_obstacle"
            ]
            if len(dynamic_constraints) != 1:
                raise RuntimeError(
                    f"validated OD spec for {label} must contain one dynamic obstacle constraint"
                )
            dynamic_clearance = dynamic_constraints[0].clearance_m
        constraint_builders: tuple[Any, ...] = ()
        if obstacles:
            constraint_builders = (
                make_static_cbf_builder(
                    obstacles,
                    obstacle_radius=cfg.cube_radius,
                    collision_radius=cfg.gripper_collision_radius,
                    gamma=active_gamma,
                    dt=mpc_config.dt,
                ),
            )
        if optimization_spec is not None:
            def add_language_height_constraints(model, mpc, x, u, ca):
                del model, u, ca
                for index, constraint in enumerate(optimization_spec.constraints):
                    if constraint.kind == "minimum_height":
                        mpc.set_nl_cons(
                            f"dsl_minimum_height_{index}",
                            constraint.value_m - x[2],
                            ub=0.0,
                        )
                    elif constraint.kind == "maximum_height":
                        mpc.set_nl_cons(
                            f"dsl_maximum_height_{index}",
                            x[2] - constraint.value_m,
                            ub=0.0,
                        )

            constraint_builders += (add_language_height_constraints,)
        model_builders: tuple[Any, ...] = ()
        online_cbf = None
        if (
            dynamic_cbf
            and true_obstacle is not None
            and measured_obstacle is not None
        ):
            online_cbf = OnlineObstacleCBF(
                measured_obstacle,
                obstacle_radius=cfg.obstacle_radius,
                collision_radius=dynamic_clearance,
                gamma=active_gamma,
                dt=mpc_config.dt,
                horizon=mpc_config.horizon,
            )
            model_builders = (online_cbf.declare_tvp,)
            constraint_builders += (online_cbf.add_constraint,)
            active_dynamic_cbf_radius = cfg.obstacle_radius + dynamic_clearance
        else:
            active_dynamic_cbf_radius = None
        _, mpc = build_mpc_controller(
            mpc_config,
            model_builders=model_builders,
            constraint_builders=constraint_builders,
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
            if online_cbf is not None and measured_obstacle is not None:
                online_cbf.update_measurement(measured_obstacle)
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
                dt=mpc_config.dt,
            )
            action[3] = gripper_command
            env.step(action)
            previous_control = candidate
            total_steps += 1
            advance_dynamic_obstacle()
            record_step(obstacles)
        stage_events.append(
            {
                "stage": label,
                "start_step": started_at,
                "end_step": total_steps,
                "reached": reached,
                "required_for_success": required_for_success,
            }
        )
        active_dynamic_cbf_radius = None
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
        if cfg.place_blue_on_red:
            scene_objects = np.asarray(
                [
                    [-0.14, -0.12, 0.02],  # blue, manipulated
                    [0.08, -0.08, 0.02],   # green distractor
                    [0.08, 0.12, 0.02],    # orange distractor
                    [-0.08, 0.14, 0.02],   # red placement base
                ]
            )
            for index, position in enumerate(scene_objects, start=1):
                env.sim.set_base_pose(
                    f"object{index}",
                    position,
                    np.array([0.0, 0.0, 0.0, 1.0]),
                )
            env.sim.physics_client.changeVisualShape(
                env.sim._bodies_idx["object3"],
                -1,
                rgbaColor=[0.95, 0.55, 0.05, 1.0],
            )
            hidden_target = np.array([2.0, 2.0, -1.0])
            for index in range(2, 5):
                env.sim.set_base_pose(
                    f"target{index}",
                    hidden_target,
                    np.array([0.0, 0.0, 0.0, 1.0]),
                )
        initial_objects = body_positions()
        if cfg.place_blue_on_red:
            targets = targets.copy()
            targets[0] = initial_objects[3] + np.array([0.0, 0.0, 0.04])
            env.sim.set_base_pose(
                "target1",
                targets[0],
                np.array([0.0, 0.0, 0.0, 1.0]),
            )
        if cfg.dynamic_obstacle:
            transport_start = initial_objects[0] + np.array(
                [0.0, 0.0, cfg.hover_height]
            )
            transport_goal = targets[0] + np.array(
                [0.0, 0.0, cfg.hover_height]
            )
            true_obstacle = 0.5 * (transport_start + transport_goal)
            route_xy = transport_goal[:2] - transport_start[:2]
            route_xy /= np.linalg.norm(route_xy)
            true_obstacle[:2] += 0.025 * np.array([-route_xy[1], route_xy[0]])
            measured_obstacle = true_obstacle + rng.normal(
                0.0, cfg.measurement_noise_sigma, size=3
            )
            sensor_update_steps.append(0)
            env.sim.create_sphere(
                body_name="language_obstacle",
                radius=cfg.obstacle_radius,
                mass=0.0,
                ghost=True,
                position=true_obstacle,
                rgba_color=np.array([0.95, 0.95, 0.95, 1.0]),
                specular_color=np.array([0.5, 0.5, 0.5]),
            )
        if language_planner is not None:
            aliases = ("blue_cube", "green_cube", "orange_cube", "red_cube")
            current_objects = body_positions()
            language_scene = tuple(
                SceneObject(alias, tuple(float(value) for value in position), cfg.cube_radius)
                for alias, position in zip(aliases, current_objects)
            ) + (
                SceneObject(
                    "moving_obstacle",
                    tuple(float(value) for value in true_obstacle),
                    cfg.obstacle_radius,
                ),
            )
            current_ee = tuple(float(value) for value in env.robot.get_ee_position())
            try:
                language_result = language_planner.formulate(
                    cfg.user_instruction,
                    language_scene,
                    current_position=current_ee,
                    required_hazards=("moving_obstacle",),
                )
            except LanguageDSLInferenceError as error:
                audit = {
                    "status": "rejected_fail_closed",
                    "stage": error.stage,
                    "cause_type": error.cause_type,
                    "message": str(error),
                    "model": getattr(
                        getattr(language_planner, "config", None),
                        "model",
                        "unknown",
                    ),
                    "provider": getattr(
                        getattr(language_planner, "config", None),
                        "provider",
                        "unknown",
                    ),
                    "instruction_hash": sha256(
                        cfg.user_instruction.strip().encode("utf-8")
                    ).hexdigest(),
                    "raw_response": error.raw_response,
                    "robot_motion_started": False,
                }
                (output_dir / "language_failure_audit.json").write_text(
                    json.dumps(audit, indent=2), encoding="utf-8"
                )
                raise
            trusted_macros = build_trusted_pick_place_macros(
                language_result,
                required_hazards=("moving_obstacle",),
            )
            if len(trusted_macros) != 1:
                raise RuntimeError(
                    "blue-on-red trusted runtime currently requires exactly one pick/place macro"
                )
            macro = trusted_macros[0]
            if (
                macro.source.object_name != "blue_cube"
                or macro.destination.object_name != "red_cube"
            ):
                raise RuntimeError(
                    "blue-on-red trusted runtime rejected a mismatched source or destination"
                )
        record_frame()
        all_stages_reached = True

        for cube_index in cfg.cube_indices:
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
            pick_spec = (
                trusted_macros[0].pick_optimization if trusted_macros else None
            )
            place_spec = (
                trusted_macros[0].place_optimization if trusted_macros else None
            )

            apply_gripper(1.0, cfg.open_steps, f"cube_{cube_index + 1}_open")
            all_stages_reached &= move_to(
                above_object,
                other_objects,
                1.0,
                f"cube_{cube_index + 1}_approach",
                optimization_spec=pick_spec,
            )
            all_stages_reached &= move_to(
                grasp_position,
                other_objects,
                1.0,
                f"cube_{cube_index + 1}_descend",
                optimization_spec=pick_spec,
            )
            apply_gripper(-1.0, cfg.close_steps, f"cube_{cube_index + 1}_close")
            constraint_id = attach_object(object_name)
            all_stages_reached &= move_to(
                above_object,
                other_objects,
                -1.0,
                f"cube_{cube_index + 1}_lift",
                optimization_spec=place_spec,
            )
            if true_obstacle is not None:
                current_position = np.asarray(
                    env.robot.get_ee_position(), dtype=float
                )
                route_xy = above_target[:2] - current_position[:2]
                route_xy /= np.linalg.norm(route_xy)
                left_normal = np.array([-route_xy[1], route_xy[0]])
                waypoint = true_obstacle.copy()
                waypoint[:2] += left_normal * (
                    cfg.obstacle_radius
                    + cfg.gripper_collision_radius
                    + 0.07
                )
                waypoint[2] = max(
                    current_position[2], above_target[2], true_obstacle[2]
                )
                waypoint_required = place_spec is None
                waypoint_reached = move_to(
                    waypoint,
                    other_objects,
                    -1.0,
                    f"cube_{cube_index + 1}_transport_avoid",
                    tolerance=0.025,
                    optimization_spec=place_spec,
                    required_for_success=waypoint_required,
                )
                if waypoint_required:
                    all_stages_reached &= waypoint_reached
            all_stages_reached &= move_to(
                above_target,
                other_objects,
                -1.0,
                f"cube_{cube_index + 1}_transport",
                optimization_spec=place_spec,
            )
            end_effector_place_reached = move_to(
                place_position,
                other_objects,
                -1.0,
                f"cube_{cube_index + 1}_place",
                tolerance=cfg.place_tolerance,
                dynamic_cbf=place_spec is not None,
                optimization_spec=place_spec,
            )
            env.sim.physics_client.removeConstraint(constraint_id)
            apply_gripper(1.0, cfg.open_steps, f"cube_{cube_index + 1}_release")
            for _ in range(cfg.settle_steps):
                env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
                total_steps += 1
                advance_dynamic_obstacle()
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
                above_target,
                body_positions(),
                1.0,
                f"cube_{cube_index + 1}_retreat",
                dynamic_cbf=place_spec is not None,
                optimization_spec=place_spec,
            )

        final_objects = np.asarray(body_positions())
        place_errors = np.linalg.norm(final_objects - targets, axis=1)
        selected_errors = place_errors[list(cfg.cube_indices)]
        cubes_placed = int(np.sum(selected_errors < cfg.place_tolerance))
        minimum_clearance = float(min(clearances)) if clearances else float("nan")
        minimum_dynamic_clearance = (
            float(min(dynamic_clearances)) if dynamic_clearances else float("nan")
        )
        minimum_dynamic_cbf_margin = (
            float(min(dynamic_cbf_margins))
            if dynamic_cbf_margins
            else float("nan")
        )
        success, collision_free = _evaluate_task_success(
            stage_events,
            cubes_placed=cubes_placed,
            expected_cubes=len(cfg.cube_indices),
            clearances=clearances,
            dynamic_clearances=dynamic_clearances,
        )
        result = BuildLDemoResult(
            success=success,
            collision_free=collision_free,
            cubes_placed=cubes_placed,
            total_steps=total_steps,
            minimum_clearance=minimum_clearance,
            minimum_dynamic_clearance=minimum_dynamic_clearance,
            minimum_dynamic_cbf_margin=minimum_dynamic_cbf_margin,
            maximum_place_error=float(np.max(selected_errors)),
            mean_solve_time=float(np.mean(solve_times)),
            max_solve_time=float(np.max(solve_times)),
            output_dir=str(output_dir),
            language_plan_used=language_result is not None,
            language_model=(
                language_result.model if language_result is not None else "not-used"
            ),
            language_tp_latency_seconds=(
                language_result.tp_latency_seconds
                if language_result is not None
                else 0.0
            ),
            language_od_latency_seconds=(
                language_result.od_latency_seconds
                if language_result is not None
                else 0.0
            ),
            language_od_fallbacks=(
                language_result.od_fallbacks if language_result is not None else 0
            ),
            language_gammas=tuple(
                spec.safety.gamma
                for spec in (
                    language_result.optimization_specs
                    if language_result is not None
                    else ()
                )
                if spec is not None
            ),
            language_execution_source=(
                getattr(language_planner, "execution_source", "live_api")
                if language_result is not None
                else "not-used"
            ),
            language_source_metrics=(
                getattr(language_planner, "source_metrics_path", None)
                if language_result is not None
                else None
            ),
        )

        trajectory = np.asarray(end_effector_positions)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].plot(trajectory[:, 0], trajectory[:, 1], color="blue", linewidth=1.2)
        axes[0].scatter(
            np.asarray(initial_objects)[:, 0],
            np.asarray(initial_objects)[:, 1],
            c=["blue", "green", "orange", "red"],
            marker="s",
            label="initial cubes",
        )
        displayed_targets = (
            targets[[0]] if cfg.place_blue_on_red else targets
        )
        axes[0].scatter(
            displayed_targets[:, 0],
            displayed_targets[:, 1],
            c="red" if cfg.place_blue_on_red else "orange",
            marker="x",
            label="blue-on-red target" if cfg.place_blue_on_red else "L targets",
        )
        axes[0].set_aspect("equal", adjustable="box")
        axes[0].set_title("End-effector MPC-CBF trajectory")
        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        displayed_indices = [index + 1 for index in cfg.cube_indices]
        axes[1].bar(displayed_indices, selected_errors)
        axes[1].axhline(0.04, color="red", linestyle="--", label="placement tolerance")
        axes[1].set_xticks(displayed_indices)
        axes[1].set_xlabel("cube")
        axes[1].set_ylabel("final position error [m]")
        axes[1].set_title(
            "Blue-on-red placement error"
            if cfg.place_blue_on_red
            else "Build-L placement error"
        )
        axes[1].grid(True, axis="y", alpha=0.3)
        axes[1].legend()
        fig.savefig(output_dir / "trajectory_and_placement.png", dpi=160)
        plt.close(fig)

        if frames:
            if cfg.place_blue_on_red and cfg.dynamic_obstacle:
                transport_frames = [
                    index
                    for index, stage in enumerate(frame_stages)
                    if stage == "cube_1_transport"
                ]
                display_frame_index = (
                    transport_frames[len(transport_frames) // 2]
                    if transport_frames
                    else len(frames) // 2
                )
                display_frame = frames[display_frame_index]
                height, width = display_frame.shape[:2]
                projection_options = {
                    "physics_client": env.sim.physics_client,
                    "width": width,
                    "height": height,
                    "target_position": env.render_target_position,
                    "distance": env.render_distance,
                    "yaw": env.render_yaw,
                    "pitch": env.render_pitch,
                    "roll": env.render_roll,
                }
                trajectory_pixels = _project_world_points(
                    trajectory, **projection_options
                )
                approach_mask = np.asarray([
                    stage in {"cube_1_approach", "cube_1_descend"}
                    for stage in trajectory_stages
                ])
                transport_mask = np.asarray([
                    stage in {
                        "cube_1_lift",
                        "cube_1_transport_avoid",
                        "cube_1_transport",
                        "cube_1_place",
                    }
                    for stage in trajectory_stages
                ])
                transport_world = trajectory[transport_mask]
                transport_pixels = trajectory_pixels[transport_mask]
                nominal_pixels = _project_world_points(
                    np.asarray([transport_world[0], transport_world[-1]]),
                    **projection_options,
                )
                displayed_obstacle = frame_obstacles[display_frame_index]
                angles = np.linspace(0.0, 2.0 * np.pi, 180)
                safety_radius = (
                    frame_cbf_radii[display_frame_index]
                    or cfg.obstacle_radius + cfg.gripper_collision_radius
                )
                safety_circle = np.column_stack([
                    displayed_obstacle[0] + safety_radius * np.cos(angles),
                    displayed_obstacle[1] + safety_radius * np.sin(angles),
                    np.full_like(angles, displayed_obstacle[2]),
                ])
                safety_pixels = _project_world_points(
                    safety_circle, **projection_options
                )
                paper_fig = plt.figure(figsize=(13.5, 7.2), facecolor="white")
                camera_axis = paper_fig.add_axes([0.02, 0.08, 0.70, 0.82])
                camera_axis.imshow(display_frame)
                camera_axis.plot(
                    trajectory_pixels[approach_mask, 0],
                    trajectory_pixels[approach_mask, 1],
                    color="#f28e2b",
                    linewidth=2.8,
                    label="actual approach trajectory",
                )
                camera_axis.plot(
                    nominal_pixels[:, 0],
                    nominal_pixels[:, 1],
                    color="#d62728",
                    linestyle=":",
                    linewidth=3.0,
                    label="nominal straight transport",
                )
                camera_axis.plot(
                    transport_pixels[:, 0],
                    transport_pixels[:, 1],
                    color="black",
                    linewidth=3.2,
                    label="executed MPC-CBF transport",
                )
                camera_axis.plot(
                    safety_pixels[:, 0],
                    safety_pixels[:, 1],
                    color="#ffd400",
                    linestyle="--",
                    linewidth=3.0,
                    label="active OD-CBF boundary h=0",
                )
                camera_axis.text(
                    0.02,
                    0.97,
                    f"User → LLM: {cfg.user_instruction}",
                    transform=camera_axis.transAxes,
                    va="top",
                    fontsize=11,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.45",
                        "facecolor": "#fff36d",
                        "edgecolor": "#27364a",
                        "alpha": 0.96,
                    },
                )
                camera_axis.legend(
                    loc="lower left", fontsize=8, framealpha=0.94
                )
                camera_axis.set_title(
                    "Language-guided Safe Panda pick-and-place",
                    fontsize=15,
                    fontweight="bold",
                )
                camera_axis.axis("off")

                box_style = {
                    "boxstyle": "round,pad=0.55",
                    "facecolor": "#f7fbff",
                    "edgecolor": "#27364a",
                    "linewidth": 1.4,
                }
                paper_fig.text(
                    0.75,
                    0.83,
                    "LLM optimization formulator\n"
                    f"model: {result.language_model if result.language_plan_used else cfg.llm_model}\n"
                    f"selected γ: {result.language_gammas if result.language_plan_used else (cfg.gamma,)}\n"
                    f"TP latency: {result.language_tp_latency_seconds:.3f} s\n"
                    f"OD latency: {result.language_od_latency_seconds:.3f} s\n"
                    f"OD fallbacks: {result.language_od_fallbacks}\n"
                    f"source: {result.language_execution_source}",
                    fontsize=10,
                    va="top",
                    bbox=box_style,
                )
                primitives = (
                    "1. move_to(blue_cube, safely)\n\n"
                    "2. close_gripper()\n\n"
                    "3. move_above(red_cube, safely)\n\n"
                    "4. open_gripper()"
                )
                paper_fig.text(
                    0.75,
                    0.55,
                    primitives,
                    fontsize=11,
                    va="top",
                    bbox=box_style,
                )
                paper_fig.text(
                    0.75,
                    0.27,
                    "Measured result\n"
                    f"goal success: {result.success}\n"
                    f"minimum raw clearance: "
                    f"{result.minimum_dynamic_clearance * 1000:.2f} mm\n"
                    f"minimum raw CBF margin: "
                    f"{result.minimum_dynamic_cbf_margin * 1000:.2f} mm\n"
                    f"MPC mean solve time: {result.mean_solve_time * 1000:.2f} ms",
                    fontsize=10,
                    va="top",
                    bbox=box_style,
                )
                paper_fig.text(
                    0.02,
                    0.015,
                    "Orange/black are raw executed robot trajectories; red is the "
                    "nominal direct route; yellow is the active OD-CBF set boundary, not a trajectory.",
                    fontsize=9,
                )
                paper_fig.savefig(
                    output_dir / "language_guided_mpc_cbf.png", dpi=180
                )
                plt.close(paper_fig)

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
            "trajectory_stages": trajectory_stages,
            "clearances": clearances,
            "dynamic_clearances": dynamic_clearances,
            "dynamic_cbf_margins": dynamic_cbf_margins,
            "dynamic_cbf_margin_stages": dynamic_cbf_margin_stages,
            "dynamic_cbf_boundary_radii": dynamic_cbf_boundary_radii,
            "obstacle_positions": [position.tolist() for position in obstacle_positions],
            "obstacle_measurements": [
                position.tolist() for position in obstacle_measurements
            ],
            "sensor_update_steps": sensor_update_steps,
            "solve_times": solve_times,
            "language_result": (
                asdict(language_result) if language_result is not None else None
            ),
            "grasp_model": "fixed PyBullet constraint after gripper closure",
            "trajectory_legend": {
                "orange": "raw executed approach trajectory",
                "red_dotted": "nominal straight transport reference",
                "black": "raw executed MPC-CBF transport trajectory",
                "yellow_dashed": "active OD-CBF safety boundary h=0 (not a trajectory)",
            },
            "language_control_boundary": (
                "Validated TP/OD JSON selects the pick/place macro and bounded MPC-CBF "
                "parameters; the trusted executor expands motion stages and MPC-CBF "
                "computes the trajectory"
            ),
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        return result
    finally:
        wrapped_env.close()
