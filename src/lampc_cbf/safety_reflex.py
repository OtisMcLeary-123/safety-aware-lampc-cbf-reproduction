"""High-rate operational-space CBF projection and short-horizon gatekeeper."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence


Vector3 = tuple[float, float, float]


def _vector3(values: Sequence[float], label: str) -> Vector3:
    if len(values) != 3:
        raise ValueError(f"{label} must contain three values")
    converted = tuple(float(value) for value in values)
    if any(not isfinite(value) for value in converted):
        raise ValueError(f"{label} must be finite")
    return converted  # type: ignore[return-value]


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: Vector3) -> float:
    return sqrt(_dot(vector, vector))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _unit(vector: Vector3) -> Vector3 | None:
    magnitude = _norm(vector)
    if magnitude <= 1e-12:
        return None
    return tuple(value / magnitude for value in vector)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ReflexObstacle:
    position: Vector3
    velocity: Vector3
    radius: float
    uncertainty: float = 0.0
    name: str = "obstacle"

    def __post_init__(self) -> None:
        _vector3(self.position, "obstacle position")
        _vector3(self.velocity, "obstacle velocity")
        if not isfinite(self.radius) or self.radius < 0.0:
            raise ValueError("obstacle radius must be finite and non-negative")
        if not isfinite(self.uncertainty) or self.uncertainty < 0.0:
            raise ValueError("obstacle uncertainty must be finite and non-negative")

    @property
    def robust_radius(self) -> float:
        return self.radius + self.uncertainty


@dataclass(frozen=True, slots=True)
class SafetyReflexConfig:
    dt: float = 0.04
    lookahead_steps: int = 8
    cbf_alpha: float = 4.0
    speed_limit: float = 0.2
    projection_passes: int = 8
    feasibility_tolerance: float = 1e-9
    uncertainty_growth_per_second: float = 0.035
    uncertainty_acceleration_bound: float = 0.0
    backup_selection: str = "task_consistent"
    committed_backup_enabled: bool = True
    committed_backup_steps: int = 8
    barrier_mode: str = "radial_cbf"
    side_latch_enabled: bool = False
    side_latch_steps: int = 15
    side_release_clearance: float = 0.05
    side_switch_hysteresis: float = 0.01
    policy_library_enabled: bool = False
    policy_speed_scales: tuple[float, ...] = (0.50, 0.75, 1.0)
    policy_progress_weight: float = 4.0
    policy_deviation_weight: float = 1.0
    policy_switch_penalty: float = 0.05
    recovery_clearance_slack: float = 0.005
    tangential_subgoal_enabled: bool = False
    tangential_subgoal_distance: float = 0.10
    dpcbf_safety_scale: float = 1.05
    dpcbf_lambda_gain: float = 0.10
    dpcbf_mu_gain: float = 0.50
    dpcbf_distance_epsilon: float = 0.10
    dpcbf_velocity_epsilon: float = 0.05

    def __post_init__(self) -> None:
        if not isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if self.lookahead_steps < 1 or self.projection_passes < 1:
            raise ValueError("lookahead_steps and projection_passes must be positive")
        values = (
            self.cbf_alpha,
            self.speed_limit,
            self.feasibility_tolerance,
            self.uncertainty_growth_per_second,
            self.uncertainty_acceleration_bound,
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("reflex parameters must be finite and non-negative")
        if self.cbf_alpha == 0.0 or self.speed_limit == 0.0:
            raise ValueError("cbf_alpha and speed_limit must be positive")
        if self.backup_selection not in {"max_clearance", "task_consistent"}:
            raise ValueError(
                "backup_selection must be max_clearance or task_consistent"
            )
        if self.committed_backup_steps < 1:
            raise ValueError("committed_backup_steps must be positive")
        if self.barrier_mode not in {
            "radial_cbf",
            "collision_cone",
            "dynamic_parabolic",
        }:
            raise ValueError(
                "barrier_mode must be radial_cbf, collision_cone, or "
                "dynamic_parabolic"
            )
        if self.side_latch_steps < 1:
            raise ValueError("side_latch_steps must be positive")
        if any(
            not isfinite(value) or value < 0.0
            for value in (
                self.side_release_clearance,
                self.side_switch_hysteresis,
                self.policy_progress_weight,
                self.policy_deviation_weight,
                self.policy_switch_penalty,
                self.recovery_clearance_slack,
                self.tangential_subgoal_distance,
                self.dpcbf_lambda_gain,
                self.dpcbf_mu_gain,
                self.dpcbf_safety_scale,
                self.dpcbf_distance_epsilon,
                self.dpcbf_velocity_epsilon,
            )
        ):
            raise ValueError("policy-library parameters must be finite and non-negative")
        if not self.policy_speed_scales or any(
            not isfinite(value) or not 0.0 < value <= 1.0
            for value in self.policy_speed_scales
        ):
            raise ValueError("policy_speed_scales must be non-empty values in (0, 1]")
        if self.dpcbf_safety_scale <= 1.0:
            raise ValueError("dpcbf_safety_scale must be greater than one")
        if self.dpcbf_distance_epsilon <= 0.0 or self.dpcbf_velocity_epsilon <= 0.0:
            raise ValueError("DPCBF smoothing epsilons must be positive")
        if self.policy_library_enabled and self.barrier_mode not in {
            "collision_cone",
            "dynamic_parabolic",
        }:
            raise ValueError(
                "policy_library_enabled requires collision_cone or "
                "dynamic_parabolic mode"
            )
        if self.tangential_subgoal_enabled and not self.policy_library_enabled:
            raise ValueError("tangential subgoals require the policy library")


@dataclass(frozen=True, slots=True)
class GatekeeperResult:
    velocity: Vector3
    intervened: bool
    backup_used: bool
    reason: str
    nominal_minimum_clearance: float
    filtered_minimum_clearance: float
    maximum_cbf_violation: float
    selected_policy: str = "nominal"
    avoidance_side: int = 0
    side_switched: bool = False
    temporary_subgoal: Vector3 | None = None
    robust_recovery: bool = False


@dataclass(frozen=True, slots=True)
class _PolicyCandidate:
    name: str
    velocity: Vector3
    side: int = 0


class OperationalSpaceSafetyReflex:
    """Velocity-level spherical CBF safety filter with rollout gating.

    This is a local operational-space reflex, not a whole-body proof: joint,
    link, self-collision and actuator dynamics remain outside this simplified
    Safe Panda Gym reproduction.
    """

    def __init__(self, config: SafetyReflexConfig | None = None) -> None:
        self.config = config or SafetyReflexConfig()
        self._committed_backup: tuple[Vector3, ...] = ()
        self._latched_side = 0
        self._side_steps_remaining = 0
        self._side_switches = 0
        self._last_policy = "nominal"

    @property
    def committed_backup_steps_remaining(self) -> int:
        return len(self._committed_backup)

    @property
    def latched_side(self) -> int:
        return self._latched_side

    @property
    def side_switches(self) -> int:
        return self._side_switches

    def reset(self) -> None:
        """Release state retained by the gatekeeper-style backup policy."""

        self._committed_backup = ()
        self._latched_side = 0
        self._side_steps_remaining = 0
        self._side_switches = 0
        self._last_policy = "nominal"

    def _release_committed(self) -> None:
        self._committed_backup = ()

    def _advance_side_latch(self, nominal_clearance: float) -> None:
        if self._side_steps_remaining > 0:
            self._side_steps_remaining -= 1
        if (
            self._side_steps_remaining == 0
            and nominal_clearance >= self.config.side_release_clearance
        ):
            self._latched_side = 0
            self._last_policy = "nominal"

    def _set_latched_side(self, side: int) -> bool:
        if not self.config.side_latch_enabled or side == 0:
            return False
        switched = self._latched_side not in (0, side)
        if switched:
            self._side_switches += 1
        self._latched_side = side
        self._side_steps_remaining = self.config.side_latch_steps
        return switched

    def _commit(self, velocity: Vector3) -> None:
        self._committed_backup = tuple(
            velocity for _ in range(self.config.committed_backup_steps)
        )

    def _consume_committed(self) -> Vector3:
        velocity = self._committed_backup[0]
        self._committed_backup = self._committed_backup[1:]
        return velocity

    def _limit(self, velocity: Vector3) -> Vector3:
        magnitude = _norm(velocity)
        if magnitude <= self.config.speed_limit:
            return velocity
        scale = self.config.speed_limit / magnitude
        return tuple(scale * item for item in velocity)  # type: ignore[return-value]

    def cbf_residual(
        self, position: Sequence[float], velocity: Sequence[float], obstacle: ReflexObstacle
    ) -> float:
        point = _vector3(position, "position")
        command = _vector3(velocity, "velocity")
        displacement = tuple(a - b for a, b in zip(point, obstacle.position))
        relative_velocity = tuple(
            a - b for a, b in zip(command, obstacle.velocity)
        )
        h = _dot(displacement, displacement) - obstacle.robust_radius**2
        return 2.0 * _dot(displacement, relative_velocity) + self.config.cbf_alpha * h

    def collision_cone_residual(
        self, position: Sequence[float], velocity: Sequence[float], obstacle: ReflexObstacle
    ) -> float:
        """Evaluate the velocity-obstacle collision-cone barrier.

        A non-negative value places the relative velocity outside the collision
        cone. The robust obstacle radius already contains measurement and
        sampled-data uncertainty.
        """

        point = _vector3(position, "position")
        command = _vector3(velocity, "velocity")
        displacement = tuple(a - b for a, b in zip(point, obstacle.position))
        distance_squared = _dot(displacement, displacement)
        radius_squared = obstacle.robust_radius**2
        if distance_squared <= radius_squared:
            return distance_squared - radius_squared
        relative_velocity = tuple(
            a - b for a, b in zip(command, obstacle.velocity)
        )
        cone_tangent = sqrt(max(0.0, distance_squared - radius_squared))
        return (
            _dot(displacement, relative_velocity)
            + _norm(relative_velocity) * cone_tangent
        )

    def dynamic_parabolic_residual(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        obstacle: ReflexObstacle,
    ) -> float:
        """Evaluate a Cartesian 3D adaptation of safe_control's DPCBF.

        The original implementation is for a planar kinematic bicycle.  Here
        the line-of-sight longitudinal speed is retained and its single lateral
        component is generalized to the squared norm of the 3D perpendicular
        relative velocity.  This is an experimental barrier ablation, not a
        claim that the nonholonomic proof transfers to the Panda model.
        """

        point = _vector3(position, "position")
        command = _vector3(velocity, "velocity")
        line_of_sight = tuple(
            obstacle_axis - robot_axis
            for obstacle_axis, robot_axis in zip(obstacle.position, point)
        )
        distance = _norm(line_of_sight)
        if distance <= 1e-12:
            return -obstacle.robust_radius
        los_unit = tuple(value / distance for value in line_of_sight)
        relative_velocity = tuple(
            obstacle_speed - robot_speed
            for obstacle_speed, robot_speed in zip(obstacle.velocity, command)
        )
        longitudinal = _dot(relative_velocity, los_unit)
        lateral_squared = max(
            0.0,
            _dot(relative_velocity, relative_velocity) - longitudinal**2,
        )
        scaled_radius = (
            obstacle.robust_radius * self.config.dpcbf_safety_scale
        )
        smooth_clearance = sqrt(
            max(
                0.0,
                distance**2
                - scaled_radius**2
                + self.config.dpcbf_distance_epsilon**2,
            )
        ) - self.config.dpcbf_distance_epsilon
        smooth_speed = sqrt(
            _dot(relative_velocity, relative_velocity)
            + self.config.dpcbf_velocity_epsilon**2
        )
        adaptive_scale = (
            sqrt(self.config.dpcbf_safety_scale**2 - 1.0) / scaled_radius
        )
        lambda_value = (
            self.config.dpcbf_lambda_gain
            * smooth_clearance
            / smooth_speed
            * adaptive_scale
        )
        mu_value = (
            self.config.dpcbf_mu_gain
            * smooth_clearance
            * adaptive_scale
        )
        return longitudinal + lambda_value * lateral_squared + mu_value

    def barrier_residual(
        self, position: Sequence[float], velocity: Sequence[float], obstacle: ReflexObstacle
    ) -> float:
        if self.config.barrier_mode == "collision_cone":
            return self.collision_cone_residual(position, velocity, obstacle)
        if self.config.barrier_mode == "dynamic_parabolic":
            return self.dynamic_parabolic_residual(position, velocity, obstacle)
        return self.cbf_residual(position, velocity, obstacle)

    def _collision_cone_projection_candidates(
        self,
        point: Vector3,
        velocity: Vector3,
        obstacle: ReflexObstacle,
    ) -> tuple[Vector3, ...]:
        displacement = tuple(a - b for a, b in zip(point, obstacle.position))
        distance = _norm(displacement)
        if distance <= obstacle.robust_radius + 1e-12:
            away = _unit(displacement) or (1.0, 0.0, 0.0)
            return (
                tuple(self.config.speed_limit * axis for axis in away),
                (0.0, 0.0, 0.0),
            )
        axis = tuple(-value / distance for value in displacement)
        relative = tuple(a - b for a, b in zip(velocity, obstacle.velocity))
        axial = _dot(relative, axis)
        lateral = tuple(value - axial * direction for value, direction in zip(relative, axis))
        lateral_unit = _unit(lateral)
        if lateral_unit is None:
            lateral_unit = _unit(_cross(axis, (0.0, 0.0, 1.0)))
        if lateral_unit is None:
            lateral_unit = _unit(_cross(axis, (0.0, 1.0, 0.0)))
        assert lateral_unit is not None
        sine = min(1.0, obstacle.robust_radius / distance)
        cosine = sqrt(max(0.0, 1.0 - sine**2))
        candidates: list[Vector3] = []
        for sign in (-1.0, 1.0):
            boundary = tuple(
                cosine * forward + sign * sine * side
                for forward, side in zip(axis, lateral_unit)
            )
            magnitude = max(0.0, _dot(relative, boundary))
            relative_boundary = tuple(magnitude * value for value in boundary)
            candidates.append(
                self._limit(
                    tuple(
                        obs_speed + rel_speed
                        for obs_speed, rel_speed in zip(
                            obstacle.velocity, relative_boundary
                        )
                    )
                )
            )
        away = tuple(-value for value in axis)
        candidates.extend(
            (
                self._limit(obstacle.velocity),
                tuple(self.config.speed_limit * value for value in away),
            )
        )
        return tuple(candidates)

    def _tangent_direction(
        self, point: Vector3, obstacle: ReflexObstacle
    ) -> Vector3:
        """Return a deterministic positive circulation direction.

        The Safe Panda task is predominantly planar, so world ``+z`` defines
        the circulation orientation.  The fallbacks keep the construction
        well-defined for vertical alignments without changing the side label.
        """

        toward = _unit(tuple(a - b for a, b in zip(obstacle.position, point)))
        if toward is None:
            toward = (1.0, 0.0, 0.0)
        tangent = _unit(_cross((0.0, 0.0, 1.0), toward))
        if tangent is None:
            tangent = _unit(_cross((0.0, 1.0, 0.0), toward))
        return tangent or (1.0, 0.0, 0.0)

    def _candidate_side(
        self,
        point: Vector3,
        velocity: Vector3,
        obstacle: ReflexObstacle,
    ) -> int:
        tangent = self._tangent_direction(point, obstacle)
        relative = tuple(a - b for a, b in zip(velocity, obstacle.velocity))
        circulation = _dot(relative, tangent)
        if abs(circulation) <= 1e-10:
            return 0
        return 1 if circulation > 0.0 else -1

    def _policy_candidates(
        self,
        point: Vector3,
        nominal: Vector3,
        obstacles: Sequence[ReflexObstacle],
        *,
        include_library: bool,
        goal: Vector3 | None,
    ) -> tuple[_PolicyCandidate, ...]:
        candidates: list[_PolicyCandidate] = []
        projected, _ = self.project(point, nominal, obstacles)
        if obstacles:
            side = self._candidate_side(point, projected, obstacles[0])
        else:
            side = 0
        candidates.append(_PolicyCandidate("cone_projection", projected, side))
        if not include_library:
            return tuple(candidates)

        for obstacle_index, obstacle in enumerate(obstacles):
            tangent = self._tangent_direction(point, obstacle)
            toward = _unit(
                tuple(a - b for a, b in zip(obstacle.position, point))
            ) or (1.0, 0.0, 0.0)
            away = tuple(-value for value in toward)
            goal_direction = (
                _unit(tuple(a - b for a, b in zip(goal, point)))
                if goal is not None
                else None
            ) or _unit(nominal) or (0.0, 1.0, 0.0)
            for side in (-1, 1):
                signed_tangent = tuple(side * value for value in tangent)
                for scale in self.config.policy_speed_scales:
                    relative_speed = self.config.speed_limit * scale
                    world_tangent_velocity = tuple(
                        relative_speed * direction
                        for direction in signed_tangent
                    )
                    candidates.append(
                        _PolicyCandidate(
                            f"world_tangent_{obstacle_index}_{side:+d}_{scale:.2f}",
                            world_tangent_velocity,  # type: ignore[arg-type]
                            side,
                        )
                    )
                    goal_tangent_direction = _unit(
                        tuple(
                            goal_axis + 1.25 * tangent_axis + 0.20 * away_axis
                            for goal_axis, tangent_axis, away_axis in zip(
                                goal_direction, signed_tangent, away
                            )
                        )
                    )
                    assert goal_tangent_direction is not None
                    candidates.append(
                        _PolicyCandidate(
                            f"goal_tangent_{obstacle_index}_{side:+d}_{scale:.2f}",
                            tuple(
                                relative_speed * direction
                                for direction in goal_tangent_direction
                            ),  # type: ignore[arg-type]
                            side,
                        )
                    )
                    tangent_velocity = self._limit(
                        tuple(
                            obstacle_speed + relative_speed * direction
                            for obstacle_speed, direction in zip(
                                obstacle.velocity, signed_tangent
                            )
                        )
                    )
                    candidates.append(
                        _PolicyCandidate(
                            f"tangent_{obstacle_index}_{side:+d}_{scale:.2f}",
                            tangent_velocity,
                            side,
                        )
                    )
                    escape_direction = _unit(
                        tuple(
                            direction + 0.35 * radial
                            for direction, radial in zip(signed_tangent, away)
                        )
                    )
                    assert escape_direction is not None
                    escape_velocity = self._limit(
                        tuple(
                            obstacle_speed + relative_speed * direction
                            for obstacle_speed, direction in zip(
                                obstacle.velocity, escape_direction
                            )
                        )
                    )
                    candidates.append(
                        _PolicyCandidate(
                            f"circulation_{obstacle_index}_{side:+d}_{scale:.2f}",
                            escape_velocity,
                            side,
                        )
                    )
            candidates.append(
                _PolicyCandidate(
                    f"match_obstacle_{obstacle_index}",
                    self._limit(obstacle.velocity),
                    0,
                )
            )
        candidates.append(_PolicyCandidate("stop", (0.0, 0.0, 0.0), 0))

        unique: list[_PolicyCandidate] = []
        seen: set[tuple[float, float, float]] = set()
        for candidate in candidates:
            key = tuple(round(value, 12) for value in candidate.velocity)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return tuple(unique)

    def _policy_score(
        self,
        candidate: _PolicyCandidate,
        point: Vector3,
        nominal: Vector3,
        goal: Vector3 | None,
    ) -> float:
        horizon = self.config.dt * self.config.lookahead_steps
        task_deviation = sum(
            (value - target) ** 2
            for value, target in zip(candidate.velocity, nominal)
        )
        if goal is None:
            progress = horizon * _dot(candidate.velocity, nominal)
        else:
            current_distance = _norm(tuple(a - b for a, b in zip(point, goal)))
            endpoint = tuple(
                value + horizon * speed
                for value, speed in zip(point, candidate.velocity)
            )
            progress = current_distance - _norm(
                tuple(a - b for a, b in zip(endpoint, goal))
            )
        switch_cost = (
            self.config.policy_switch_penalty
            if self._latched_side not in (0, candidate.side)
            else 0.0
        )
        neutral_cost = (
            self.config.policy_switch_penalty
            if candidate.side == 0 and self._latched_side != 0
            else 0.0
        )
        return (
            -self.config.policy_progress_weight * progress
            + self.config.policy_deviation_weight * task_deviation
            + switch_cost
            + neutral_cost
        )

    def _temporary_subgoal(
        self,
        point: Vector3,
        obstacle: ReflexObstacle,
        side: int,
        goal: Vector3 | None,
    ) -> Vector3 | None:
        if not self.config.tangential_subgoal_enabled or side == 0:
            return None
        tangent = self._tangent_direction(point, obstacle)
        away = _unit(tuple(a - b for a, b in zip(point, obstacle.position)))
        away = away or (1.0, 0.0, 0.0)
        goal_direction = (
            _unit(tuple(a - b for a, b in zip(goal, point)))
            if goal is not None
            else None
        ) or (0.0, 0.0, 0.0)
        direction = _unit(
            tuple(
                0.75 * side * tangent_axis
                + goal_axis
                + 0.15 * away_axis
                for tangent_axis, goal_axis, away_axis in zip(
                    tangent, goal_direction, away
                )
            )
        )
        assert direction is not None
        return tuple(
            value + self.config.tangential_subgoal_distance * axis
            for value, axis in zip(point, direction)
        )  # type: ignore[return-value]

    def _select_collision_cone_policy(
        self,
        point: Vector3,
        nominal: Vector3,
        obstacles: Sequence[ReflexObstacle],
        goal: Vector3 | None,
    ) -> tuple[_PolicyCandidate, float, float, bool, bool] | None:
        candidates = self._policy_candidates(
            point,
            nominal,
            obstacles,
            include_library=self.config.policy_library_enabled,
            goal=goal,
        )
        evaluated: list[tuple[float, float, float, _PolicyCandidate]] = []
        recovery: list[tuple[float, float, float, _PolicyCandidate]] = []
        for candidate in candidates:
            clearance = self.rollout_minimum_clearance(
                point, candidate.velocity, obstacles
            )
            violation = max(
                (
                    max(
                        0.0,
                        -self.barrier_residual(
                            point, candidate.velocity, obstacle
                        ),
                    )
                    for obstacle in obstacles
                ),
                default=0.0,
            )
            if (
                clearance >= 0.0
                and violation <= self.config.feasibility_tolerance
            ):
                evaluated.append(
                    (
                        self._policy_score(candidate, point, nominal, goal),
                        clearance,
                        violation,
                        candidate,
                    )
                )
            elif self.config.policy_library_enabled:
                physical_obstacles = tuple(
                    ReflexObstacle(
                        obstacle.position,
                        obstacle.velocity,
                        obstacle.radius,
                        0.0,
                        obstacle.name,
                    )
                    for obstacle in obstacles
                )
                physical_clearance = self.rollout_minimum_clearance(
                    point,
                    candidate.velocity,
                    physical_obstacles,
                    include_uncertainty_growth=False,
                )
                physical_violation = max(
                    (
                        max(
                            0.0,
                            -self.barrier_residual(
                                point, candidate.velocity, obstacle
                            ),
                        )
                        for obstacle in physical_obstacles
                    ),
                    default=0.0,
                )
                if (
                    physical_clearance >= 0.0
                    and physical_violation <= self.config.feasibility_tolerance
                ):
                    recovery.append(
                        (
                            clearance,
                            violation,
                            self._policy_score(
                                candidate, point, nominal, goal
                            ),
                            candidate,
                        )
                    )
        if not evaluated and not recovery:
            return None

        current_side = self._latched_side if self.config.side_latch_enabled else 0
        if not evaluated:
            same_side_recovery = [
                item for item in recovery if item[3].side in (0, current_side)
            ] if current_side else recovery
            sided_recovery = [item for item in same_side_recovery if item[3].side]
            pool = sided_recovery or same_side_recovery or recovery
            best_clearance = max(item[0] for item in pool)
            near_best = [
                item
                for item in pool
                if item[0]
                >= best_clearance - self.config.recovery_clearance_slack
            ]
            selected_recovery = min(
                near_best,
                key=lambda item: (
                    item[2],
                    item[1],
                    -item[0],
                    item[3].name,
                ),
            )
            switched = self._set_latched_side(selected_recovery[3].side)
            self._last_policy = selected_recovery[3].name
            return (
                selected_recovery[3],
                selected_recovery[0],
                selected_recovery[1],
                switched,
                True,
            )
        same_side = [
            item for item in evaluated if item[3].side in (0, current_side)
        ] if current_side else evaluated
        opposite = [
            item for item in evaluated if item[3].side not in (0, current_side)
        ] if current_side else []
        selected = min(same_side or evaluated, key=lambda item: (item[0], -item[1], item[3].name))
        if opposite and self._side_steps_remaining == 0:
            best_opposite = min(
                opposite, key=lambda item: (item[0], -item[1], item[3].name)
            )
            if best_opposite[0] + self.config.side_switch_hysteresis < selected[0]:
                selected = best_opposite
        switched = self._set_latched_side(selected[3].side)
        self._last_policy = selected[3].name
        return selected[3], selected[1], selected[2], switched, False

    def project(
        self,
        position: Sequence[float],
        nominal_velocity: Sequence[float],
        obstacles: Sequence[ReflexObstacle],
    ) -> tuple[Vector3, float]:
        """Project onto radial-CBF half-spaces or collision-cone boundaries."""

        point = _vector3(position, "position")
        velocity = self._limit(_vector3(nominal_velocity, "nominal_velocity"))
        if self.config.barrier_mode in {"collision_cone", "dynamic_parabolic"}:
            for _ in range(self.config.projection_passes):
                changed = False
                for obstacle in obstacles:
                    if self.barrier_residual(point, velocity, obstacle) >= -self.config.feasibility_tolerance:
                        continue
                    candidates = self._collision_cone_projection_candidates(
                        point, velocity, obstacle
                    )
                    feasible = [
                        candidate
                        for candidate in candidates
                        if self.barrier_residual(point, candidate, obstacle)
                        >= -self.config.feasibility_tolerance
                    ]
                    pool = feasible or list(candidates)
                    velocity = min(
                        pool,
                        key=lambda candidate: (
                            sum(
                                (value - target) ** 2
                                for value, target in zip(candidate, velocity)
                            ),
                            -self.barrier_residual(point, candidate, obstacle),
                            candidate,
                        ),
                    )
                    changed = True
                if not changed:
                    break
            max_violation = max(
                (
                    max(0.0, -self.barrier_residual(point, velocity, obstacle))
                    for obstacle in obstacles
                ),
                default=0.0,
            )
            return velocity, max_violation
        for _ in range(self.config.projection_passes):
            changed = False
            for obstacle in obstacles:
                displacement = tuple(a - b for a, b in zip(point, obstacle.position))
                normal = tuple(2.0 * item for item in displacement)
                normal_squared = _dot(normal, normal)
                if normal_squared <= 1e-16:
                    # At the obstacle center no separating normal is defined.
                    continue
                h = _dot(displacement, displacement) - obstacle.robust_radius**2
                lower_bound = (
                    2.0 * _dot(displacement, obstacle.velocity)
                    - self.config.cbf_alpha * h
                )
                deficit = lower_bound - _dot(normal, velocity)
                if deficit > self.config.feasibility_tolerance:
                    velocity = tuple(
                        value + deficit / normal_squared * axis
                        for value, axis in zip(velocity, normal)
                    )  # type: ignore[assignment]
                    velocity = self._limit(velocity)
                    changed = True
            if not changed:
                break
        max_violation = max(
            (max(0.0, -self.barrier_residual(point, velocity, obstacle)) for obstacle in obstacles),
            default=0.0,
        )
        return velocity, max_violation

    def rollout_minimum_clearance(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        obstacles: Sequence[ReflexObstacle],
        *,
        include_uncertainty_growth: bool = True,
    ) -> float:
        point = _vector3(position, "position")
        command = _vector3(velocity, "velocity")
        minimum = float("inf")
        for step in range(1, self.config.lookahead_steps + 1):
            time = step * self.config.dt
            robot = tuple(value + time * speed for value, speed in zip(point, command))
            for obstacle in obstacles:
                center = tuple(
                    value + time * speed
                    for value, speed in zip(obstacle.position, obstacle.velocity)
                )
                radius = (
                    (
                        obstacle.robust_radius
                        + self.config.uncertainty_growth_per_second * time
                        + 0.5
                        * self.config.uncertainty_acceleration_bound
                        * time**2
                    )
                    if include_uncertainty_growth
                    else obstacle.radius
                )
                clearance = _norm(tuple(a - b for a, b in zip(robot, center))) - radius
                minimum = min(minimum, clearance)
        return minimum

    def gate(
        self,
        position: Sequence[float],
        nominal_velocity: Sequence[float],
        obstacles: Sequence[ReflexObstacle],
        goal_position: Sequence[float] | None = None,
    ) -> GatekeeperResult:
        point = _vector3(position, "position")
        nominal = self._limit(_vector3(nominal_velocity, "nominal_velocity"))
        goal = (
            _vector3(goal_position, "goal_position")
            if goal_position is not None
            else None
        )
        nominal_clearance = self.rollout_minimum_clearance(position, nominal, obstacles)
        self._advance_side_latch(nominal_clearance)
        nominal_violation = max(
            (max(0.0, -self.barrier_residual(position, nominal, obstacle)) for obstacle in obstacles),
            default=0.0,
        )
        if nominal_clearance >= 0.0 and nominal_violation <= self.config.feasibility_tolerance:
            self._release_committed()
            subgoal = (
                self._temporary_subgoal(
                    point, obstacles[0], self._latched_side, goal
                )
                if obstacles
                else None
            )
            return GatekeeperResult(
                nominal, False, False, "nominal_safe", nominal_clearance,
                nominal_clearance, nominal_violation,
                selected_policy=self._last_policy,
                avoidance_side=self._latched_side,
                temporary_subgoal=subgoal,
            )

        if (
            self.config.barrier_mode in {"collision_cone", "dynamic_parabolic"}
            and (self.config.side_latch_enabled or self.config.policy_library_enabled)
        ):
            selected = self._select_collision_cone_policy(
                point, nominal, obstacles, goal
            )
            if selected is not None:
                policy, clearance, violation, switched, robust_recovery = selected
                subgoal = (
                    self._temporary_subgoal(
                        point, obstacles[0], policy.side, goal
                    )
                    if obstacles
                    else None
                )
                return GatekeeperResult(
                    policy.velocity,
                    True,
                    policy.name != "cone_projection",
                    (
                        (
                            "policy_library_recovery"
                            if robust_recovery
                            else "policy_library"
                        )
                        if self.config.policy_library_enabled
                        else "side_latched_projection"
                    ),
                    nominal_clearance,
                    clearance,
                    violation,
                    selected_policy=policy.name,
                    avoidance_side=policy.side,
                    side_switched=switched,
                    temporary_subgoal=subgoal,
                    robust_recovery=robust_recovery,
                )

        projected, violation = self.project(position, nominal, obstacles)
        projected_clearance = self.rollout_minimum_clearance(position, projected, obstacles)
        if projected_clearance >= 0.0 and violation <= self.config.feasibility_tolerance:
            self._release_committed()
            return GatekeeperResult(
                projected, True, False, "oscbf_projection", nominal_clearance,
                projected_clearance, violation,
            )

        # A second projection starting from zero is one backup candidate.  A
        # stationary fallback is not intrinsically safe for moving obstacles,
        # so deterministic escape directions are evaluated as well.
        backup, backup_violation = self.project(position, (0.0, 0.0, 0.0), obstacles)
        backup_clearance = self.rollout_minimum_clearance(position, backup, obstacles)
        candidates: list[Vector3] = [backup, projected, (0.0, 0.0, 0.0)]
        axes: tuple[Vector3, ...] = (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
        for obstacle in obstacles:
            away = _unit(tuple(a - b for a, b in zip(point, obstacle.position)))
            obstacle_direction = _unit(obstacle.velocity)
            directions = list(axes)
            if away is not None:
                directions.extend((away, tuple(-value for value in away)))
            if obstacle_direction is not None:
                directions.extend(
                    (obstacle_direction, tuple(-value for value in obstacle_direction))
                )
                lateral = _unit(_cross(obstacle_direction, (0.0, 0.0, 1.0)))
                if lateral is not None:
                    directions.extend((lateral, tuple(-value for value in lateral)))
            if away is not None and obstacle_direction is not None:
                lateral = _unit(_cross(away, obstacle_direction))
                if lateral is not None:
                    directions.extend((lateral, tuple(-value for value in lateral)))
            candidates.extend(
                tuple(self.config.speed_limit * value for value in direction)
                for direction in directions
            )

        evaluated: list[tuple[float, float, Vector3]] = []
        seen: set[tuple[float, float, float]] = set()
        for candidate in candidates:
            limited = self._limit(candidate)
            key = tuple(round(value, 12) for value in limited)
            if key in seen:
                continue
            seen.add(key)
            clearance = self.rollout_minimum_clearance(point, limited, obstacles)
            violation = max(
                (
                    max(0.0, -self.barrier_residual(point, limited, obstacle))
                    for obstacle in obstacles
                ),
                default=0.0,
            )
            evaluated.append((clearance, violation, limited))
        feasible = [
            item
            for item in evaluated
            if item[0] >= 0.0
            and item[1] <= self.config.feasibility_tolerance
        ]
        committed_evaluation: tuple[float, float, Vector3] | None = None
        if self.config.committed_backup_enabled and self._committed_backup:
            committed = self._committed_backup[0]
            committed_clearance = self.rollout_minimum_clearance(
                position, committed, obstacles
            )
            committed_violation = max(
                (
                    max(0.0, -self.barrier_residual(position, committed, obstacle))
                    for obstacle in obstacles
                ),
                default=0.0,
            )
            if (
                committed_clearance >= 0.0
                and committed_violation <= self.config.feasibility_tolerance
            ):
                committed_evaluation = (
                    committed_clearance,
                    committed_violation,
                    committed,
                )
            else:
                self._release_committed()
        if feasible:
            if self.config.backup_selection == "max_clearance":
                backup_clearance, backup_violation, backup = max(
                    feasible, key=lambda item: item[0]
                )
                reason = "max_clearance_escape"
            else:
                # OSCBF-style task consistency: safety is a hard filter above,
                # then the objective minimally changes the nominal task command.
                # Clearance is only a deterministic tie-break, never the primary
                # objective, so the reflex does not unnecessarily drive away
                # from the goal after several commands are already safe.
                backup_clearance, backup_violation, backup = min(
                    feasible,
                    key=lambda item: (
                        sum(
                            (value - target) ** 2
                            for value, target in zip(item[2], nominal)
                        ),
                        -item[0],
                        item[2],
                    ),
                )
                reason = "task_consistent_escape"
            if committed_evaluation is not None:
                committed_cost = sum(
                    (value - target) ** 2
                    for value, target in zip(committed_evaluation[2], nominal)
                )
                new_cost = sum(
                    (value - target) ** 2
                    for value, target in zip(backup, nominal)
                )
                # Keep the certified suffix unless the newly verified candidate
                # improves task tracking. This follows gatekeeper's update rule
                # more closely than blindly freezing an old escape command.
                if committed_cost <= new_cost + 1e-12:
                    backup_clearance, backup_violation, _ = committed_evaluation
                    backup = self._consume_committed()
                    reason = "committed_backup"
                    return GatekeeperResult(
                        backup, True, True, reason, nominal_clearance,
                        backup_clearance, backup_violation,
                    )
        else:
            if committed_evaluation is not None:
                backup_clearance, backup_violation, _ = committed_evaluation
                backup = self._consume_committed()
                return GatekeeperResult(
                    backup, True, True, "committed_backup", nominal_clearance,
                    backup_clearance, backup_violation,
                )
            backup_clearance, backup_violation, backup = max(
                evaluated, key=lambda item: (item[0], -item[1], _norm(item[2]))
            )
            reason = "best_effort_escape"
        if (
            feasible
            and self.config.committed_backup_enabled
        ):
            self._commit(backup)
            self._consume_committed()
            reason = f"{reason}_committed"
        return GatekeeperResult(
            backup, True, True, reason, nominal_clearance,
            backup_clearance, backup_violation,
        )
