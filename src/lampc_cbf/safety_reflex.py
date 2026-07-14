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
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("reflex parameters must be finite and non-negative")
        if self.cbf_alpha == 0.0 or self.speed_limit == 0.0:
            raise ValueError("cbf_alpha and speed_limit must be positive")


@dataclass(frozen=True, slots=True)
class GatekeeperResult:
    velocity: Vector3
    intervened: bool
    backup_used: bool
    reason: str
    nominal_minimum_clearance: float
    filtered_minimum_clearance: float
    maximum_cbf_violation: float


class OperationalSpaceSafetyReflex:
    """Velocity-level spherical CBF safety filter with rollout gating.

    This is a local operational-space reflex, not a whole-body proof: joint,
    link, self-collision and actuator dynamics remain outside this simplified
    Safe Panda Gym reproduction.
    """

    def __init__(self, config: SafetyReflexConfig | None = None) -> None:
        self.config = config or SafetyReflexConfig()

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

    def project(
        self,
        position: Sequence[float],
        nominal_velocity: Sequence[float],
        obstacles: Sequence[ReflexObstacle],
    ) -> tuple[Vector3, float]:
        """Cyclically project onto all affine CBF half-spaces."""

        point = _vector3(position, "position")
        velocity = self._limit(_vector3(nominal_velocity, "nominal_velocity"))
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
            (max(0.0, -self.cbf_residual(point, velocity, obstacle)) for obstacle in obstacles),
            default=0.0,
        )
        return velocity, max_violation

    def rollout_minimum_clearance(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        obstacles: Sequence[ReflexObstacle],
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
                    obstacle.robust_radius
                    + self.config.uncertainty_growth_per_second * time
                )
                clearance = _norm(tuple(a - b for a, b in zip(robot, center))) - radius
                minimum = min(minimum, clearance)
        return minimum

    def gate(
        self,
        position: Sequence[float],
        nominal_velocity: Sequence[float],
        obstacles: Sequence[ReflexObstacle],
    ) -> GatekeeperResult:
        nominal = self._limit(_vector3(nominal_velocity, "nominal_velocity"))
        nominal_clearance = self.rollout_minimum_clearance(position, nominal, obstacles)
        nominal_violation = max(
            (max(0.0, -self.cbf_residual(position, nominal, obstacle)) for obstacle in obstacles),
            default=0.0,
        )
        if nominal_clearance >= 0.0 and nominal_violation <= self.config.feasibility_tolerance:
            return GatekeeperResult(
                nominal, False, False, "nominal_safe", nominal_clearance,
                nominal_clearance, nominal_violation,
            )

        projected, violation = self.project(position, nominal, obstacles)
        projected_clearance = self.rollout_minimum_clearance(position, projected, obstacles)
        if projected_clearance >= 0.0 and violation <= self.config.feasibility_tolerance:
            return GatekeeperResult(
                projected, True, False, "oscbf_projection", nominal_clearance,
                projected_clearance, violation,
            )

        # A second projection starting from zero is one backup candidate.  A
        # stationary fallback is not intrinsically safe for moving obstacles,
        # so deterministic escape directions are evaluated as well.
        backup, backup_violation = self.project(position, (0.0, 0.0, 0.0), obstacles)
        backup_clearance = self.rollout_minimum_clearance(position, backup, obstacles)
        point = _vector3(position, "position")
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
                    max(0.0, -self.cbf_residual(point, limited, obstacle))
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
        if feasible:
            backup_clearance, backup_violation, backup = max(
                feasible, key=lambda item: item[0]
            )
            reason = "escape_policy"
        else:
            backup_clearance, backup_violation, backup = max(
                evaluated, key=lambda item: (item[0], -item[1], _norm(item[2]))
            )
            reason = "best_effort_escape"
        return GatekeeperResult(
            backup, True, True, reason, nominal_clearance,
            backup_clearance, backup_violation,
        )
