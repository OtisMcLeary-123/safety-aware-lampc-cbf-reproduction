"""Set-bounded discrete Cartesian safety filter with an invariant escape backup.

The certificate is intentionally scoped to the Safe Panda task's spherical
end-effector geometry.  It does not certify links, joints, self-collision, or
unmodeled contact dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi, sin, sqrt
from typing import Sequence


Vector3 = tuple[float, float, float]


def _vector3(value: Sequence[float], label: str) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{label} must contain three values")
    result = tuple(float(item) for item in value)
    if any(not isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return result  # type: ignore[return-value]


def _add(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _scale(value: Vector3, factor: float) -> Vector3:
    return tuple(factor * item for item in value)  # type: ignore[return-value]


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(value: Vector3) -> float:
    return sqrt(_dot(value, value))


def _unit(value: Vector3) -> Vector3 | None:
    magnitude = _norm(value)
    if magnitude <= 1e-12:
        return None
    return _scale(value, 1.0 / magnitude)


@dataclass(frozen=True, slots=True)
class FormalSafetyConfig:
    dt: float = 0.04
    gamma: float = 0.15
    speed_limit: float = 0.4
    measurement_error_bound: float = 0.005
    obstacle_velocity_error_bound: float = 0.0
    obstacle_acceleration_bound: float = 0.0
    robot_transition_error_bound: float = 0.008
    obstacle_speed_bound: float = 0.20
    feasibility_tolerance: float = 1e-9
    angular_candidates: int = 32

    def __post_init__(self) -> None:
        if not isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not isfinite(self.gamma) or not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        values = (
            self.speed_limit,
            self.measurement_error_bound,
            self.obstacle_velocity_error_bound,
            self.obstacle_acceleration_bound,
            self.robot_transition_error_bound,
            self.obstacle_speed_bound,
            self.feasibility_tolerance,
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("formal safety bounds must be finite and non-negative")
        if self.speed_limit == 0.0 or self.angular_candidates < 8:
            raise ValueError("speed_limit must be positive and angular_candidates >= 8")


@dataclass(frozen=True, slots=True)
class FormalObstacle:
    position: Vector3
    velocity: Vector3
    radius: float

    def __post_init__(self) -> None:
        _vector3(self.position, "obstacle position")
        _vector3(self.velocity, "obstacle velocity")
        if not isfinite(self.radius) or self.radius < 0.0:
            raise ValueError("obstacle radius must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class FormalSafetyResult:
    velocity: Vector3
    intervened: bool
    one_step_certified: bool
    terminal_backup_certified: bool
    robust_cbf_residual: float
    backup_authority_margin: float


class FormalDiscreteSafetyFilter:
    """One-step MPS filter for a bounded-error Cartesian single integrator.

    The candidate is accepted only when its robust discrete-CBF residual is
    nonnegative and the predicted next state remains inside the invariant set
    of a maximum-speed radial escape backup.
    """

    def __init__(self, config: FormalSafetyConfig | None = None) -> None:
        self.config = config or FormalSafetyConfig()

    def _limit(self, velocity: Vector3) -> Vector3:
        magnitude = _norm(velocity)
        if magnitude <= self.config.speed_limit:
            return velocity
        return _scale(velocity, self.config.speed_limit / magnitude)

    @property
    def backup_authority_margin(self) -> float:
        cfg = self.config
        return (
            cfg.speed_limit
            - cfg.obstacle_speed_bound
            - cfg.obstacle_velocity_error_bound
            - cfg.robot_transition_error_bound / cfg.dt
        )

    def robust_cbf_residual(
        self,
        position: Sequence[float],
        velocity: Sequence[float],
        obstacle: FormalObstacle,
    ) -> float:
        cfg = self.config
        point = _vector3(position, "position")
        command = self._limit(_vector3(velocity, "velocity"))
        displacement = _subtract(point, obstacle.position)
        current_radius = obstacle.radius + cfg.measurement_error_bound
        h_current = _dot(displacement, displacement) - current_radius**2
        robot_next = _add(point, _scale(command, cfg.dt))
        obstacle_next = _add(obstacle.position, _scale(obstacle.velocity, cfg.dt))
        next_radius = (
            current_radius
            + cfg.robot_transition_error_bound
            + cfg.obstacle_velocity_error_bound * cfg.dt
            + 0.5 * cfg.obstacle_acceleration_bound * cfg.dt**2
        )
        next_displacement = _subtract(robot_next, obstacle_next)
        h_next = _dot(next_displacement, next_displacement) - next_radius**2
        return h_next - (1.0 - cfg.gamma) * h_current

    def terminal_backup_margin(
        self,
        position: Sequence[float],
        obstacle: FormalObstacle,
    ) -> float:
        cfg = self.config
        point = _vector3(position, "position")
        clearance = (
            _norm(_subtract(point, obstacle.position))
            - obstacle.radius
            - cfg.measurement_error_bound
        )
        return min(clearance, self.backup_authority_margin)

    def backup_velocity(
        self, position: Sequence[float], obstacle: FormalObstacle
    ) -> Vector3:
        """Return the maximum-authority radial escape backup policy."""

        point = _vector3(position, "position")
        away = _unit(_subtract(point, obstacle.position)) or (1.0, 0.0, 0.0)
        return _scale(away, self.config.speed_limit)

    def _next_terminal_margin(
        self, point: Vector3, velocity: Vector3, obstacle: FormalObstacle
    ) -> float:
        cfg = self.config
        robot_next = _add(point, _scale(velocity, cfg.dt))
        obstacle_next = FormalObstacle(
            _add(obstacle.position, _scale(obstacle.velocity, cfg.dt)),
            obstacle.velocity,
            obstacle.radius
            + cfg.robot_transition_error_bound
            + cfg.obstacle_velocity_error_bound * cfg.dt
            + 0.5 * cfg.obstacle_acceleration_bound * cfg.dt**2,
        )
        return self.terminal_backup_margin(robot_next, obstacle_next)

    def _boundary_candidate(
        self, point: Vector3, nominal: Vector3, obstacle: FormalObstacle
    ) -> Vector3:
        cfg = self.config
        center_next = _add(obstacle.position, _scale(obstacle.velocity, cfg.dt))
        current = _subtract(point, obstacle.position)
        current_radius = obstacle.radius + cfg.measurement_error_bound
        h_current = _dot(current, current) - current_radius**2
        next_radius = (
            current_radius
            + cfg.robot_transition_error_bound
            + cfg.obstacle_velocity_error_bound * cfg.dt
            + 0.5 * cfg.obstacle_acceleration_bound * cfg.dt**2
        )
        required_distance = sqrt(
            max(next_radius**2, next_radius**2 + (1.0 - cfg.gamma) * h_current)
        )
        velocity_center = _scale(_subtract(center_next, point), 1.0 / cfg.dt)
        direction = _unit(_subtract(nominal, velocity_center))
        if direction is None:
            direction = _unit(_subtract(point, obstacle.position)) or (1.0, 0.0, 0.0)
        return self._limit(
            _add(velocity_center, _scale(direction, required_distance / cfg.dt))
        )

    def _candidates(
        self, point: Vector3, nominal: Vector3, obstacle: FormalObstacle
    ) -> tuple[Vector3, ...]:
        backup = self.backup_velocity(point, obstacle)
        away = _unit(backup) or (1.0, 0.0, 0.0)
        candidates = [nominal, self._boundary_candidate(point, nominal, obstacle)]
        for scale in (0.5, 0.75, 1.0):
            candidates.append(_scale(away, scale * self.config.speed_limit))
            for index in range(self.config.angular_candidates):
                angle = 2.0 * pi * index / self.config.angular_candidates
                candidates.append(
                    (
                        scale * self.config.speed_limit * cos(angle),
                        scale * self.config.speed_limit * sin(angle),
                        0.0,
                    )
                )
        unique: dict[tuple[float, float, float], Vector3] = {}
        for candidate in candidates:
            limited = self._limit(candidate)
            unique[tuple(round(value, 12) for value in limited)] = limited
        return tuple(unique.values())

    def filter(
        self,
        position: Sequence[float],
        nominal_velocity: Sequence[float],
        obstacle: FormalObstacle,
    ) -> FormalSafetyResult:
        point = _vector3(position, "position")
        nominal = self._limit(_vector3(nominal_velocity, "nominal velocity"))
        candidates = self._candidates(point, nominal, obstacle)
        evaluated = [
            (
                self.robust_cbf_residual(point, candidate, obstacle),
                self._next_terminal_margin(point, candidate, obstacle),
                candidate,
            )
            for candidate in candidates
        ]
        feasible = [
            item for item in evaluated
            if item[0] >= -self.config.feasibility_tolerance
            and item[1] >= -self.config.feasibility_tolerance
        ]
        if feasible:
            residual, terminal_margin, selected = min(
                feasible,
                key=lambda item: (
                    sum(
                        (value - target) ** 2
                        for value, target in zip(item[2], nominal)
                    ),
                    -item[0],
                ),
            )
            certified = True
        else:
            # No command in the finite candidate library satisfies the
            # declared contract.  The caller must treat this result as a
            # certificate failure; ``selected`` is diagnostic only.
            residual, terminal_margin, selected = max(
                evaluated, key=lambda item: (min(item[0], item[1]), item[0])
            )
            certified = False
        return FormalSafetyResult(
            velocity=selected,
            intervened=any(
                abs(value - target) > 1e-12
                for value, target in zip(selected, nominal)
            ),
            one_step_certified=certified,
            terminal_backup_certified=terminal_margin >= -self.config.feasibility_tolerance,
            robust_cbf_residual=residual,
            backup_authority_margin=self.backup_authority_margin,
        )
