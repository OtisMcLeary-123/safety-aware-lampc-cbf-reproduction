"""Dependency-free integration types shared by simulator and controller code."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, Sequence, runtime_checkable

Vector3 = tuple[float, float, float]
StateVector = tuple[float, float, float, float, float, float, float, float]
ActionVector = tuple[float, float, float, float]


def _require_finite(values: Sequence[float], label: str) -> None:
    if any(not isfinite(value) for value in values):
        raise ValueError(f"{label} values must be finite")


@dataclass(frozen=True, slots=True)
class RobotState:
    """Paper state ``[x, y, z, psi, dx, dy, dz, dpsi]``."""

    x: float
    y: float
    z: float
    psi: float
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    dpsi: float = 0.0

    def __post_init__(self) -> None:
        _require_finite(self.as_vector(), "state")

    @property
    def position(self) -> Vector3:
        return (self.x, self.y, self.z)

    def as_vector(self) -> StateVector:
        return (self.x, self.y, self.z, self.psi, self.dx, self.dy, self.dz, self.dpsi)

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> RobotState:
        if len(values) != 8:
            raise ValueError("a robot state must contain eight values")
        return cls(*values)


@dataclass(frozen=True, slots=True)
class ControlInput:
    """Cartesian and yaw-rate control input ``[ux, uy, uz, upsi]``."""

    ux: float
    uy: float
    uz: float
    upsi: float

    def __post_init__(self) -> None:
        _require_finite(self.as_vector(), "control input")

    def as_vector(self) -> ActionVector:
        return (self.ux, self.uy, self.uz, self.upsi)


@dataclass(frozen=True, slots=True)
class Obstacle:
    """Spherical obstacle observation used by the CBF integration boundary."""

    position: Vector3
    radius: float
    name: str = "obstacle"

    def __post_init__(self) -> None:
        if len(self.position) != 3:
            raise ValueError("obstacle position must contain three values")
        _require_finite(self.position, "obstacle position")
        if not isfinite(self.radius) or self.radius < 0.0:
            raise ValueError("obstacle radius must be finite and non-negative")


@runtime_checkable
class Controller(Protocol):
    """Small interface implemented by the do-mpc controller specialist."""

    @property
    def gamma(self) -> float:
        """Current discrete CBF decay parameter."""

    def set_gamma(self, gamma: float) -> None:
        """Update the online safety parameter without an external API call."""

    def control(self, state: RobotState, obstacles: Sequence[Obstacle]) -> ControlInput:
        """Return the first action of the receding-horizon solution."""


@runtime_checkable
class Environment(Protocol):
    """Boundary implemented by the Safe Panda Gym adapter specialist."""

    def observe(self) -> tuple[RobotState, Sequence[Obstacle]]:
        """Read the current robot state and sensed obstacles."""

    def step(self, action: ControlInput) -> bool:
        """Apply an action and return whether the episode has terminated."""

