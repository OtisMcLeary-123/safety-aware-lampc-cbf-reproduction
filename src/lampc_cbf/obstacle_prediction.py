"""Constant-velocity obstacle prediction with a deterministic uncertainty tube."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Sequence


def _vector3(values: Sequence[float], label: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{label} must contain three values")
    converted = tuple(float(value) for value in values)
    if any(not isfinite(value) for value in converted):
        raise ValueError(f"{label} must be finite")
    return converted  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class UncertaintyTubeConfig:
    """Bounds used to inflate an obstacle over the MPC prediction horizon.

    ``measurement_sigma`` is converted to a bounded working assumption using
    ``confidence_multiplier``.  This does not turn Gaussian noise into an
    absolute guarantee; the statistical assumption must be reported with any
    safety claim.
    """

    measurement_sigma: float = 0.005
    confidence_multiplier: float = 3.0
    velocity_error_bound: float = 0.03
    model_error_growth: float = 0.005
    max_relative_speed: float = 0.4
    total_latency: float = 0.04

    def __post_init__(self) -> None:
        values = (
            self.measurement_sigma,
            self.confidence_multiplier,
            self.velocity_error_bound,
            self.model_error_growth,
            self.max_relative_speed,
            self.total_latency,
        )
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("uncertainty tube values must be finite and non-negative")
        if self.confidence_multiplier == 0.0:
            raise ValueError("confidence_multiplier must be positive")

    @property
    def measurement_bound(self) -> float:
        return self.confidence_multiplier * self.measurement_sigma

    @property
    def latency_bound(self) -> float:
        return self.max_relative_speed * self.total_latency

    def inflation(self, prediction_age: float) -> float:
        """Return radial uncertainty at age ``prediction_age`` in meters."""

        age = float(prediction_age)
        if not isfinite(age) or age < 0.0:
            raise ValueError("prediction_age must be finite and non-negative")
        growth = (self.velocity_error_bound + self.model_error_growth) * age
        return self.measurement_bound + self.latency_bound + growth


def predict_constant_velocity(
    position: Sequence[float], velocity: Sequence[float], prediction_age: float
) -> tuple[float, float, float]:
    """Predict ``position + velocity * prediction_age`` in SI units."""

    point = _vector3(position, "position")
    speed = _vector3(velocity, "velocity")
    age = float(prediction_age)
    if not isfinite(age) or age < 0.0:
        raise ValueError("prediction_age must be finite and non-negative")
    return tuple(p + age * v for p, v in zip(point, speed))  # type: ignore[return-value]


@dataclass(slots=True)
class ConstantVelocityObserver:
    """Timestamped finite-difference observer with low-pass velocity filtering."""

    initial_position: tuple[float, float, float]
    initial_time: float = 0.0
    velocity_filter: float = 0.5
    position: tuple[float, float, float] = field(init=False)
    timestamp: float = field(init=False)
    velocity: tuple[float, float, float] = field(init=False)
    updates: int = field(init=False)

    def __post_init__(self) -> None:
        self.initial_position = _vector3(self.initial_position, "initial_position")
        if not isfinite(self.initial_time):
            raise ValueError("initial_time must be finite")
        if not 0.0 <= self.velocity_filter <= 1.0:
            raise ValueError("velocity_filter must be in [0, 1]")
        self.position = self.initial_position
        self.timestamp = float(self.initial_time)
        self.velocity = (0.0, 0.0, 0.0)
        self.updates = 1

    def observe(self, position: Sequence[float], timestamp: float) -> bool:
        point = _vector3(position, "position")
        observed_at = float(timestamp)
        if not isfinite(observed_at):
            raise ValueError("timestamp must be finite")
        if observed_at < self.timestamp:
            raise ValueError("observation timestamps must be monotone")
        if observed_at == self.timestamp:
            self.position = point
            return False
        dt = observed_at - self.timestamp
        raw = tuple((new - old) / dt for new, old in zip(point, self.position))
        alpha = self.velocity_filter
        self.velocity = tuple(
            alpha * measured + (1.0 - alpha) * previous
            for measured, previous in zip(raw, self.velocity)
        )  # type: ignore[assignment]
        self.position = point
        self.timestamp = observed_at
        self.updates += 1
        return True

    def predict(self, query_time: float) -> tuple[float, float, float]:
        age = float(query_time) - self.timestamp
        return predict_constant_velocity(self.position, self.velocity, max(0.0, age))
