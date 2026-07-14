"""Deterministic context-aware safety scheduling around slow language updates.

The language model expresses intent.  This module keeps the time-critical
decision local by combining that intent with predicted time-to-collision and
the current solver-feasibility state.  It deliberately does not call a model,
an optimizer, or the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence


def _finite_nonnegative(value: float, label: str) -> float:
    converted = float(value)
    if not isfinite(converted) or converted < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return converted


@dataclass(frozen=True, slots=True)
class SafetyProfile:
    """Controller-facing interpretation of a semantic safety request."""

    gamma: float
    clearance_margin: float = 0.0
    speed_scale: float = 1.0
    emergency: bool = False
    reason: str = "nominal"

    def __post_init__(self) -> None:
        if not isfinite(self.gamma) or not 0.0 < self.gamma <= 0.15:
            raise ValueError("gamma must be finite and in (0, 0.15]")
        _finite_nonnegative(self.clearance_margin, "clearance_margin")
        if not isfinite(self.speed_scale) or not 0.0 < self.speed_scale <= 1.0:
            raise ValueError("speed_scale must be finite and in (0, 1]")
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")


@dataclass(frozen=True, slots=True)
class ContextAwareSafetyConfig:
    """Pre-registered thresholds for the deterministic local safety policy."""

    emergency_ttc: float = 1.0
    cautious_ttc: float = 2.5
    emergency_gamma: float = 0.02
    cautious_gamma: float = 0.05
    nominal_gamma: float = 0.15
    emergency_clearance_margin: float = 0.04
    cautious_clearance_margin: float = 0.02
    emergency_speed_scale: float = 0.35
    cautious_speed_scale: float = 0.65
    reaction_margin: float = 0.20

    def __post_init__(self) -> None:
        _finite_nonnegative(self.emergency_ttc, "emergency_ttc")
        _finite_nonnegative(self.cautious_ttc, "cautious_ttc")
        _finite_nonnegative(self.reaction_margin, "reaction_margin")
        if self.emergency_ttc >= self.cautious_ttc:
            raise ValueError("emergency_ttc must be smaller than cautious_ttc")
        SafetyProfile(
            self.emergency_gamma,
            self.emergency_clearance_margin,
            self.emergency_speed_scale,
            True,
            "validation",
        )
        SafetyProfile(
            self.cautious_gamma,
            self.cautious_clearance_margin,
            self.cautious_speed_scale,
            False,
            "validation",
        )
        SafetyProfile(self.nominal_gamma, reason="validation")


def feedback_has_causal_opportunity(
    predicted_ttc: float | None,
    response_latency: float,
    *,
    reaction_margin: float = 0.20,
) -> bool:
    """Return whether a delayed update can arrive before the predicted hazard."""

    latency = _finite_nonnegative(response_latency, "response_latency")
    margin = _finite_nonnegative(reaction_margin, "reaction_margin")
    if predicted_ttc is None:
        return False
    ttc = float(predicted_ttc)
    return isfinite(ttc) and ttc > latency + margin


def feedback_update_deadline(
    request_time: float,
    predicted_ttc: float,
    update_ttl: float,
    *,
    reaction_margin: float = 0.20,
) -> float:
    """Return the earliest TTL or hazard deadline for an async update."""

    requested = _finite_nonnegative(request_time, "request_time")
    ttc = _finite_nonnegative(predicted_ttc, "predicted_ttc")
    ttl = _finite_nonnegative(update_ttl, "update_ttl")
    margin = _finite_nonnegative(reaction_margin, "reaction_margin")
    if ttl <= 0.0:
        raise ValueError("update_ttl must be positive")
    return min(requested + ttl, requested + max(0.0, ttc - margin))


def constant_velocity_ttc(
    relative_position: Sequence[float],
    relative_velocity: Sequence[float],
    collision_radius: float,
) -> float | None:
    """Return first non-negative contact time for a constant-velocity pair."""

    if len(relative_position) != 3 or len(relative_velocity) != 3:
        raise ValueError("relative position and velocity must be 3-vectors")
    if collision_radius <= 0.0 or not isfinite(collision_radius):
        raise ValueError("collision_radius must be finite and positive")
    position = tuple(float(value) for value in relative_position)
    velocity = tuple(float(value) for value in relative_velocity)
    if not all(isfinite(value) for value in (*position, *velocity)):
        raise ValueError("relative state must be finite")
    a = sum(value * value for value in velocity)
    b = 2.0 * sum(p * v for p, v in zip(position, velocity))
    c = sum(value * value for value in position) - collision_radius**2
    if c <= 0.0:
        return 0.0
    discriminant = b * b - 4.0 * a * c
    if a <= 1e-12 or discriminant < 0.0:
        return None
    root = (-b - sqrt(discriminant)) / (2.0 * a)
    return root if root >= 0.0 else None


class ContextAwareSafetyScheduler:
    """Map intent and control context to a bounded controller safety profile."""

    def __init__(self, config: ContextAwareSafetyConfig | None = None) -> None:
        self.config = config or ContextAwareSafetyConfig()

    def select(
        self,
        *,
        predicted_ttc: float | None,
        requested_safety_level: int = 3,
        solver_feasible: bool = True,
    ) -> SafetyProfile:
        if requested_safety_level not in range(1, 6):
            raise ValueError("requested_safety_level must be in 1..5")
        if not solver_feasible:
            # Gamma relaxation is not a safety fallback.  The caller must use
            # the local gatekeeper instead of applying another MPC candidate.
            return SafetyProfile(
                self.config.nominal_gamma,
                speed_scale=self.config.emergency_speed_scale,
                emergency=True,
                reason="solver_infeasible_use_local_reflex",
            )
        ttc = float("inf") if predicted_ttc is None else float(predicted_ttc)
        if not isfinite(ttc) and predicted_ttc is not None:
            raise ValueError("predicted_ttc must be finite or None")
        intent_is_cautious = requested_safety_level <= 2
        if ttc <= self.config.emergency_ttc:
            return SafetyProfile(
                self.config.emergency_gamma,
                self.config.emergency_clearance_margin,
                self.config.emergency_speed_scale,
                True,
                "ttc_emergency",
            )
        if ttc <= self.config.cautious_ttc or intent_is_cautious:
            return SafetyProfile(
                self.config.cautious_gamma,
                self.config.cautious_clearance_margin,
                self.config.cautious_speed_scale,
                False,
                "ttc_or_language_caution",
            )
        return SafetyProfile(self.config.nominal_gamma, reason="nominal_context")
