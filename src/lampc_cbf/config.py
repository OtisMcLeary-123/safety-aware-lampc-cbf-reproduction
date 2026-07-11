"""Paper-aligned configuration for the LaMPC-CBF reproduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True, slots=True)
class PaperConfig:
    """Numerical values reported in the paper's experimental setup.

    The paper defines the CBF for ``0 < gamma <= 1`` and uses the narrower
    ``0 < gamma <= 0.15`` interval in its trajectory experiments.  Both
    bounds are retained so callers can distinguish theory from the
    reproduction's default operating range.
    """

    dt: float = 0.04
    horizon: int = 15
    gamma: float = 0.10
    experimental_gamma_max: float = 0.15
    theoretical_gamma_max: float = 1.0
    state_weights: tuple[float, ...] = field(default=(1.0,) * 8)
    linear_input_weight: float = 0.5
    angular_input_weight: float = 1e-5
    velocity_regularization_weight: float = 0.1
    rotation_regularization_weight: float = 5e-5

    def __post_init__(self) -> None:
        if not isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be a finite positive number")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if len(self.state_weights) != 8:
            raise ValueError("state_weights must contain eight entries")
        if any(not isfinite(value) or value < 0.0 for value in self.state_weights):
            raise ValueError("state_weights must be finite and non-negative")
        for name in (
            "linear_input_weight",
            "angular_input_weight",
            "velocity_regularization_weight",
            "rotation_regularization_weight",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 < self.experimental_gamma_max <= self.theoretical_gamma_max <= 1.0:
            raise ValueError("gamma bounds must satisfy 0 < experimental <= theoretical <= 1")
        self.validate_gamma(self.gamma, experimental=False)

    @property
    def prediction_duration(self) -> float:
        """Length of the prediction horizon in seconds."""

        return self.dt * self.horizon

    def validate_gamma(self, gamma: float, *, experimental: bool = True) -> float:
        """Return a valid gamma or raise ``ValueError``.

        Set ``experimental=False`` to accept the full theoretical CBF range.
        """

        maximum = (
            self.experimental_gamma_max if experimental else self.theoretical_gamma_max
        )
        if not isfinite(gamma) or not 0.0 < gamma <= maximum:
            qualifier = "experimental" if experimental else "theoretical"
            raise ValueError(f"gamma must be in the {qualifier} interval (0, {maximum}]")
        return gamma
