"""CasADi and dependency-free forms of the paper's nonlinear terms.

The numerical helpers intentionally use only the Python standard library.  They
are useful for checking generated CasADi expressions and keep the core package
importable when the optional ``casadi`` dependency is not installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import sin
from numbers import Real
from typing import Any

try:  # CasADi is optional until the controller extra is installed.
    import casadi as ca
except ImportError:  # pragma: no cover - availability depends on the environment
    ca = None


CASADI_AVAILABLE = ca is not None


def _require_casadi() -> Any:
    if ca is None:
        raise ImportError(
            "CasADi is required for symbolic expressions; install the controller "
            "dependencies (including 'casadi')."
        )
    return ca


def _validate_radii(obstacle_radius: Real, collision_radius: Real) -> None:
    if obstacle_radius < 0 or collision_radius < 0:
        raise ValueError("obstacle_radius and collision_radius must be non-negative")


def _validate_gamma(gamma: Real) -> None:
    if not 0 < gamma <= 1:
        raise ValueError("gamma must satisfy the paper's condition 0 < gamma <= 1")


def barrier_value(
    position: Sequence[Real],
    obstacle_position: Sequence[Real],
    obstacle_radius: Real,
    collision_radius: Real,
) -> float:
    """Evaluate ``h(p) = ||p - p_obs||² - (r_obs + r_collision)²``."""

    _validate_radii(obstacle_radius, collision_radius)
    if len(position) != len(obstacle_position):
        raise ValueError("position and obstacle_position must have the same dimension")
    if not position:
        raise ValueError("positions must not be empty")
    distance_squared = sum(
        (float(coordinate) - float(obstacle_coordinate)) ** 2
        for coordinate, obstacle_coordinate in zip(position, obstacle_position)
    )
    return distance_squared - float(obstacle_radius + collision_radius) ** 2


def discrete_cbf_value(h_current: Real, h_next: Real, gamma: Real) -> float:
    """Return the CBF residual, feasible when it is non-negative.

    This is equation (12) rearranged as
    ``h_next - (1 - gamma) * h_current >= 0``.
    """

    _validate_gamma(gamma)
    return float(h_next) - (1.0 - float(gamma)) * float(h_current)


def quadratic_form_value(state: Sequence[Real], phi: Sequence[Sequence[Real]]) -> float:
    """Evaluate ``state.T @ phi @ state`` without NumPy."""

    dimension = len(state)
    if len(phi) != dimension or any(len(row) != dimension for row in phi):
        raise ValueError("phi must be a square matrix matching the state dimension")
    values = tuple(float(value) for value in state)
    return sum(
        values[row] * float(phi[row][column]) * values[column]
        for row in range(dimension)
        for column in range(dimension)
    )


def regularizer_value(
    state: Sequence[Real],
    phi: Sequence[Sequence[Real]],
    lambda3: Real,
    lambda4: Real,
    *,
    psi_index: int = 3,
) -> float:
    """Evaluate equation (19): ``lambda3*x.T*Phi*x + lambda4*sin²(2*psi)``."""

    if not -len(state) <= psi_index < len(state):
        raise IndexError("psi_index is outside the state vector")
    psi = float(state[psi_index])
    return float(lambda3) * quadratic_form_value(state, phi) + float(lambda4) * sin(
        2.0 * psi
    ) ** 2


def barrier_expression(
    position: Any,
    obstacle_position: Any,
    obstacle_radius: Any,
    collision_radius: Any,
) -> Any:
    """Build the CasADi form of the obstacle barrier in equation (13)."""

    casadi = _require_casadi()
    displacement = position - obstacle_position
    return casadi.sumsqr(displacement) - (obstacle_radius + collision_radius) ** 2


def discrete_cbf_expression(h_current: Any, h_next: Any, gamma: Any) -> Any:
    """Build the CasADi CBF residual constrained to be greater than or equal to zero."""

    _require_casadi()
    return h_next - (1 - gamma) * h_current


def regularizer_expression(
    state: Any,
    phi: Any,
    lambda3: Any,
    lambda4: Any,
    *,
    psi_index: int = 3,
) -> Any:
    """Build the CasADi stability/rotation regularizer from equation (19)."""

    casadi = _require_casadi()
    quadratic_term = casadi.mtimes([state.T, phi, state])
    return lambda3 * quadratic_term + lambda4 * casadi.sin(2 * state[psi_index]) ** 2

