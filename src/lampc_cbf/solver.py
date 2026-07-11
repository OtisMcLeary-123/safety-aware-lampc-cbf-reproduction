"""IPOPT configuration, diagnostics, and fail-closed control selection.

The helpers in this module are deliberately independent of CasADi.  A caller
may pass ``IpoptConfig.casadi_options()`` to CasADi or do-mpc and translate the
returned solver statistics with :func:`diagnostics_from_stats`.  Most
importantly, a candidate control is never considered usable solely because an
optimizer returned a vector: termination and independently measured primal
feasibility must both pass :class:`FeasibilityPolicy`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from numbers import Real
from typing import Any


@dataclass(frozen=True, slots=True)
class IpoptConfig:
    """Conservative IPOPT defaults for the paper's nonlinear MPC problem."""

    tolerance: float = 1e-8
    acceptable_tolerance: float = 1e-6
    constraint_violation_tolerance: float = 1e-6
    max_iterations: int = 200
    max_cpu_time: float = 0.035
    print_level: int = 0
    linear_solver: str = "mumps"
    mu_strategy: str = "adaptive"
    warm_start: bool = True
    print_time: bool = False

    def __post_init__(self) -> None:
        for name in (
            "tolerance",
            "acceptable_tolerance",
            "constraint_violation_tolerance",
            "max_cpu_time",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least one")
        if not 0 <= self.print_level <= 12:
            raise ValueError("print_level must be between 0 and 12")
        if not self.linear_solver.strip():
            raise ValueError("linear_solver must not be empty")
        if self.mu_strategy not in {"adaptive", "monotone"}:
            raise ValueError("mu_strategy must be 'adaptive' or 'monotone'")

    def casadi_options(self) -> dict[str, Any]:
        """Return options accepted by CasADi ``nlpsol`` and do-mpc."""

        options: dict[str, Any] = {
            "ipopt.tol": self.tolerance,
            "ipopt.acceptable_tol": self.acceptable_tolerance,
            "ipopt.constr_viol_tol": self.constraint_violation_tolerance,
            "ipopt.max_iter": self.max_iterations,
            "ipopt.max_cpu_time": self.max_cpu_time,
            "ipopt.print_level": self.print_level,
            "ipopt.linear_solver": self.linear_solver,
            "ipopt.mu_strategy": self.mu_strategy,
            "ipopt.sb": "yes" if self.print_level == 0 else "no",
            "print_time": self.print_time,
        }
        if self.warm_start:
            options.update(
                {
                    "ipopt.warm_start_init_point": "yes",
                    "ipopt.warm_start_bound_push": 1e-6,
                    "ipopt.warm_start_mult_bound_push": 1e-6,
                    "ipopt.warm_start_slack_bound_push": 1e-6,
                }
            )
        return options


class Termination(str, Enum):
    """Normalized IPOPT termination categories."""

    SOLVED = "solved"
    ACCEPTABLE = "acceptable"
    FEASIBLE_POINT = "feasible_point"
    MAX_ITERATIONS = "max_iterations"
    MAX_CPU_TIME = "max_cpu_time"
    INFEASIBLE = "infeasible"
    RESTORATION_FAILED = "restoration_failed"
    NUMERICAL_ERROR = "numerical_error"
    USER_STOPPED = "user_stopped"
    INVALID_PROBLEM = "invalid_problem"
    ERROR = "error"
    UNKNOWN = "unknown"


_IPOPT_TERMINATIONS = {
    "Solve_Succeeded": Termination.SOLVED,
    "Solved_To_Acceptable_Level": Termination.ACCEPTABLE,
    "Feasible_Point_Found": Termination.FEASIBLE_POINT,
    "Maximum_Iterations_Exceeded": Termination.MAX_ITERATIONS,
    "Maximum_CpuTime_Exceeded": Termination.MAX_CPU_TIME,
    "Infeasible_Problem_Detected": Termination.INFEASIBLE,
    "Restoration_Failed": Termination.RESTORATION_FAILED,
    "User_Requested_Stop": Termination.USER_STOPPED,
    "Invalid_Problem_Definition": Termination.INVALID_PROBLEM,
    "Not_Enough_Degrees_Of_Freedom": Termination.INVALID_PROBLEM,
    "Invalid_Option": Termination.INVALID_PROBLEM,
    "Invalid_Number_Detected": Termination.NUMERICAL_ERROR,
    "Search_Direction_Becomes_Too_Small": Termination.NUMERICAL_ERROR,
    "Diverging_Iterates": Termination.NUMERICAL_ERROR,
    "Error_In_Step_Computation": Termination.NUMERICAL_ERROR,
    "Unrecoverable_Exception": Termination.ERROR,
    "NonIpopt_Exception_Thrown": Termination.ERROR,
    "Insufficient_Memory": Termination.ERROR,
    "Internal_Error": Termination.ERROR,
}


def normalize_termination(return_status: object) -> Termination:
    """Normalize a CasADi/IPOPT ``return_status`` without fuzzy success tests."""

    if return_status is None:
        return Termination.UNKNOWN
    status = str(return_status).strip()
    if status in _IPOPT_TERMINATIONS:
        return _IPOPT_TERMINATIONS[status]
    # Some wrappers replace separators or alter case.
    canonical = status.lower().replace("-", "_").replace(" ", "_")
    for raw_status, normalized in _IPOPT_TERMINATIONS.items():
        if canonical == raw_status.lower():
            return normalized
    return Termination.UNKNOWN


def _optional_finite_float(value: object) -> float | None:
    if not isinstance(value, Real):
        return None
    converted = float(value)
    return converted if isfinite(converted) else None


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    converted = int(value)
    return converted if converted >= 0 else None


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    """Stable subset of solver diagnostics used by the control safety gate."""

    termination: Termination
    raw_status: str
    solver_success: bool
    constraint_violation: float | None
    iterations: int | None = None
    objective: float | None = None
    solve_time: float | None = None

    @property
    def has_finite_feasibility_measure(self) -> bool:
        return self.constraint_violation is not None


def diagnostics_from_stats(
    stats: Mapping[str, object],
    *,
    constraint_violation: float | None,
    objective: float | None = None,
) -> SolverDiagnostics:
    """Translate CasADi solver stats and a caller-computed constraint residual.

    CasADi does not expose a uniform final primal-violation field for every
    plugin/version.  The MPC layer must therefore evaluate all variable bounds
    and nonlinear constraints at the returned candidate and pass the maximum
    violation explicitly.  Missing or non-finite measurements fail closed.
    """

    raw_status = str(stats.get("return_status", ""))
    iterations = stats.get("iter_count")
    if iterations is None and isinstance(stats.get("iterations"), Mapping):
        iterations = stats["iterations"].get("iter_count")  # type: ignore[union-attr]
    solve_time = stats.get("t_wall_total", stats.get("t_proc_total"))
    violation = _optional_finite_float(constraint_violation)
    if violation is not None:
        violation = max(0.0, violation)
    return SolverDiagnostics(
        termination=normalize_termination(raw_status),
        raw_status=raw_status,
        solver_success=stats.get("success") is True,
        constraint_violation=violation,
        iterations=_optional_nonnegative_int(iterations),
        objective=_optional_finite_float(objective),
        solve_time=_optional_finite_float(solve_time),
    )


@dataclass(frozen=True, slots=True)
class FeasibilityPolicy:
    """Fail-closed policy deciding whether an MPC action may reach the robot."""

    max_constraint_violation: float = 1e-6
    accept_acceptable_level: bool = True
    accept_feasible_point: bool = False
    require_solver_success: bool = True

    def __post_init__(self) -> None:
        if (
            not isfinite(self.max_constraint_violation)
            or self.max_constraint_violation < 0.0
        ):
            raise ValueError("max_constraint_violation must be finite and non-negative")

    def accepts(self, diagnostics: SolverDiagnostics) -> bool:
        allowed = {Termination.SOLVED}
        if self.accept_acceptable_level:
            allowed.add(Termination.ACCEPTABLE)
        if self.accept_feasible_point:
            allowed.add(Termination.FEASIBLE_POINT)
        if diagnostics.termination not in allowed:
            return False
        if self.require_solver_success and not diagnostics.solver_success:
            return False
        violation = diagnostics.constraint_violation
        return violation is not None and violation <= self.max_constraint_violation


def safe_control_or_none(
    candidate: Sequence[Real] | None,
    diagnostics: SolverDiagnostics,
    *,
    policy: FeasibilityPolicy | None = None,
    expected_dimension: int = 4,
) -> tuple[float, ...] | None:
    """Return a finite feasible action, otherwise ``None`` (never stale input)."""

    active_policy = policy or FeasibilityPolicy()
    if candidate is None or expected_dimension < 1:
        return None
    if len(candidate) != expected_dimension or not active_policy.accepts(diagnostics):
        return None
    try:
        action = tuple(float(value) for value in candidate)
    except (TypeError, ValueError):
        return None
    return action if all(isfinite(value) for value in action) else None


def solve_ipopt_smoke_problem(
    config: IpoptConfig | None = None,
) -> tuple[float, SolverDiagnostics]:
    """Solve ``min (x-1)^2`` subject to ``x >= 0`` with CasADi/IPOPT.

    This is an installation smoke test, not part of the runtime MPC loop.
    ``RuntimeError`` gives one consistent failure mode when the optional stack
    or the IPOPT plugin is unavailable.
    """

    try:
        import casadi as ca
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise RuntimeError("CasADi is required for the IPOPT smoke problem") from exc

    cfg = config or IpoptConfig()
    try:
        x = ca.MX.sym("x")
        solver = ca.nlpsol(
            "ipopt_smoke",
            "ipopt",
            {"x": x, "f": (x - 1.0) ** 2, "g": x},
            cfg.casadi_options(),
        )
        result = solver(x0=-0.5, lbg=0.0, ubg=ca.inf)
    except Exception as exc:  # CasADi plugin exceptions vary by build.
        raise RuntimeError("CasADi's IPOPT plugin is unavailable or failed") from exc

    solution = float(result["x"])
    constraint_violation = max(0.0, -solution)
    diagnostics = diagnostics_from_stats(
        solver.stats(),
        constraint_violation=constraint_violation,
        objective=float(result["f"]),
    )
    return solution, diagnostics
