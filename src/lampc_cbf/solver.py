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

    @classmethod
    def reference_defaults(cls) -> "IpoptConfig":
        """Library-default IPOPT options as used by the reference repositories.

        Verified 2026-07-17 at the pinned revisions: ``elena-ecn/mpc-cbf``
        (do-mpc, no explicit solver options) and
        ``HybridRobotics/NMPC-DCLF-DCBF`` cdc2021
        (``sdpsettings('solver','ipopt','verbose',0)``) both run IPOPT at
        library defaults. The paper publishes no options (deviation registry
        entry 1.3), so this factory is the closest sourced reconstruction:
        IPOPT defaults with output silenced and no CPU limit.
        """

        return cls(
            tolerance=1e-8,
            acceptable_tolerance=1e-6,
            constraint_violation_tolerance=1e-4,
            max_iterations=3000,
            max_cpu_time=1e6,
            mu_strategy="monotone",
            warm_start=False,
        )

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


@dataclass(frozen=True, slots=True)
class ConstraintViolation:
    """One violated nonlinear bound with its horizon-stage attribution."""

    flat_index: int
    stage: int | None
    constraint: str
    value: float
    lower: float
    upper: float
    violation: float


@dataclass(frozen=True, slots=True)
class ConstraintViolationProfile:
    """Auditable bound residuals; unsupported do-mpc layouts fail closed."""

    maximum: float | None
    violating_count: int
    first_violating_index: int | None
    maximum_index: int | None
    stage_layout_supported: bool
    violations: tuple[ConstraintViolation, ...]


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


def _numeric_vector(value: object) -> tuple[float, ...]:
    """Flatten CasADi DM/structured values without importing CasADi."""

    candidate = getattr(value, "cat", value)
    full = getattr(candidate, "full", None)
    if callable(full):
        candidate = full()
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        candidate = tolist()

    flattened: list[float] = []

    def visit(item: object) -> None:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for nested in item:
                visit(nested)
            return
        if not isinstance(item, Real):
            raise TypeError("constraint vector contains a non-numeric value")
        flattened.append(float(item))

    visit(candidate)
    return tuple(flattened)


def maximum_constraint_violation_from_mpc(mpc: object) -> float | None:
    """Measure the final do-mpc nonlinear-constraint bound violation.

    Missing, malformed, or non-finite optimizer state returns ``None`` so the
    feasibility gate fails closed instead of assuming a successful solve.
    """

    try:
        values = _numeric_vector(getattr(mpc, "opt_g_num"))
        lower_source = getattr(mpc, "lb_opt_g", None)
        if lower_source is None:
            lower_source = getattr(mpc, "nlp_cons_lb")
        upper_source = getattr(mpc, "ub_opt_g", None)
        if upper_source is None:
            upper_source = getattr(mpc, "nlp_cons_ub")
        lower = _numeric_vector(lower_source)
        upper = _numeric_vector(upper_source)
    except (AttributeError, TypeError, ValueError):
        return None
    if not values or len(values) != len(lower) or len(values) != len(upper):
        return None
    violation = 0.0
    for value, low, high in zip(values, lower, upper):
        if not isfinite(value):
            return None
        if isfinite(low):
            violation = max(violation, low - value)
        if isfinite(high):
            violation = max(violation, value - high)
    return max(0.0, violation)


def constraint_violation_profile_from_mpc(
    mpc: object, *, tolerance: float = 0.0
) -> ConstraintViolationProfile:
    """Measure and attribute this project's do-mpc constraints per horizon stage.

    The supported layout is the deterministic, discrete do-mpc formulation used
    here: initial-state equality followed by, for each stage, state continuity
    and the registered nonlinear constraints. Unknown layouts retain the flat
    residual evidence but deliberately omit stage attribution.
    """

    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    try:
        values = _numeric_vector(getattr(mpc, "opt_g_num"))
        lower_source = getattr(mpc, "lb_opt_g", None)
        if lower_source is None:
            lower_source = getattr(mpc, "nlp_cons_lb")
        upper_source = getattr(mpc, "ub_opt_g", None)
        if upper_source is None:
            upper_source = getattr(mpc, "nlp_cons_ub")
        lower = _numeric_vector(lower_source)
        upper = _numeric_vector(upper_source)
    except (AttributeError, TypeError, ValueError):
        return ConstraintViolationProfile(None, 0, None, None, False, ())
    if not values or len(values) != len(lower) or len(values) != len(upper):
        return ConstraintViolationProfile(None, 0, None, None, False, ())

    n_x = getattr(getattr(mpc, "model", None), "n_x", None)
    horizon = getattr(getattr(mpc, "settings", None), "n_horizon", None)
    nl_struct = getattr(mpc, "_nl_cons", None)
    labels_method = getattr(nl_struct, "labels", None)
    labels = tuple(labels_method()) if callable(labels_method) else ()
    n_nl = len(labels)
    supported = (
        isinstance(n_x, int)
        and isinstance(horizon, int)
        and n_x > 0
        and horizon > 0
        and n_nl > 0
        and len(values) == n_x + horizon * (n_x + n_nl)
    )
    records: list[ConstraintViolation] = []
    max_violation = 0.0
    max_index: int | None = None
    for index, (value, low, high) in enumerate(zip(values, lower, upper)):
        if not isfinite(value):
            return ConstraintViolationProfile(None, 0, None, None, False, ())
        amount = 0.0
        if isfinite(low):
            amount = max(amount, low - value)
        if isfinite(high):
            amount = max(amount, value - high)
        if amount > max_violation:
            max_violation = amount
            max_index = index
        if amount <= tolerance:
            continue
        stage: int | None = None
        name = "unattributed"
        if supported and index >= n_x:
            stage = (index - n_x) // (n_x + n_nl)
            offset = (index - n_x) % (n_x + n_nl)
            name = (
                f"state_continuity[{offset}]"
                if offset < n_x
                else labels[offset - n_x].strip("[]")
            )
        elif supported:
            name = f"initial_state[{index}]"
        records.append(
            ConstraintViolation(index, stage, name, value, low, high, amount)
        )
    return ConstraintViolationProfile(
        maximum=max(0.0, max_violation),
        violating_count=len(records),
        first_violating_index=records[0].flat_index if records else None,
        maximum_index=max_index,
        stage_layout_supported=supported,
        violations=tuple(records),
    )


def diagnostics_from_do_mpc(
    mpc: object, *, measured_solve_time: float | None = None
) -> SolverDiagnostics:
    """Build fail-closed diagnostics from one completed do-mpc solve."""

    stats = getattr(mpc, "solver_stats", {})
    if not isinstance(stats, Mapping):
        stats = {}
    diagnostics = diagnostics_from_stats(
        stats,
        constraint_violation=maximum_constraint_violation_from_mpc(mpc),
    )
    measured = _optional_finite_float(measured_solve_time)
    if measured is None:
        return diagnostics
    return SolverDiagnostics(
        termination=diagnostics.termination,
        raw_status=diagnostics.raw_status,
        solver_success=diagnostics.solver_success,
        constraint_violation=diagnostics.constraint_violation,
        iterations=diagnostics.iterations,
        objective=diagnostics.objective,
        solve_time=measured,
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
