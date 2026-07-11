from __future__ import annotations

import importlib.util
from math import nan

import pytest

from lampc_cbf.solver import (
    FeasibilityPolicy,
    IpoptConfig,
    SolverDiagnostics,
    Termination,
    diagnostics_from_stats,
    normalize_termination,
    safe_control_or_none,
    solve_ipopt_smoke_problem,
)


def diagnostics(
    termination: Termination = Termination.SOLVED,
    *,
    success: bool = True,
    violation: float | None = 0.0,
) -> SolverDiagnostics:
    return SolverDiagnostics(termination, termination.value, success, violation)


def test_ipopt_options_are_casadi_and_do_mpc_compatible() -> None:
    options = IpoptConfig(max_iterations=75, warm_start=True).casadi_options()

    assert options["ipopt.max_iter"] == 75
    assert options["ipopt.tol"] == pytest.approx(1e-8)
    assert options["ipopt.constr_viol_tol"] == pytest.approx(1e-6)
    assert options["ipopt.mu_strategy"] == "adaptive"
    assert options["ipopt.warm_start_init_point"] == "yes"
    assert options["ipopt.print_level"] == 0
    assert options["print_time"] is False


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"tolerance": 0.0}, "tolerance must be"),
        ({"max_iterations": 0}, "at least one"),
        ({"max_cpu_time": nan}, "max_cpu_time must be"),
        ({"print_level": 13}, "between 0 and 12"),
        ({"mu_strategy": "invalid"}, "adaptive.*monotone"),
    ],
)
def test_invalid_ipopt_options_are_rejected(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        IpoptConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Solve_Succeeded", Termination.SOLVED),
        ("Solved_To_Acceptable_Level", Termination.ACCEPTABLE),
        ("Infeasible_Problem_Detected", Termination.INFEASIBLE),
        ("Maximum_Iterations_Exceeded", Termination.MAX_ITERATIONS),
        ("Restoration_Failed", Termination.RESTORATION_FAILED),
        ("Invalid_Number_Detected", Termination.NUMERICAL_ERROR),
        ("new future status", Termination.UNKNOWN),
        (None, Termination.UNKNOWN),
    ],
)
def test_ipopt_termination_is_normalized(raw: object, expected: Termination) -> None:
    assert normalize_termination(raw) is expected


def test_diagnostics_preserve_only_finite_reliable_measurements() -> None:
    result = diagnostics_from_stats(
        {
            "return_status": "Solve_Succeeded",
            "success": True,
            "iter_count": 12,
            "t_wall_total": 0.004,
        },
        constraint_violation=-1e-12,
        objective=2.5,
    )

    assert result.termination is Termination.SOLVED
    assert result.solver_success
    assert result.constraint_violation == 0.0
    assert result.iterations == 12
    assert result.objective == pytest.approx(2.5)
    assert result.solve_time == pytest.approx(0.004)

    invalid = diagnostics_from_stats({}, constraint_violation=nan, objective=nan)
    assert invalid.termination is Termination.UNKNOWN
    assert invalid.constraint_violation is None
    assert invalid.objective is None


def test_safety_gate_returns_control_only_for_verified_solution() -> None:
    candidate = (0.1, -0.1, 0.0, 0.2)

    assert safe_control_or_none(candidate, diagnostics()) == pytest.approx(candidate)
    assert safe_control_or_none(candidate, diagnostics(violation=1.1e-6)) is None
    assert safe_control_or_none(candidate, diagnostics(violation=None)) is None
    assert safe_control_or_none(candidate, diagnostics(success=False)) is None
    assert safe_control_or_none(
        candidate, diagnostics(Termination.INFEASIBLE)
    ) is None
    assert safe_control_or_none((0.1, 0.2), diagnostics()) is None
    assert safe_control_or_none((0.1, nan, 0.0, 0.0), diagnostics()) is None


def test_acceptable_termination_can_be_disabled_explicitly() -> None:
    acceptable = diagnostics(Termination.ACCEPTABLE, violation=1e-8)

    assert safe_control_or_none((0, 0, 0, 0), acceptable) == (0.0,) * 4
    assert (
        safe_control_or_none(
            (0, 0, 0, 0),
            acceptable,
            policy=FeasibilityPolicy(accept_acceptable_level=False),
        )
        is None
    )


@pytest.mark.skipif(
    importlib.util.find_spec("casadi") is None,
    reason="CasADi/IPOPT control extra is not installed",
)
def test_optional_ipopt_smoke_problem() -> None:
    try:
        solution, result = solve_ipopt_smoke_problem()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    assert solution == pytest.approx(1.0, abs=1e-6)
    assert FeasibilityPolicy().accepts(result)
