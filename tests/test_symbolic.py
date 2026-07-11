from __future__ import annotations

from math import pi

import pytest

from lampc_cbf import symbolic


PAPER_PHI = tuple(
    tuple(1.0 if row == column and row in (4, 5, 6) else 0.0 for column in range(8))
    for row in range(8)
)


def test_barrier_value_sign_marks_safe_boundary_and_collision() -> None:
    assert symbolic.barrier_value((2, 0, 0), (0, 0, 0), 1.0, 0.5) == pytest.approx(1.75)
    assert symbolic.barrier_value((1.5, 0, 0), (0, 0, 0), 1.0, 0.5) == pytest.approx(0.0)
    assert symbolic.barrier_value((1, 0, 0), (0, 0, 0), 1.0, 0.5) == pytest.approx(-1.25)


def test_barrier_value_supports_arbitrary_matching_dimension() -> None:
    assert symbolic.barrier_value((1, 2), (4, 6), 2, 1) == pytest.approx(16.0)


def test_barrier_value_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        symbolic.barrier_value((0, 0), (0, 0, 0), 1, 1)
    with pytest.raises(ValueError, match="non-negative"):
        symbolic.barrier_value((0,), (0,), -1, 1)


def test_discrete_cbf_value_matches_rearranged_paper_constraint() -> None:
    # At gamma=0.2 and h_k=10, the next barrier must be at least 8.
    assert symbolic.discrete_cbf_value(10, 8, 0.2) == pytest.approx(0.0)
    assert symbolic.discrete_cbf_value(10, 9, 0.2) == pytest.approx(1.0)
    assert symbolic.discrete_cbf_value(10, 7, 0.2) == pytest.approx(-1.0)


@pytest.mark.parametrize("gamma", [0, -0.1, 1.01])
def test_discrete_cbf_value_enforces_paper_gamma_domain(gamma: float) -> None:
    with pytest.raises(ValueError, match="0 < gamma <= 1"):
        symbolic.discrete_cbf_value(1, 1, gamma)


def test_regularizer_value_matches_paper_phi_and_yaw_term() -> None:
    state = (100, 100, 100, pi / 4, 1, 2, 3, 100)
    # Phi ignores position, psi, and dpsi: x.T Phi x = 1 + 4 + 9 = 14.
    assert symbolic.regularizer_value(state, PAPER_PHI, 0.1, 5e-5) == pytest.approx(
        1.4 + 5e-5
    )


@pytest.mark.parametrize("psi", [0.0, pi / 2, -pi / 2])
def test_rotation_regularizer_is_zero_at_square_grasp_alignments(psi: float) -> None:
    state = (0, 0, 0, psi, 0, 0, 0, 0)
    assert symbolic.regularizer_value(state, PAPER_PHI, 0.1, 5e-5) == pytest.approx(0.0)


def test_quadratic_form_rejects_wrong_phi_shape() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        symbolic.quadratic_form_value((1, 2), ((1, 0),))


def test_symbolic_functions_fail_clearly_when_casadi_is_missing() -> None:
    if symbolic.CASADI_AVAILABLE:
        pytest.skip("CasADi is installed")
    with pytest.raises(ImportError, match="CasADi is required"):
        symbolic.barrier_expression(object(), object(), 1, 1)


@pytest.mark.skipif(not symbolic.CASADI_AVAILABLE, reason="CasADi is optional")
def test_casadi_expressions_match_dependency_free_helpers() -> None:
    ca = symbolic.ca
    position = ca.MX.sym("position", 3)
    obstacle = ca.MX.sym("obstacle", 3)
    state = ca.MX.sym("state", 8)
    gamma = ca.MX.sym("gamma")

    h_current = symbolic.barrier_expression(position, obstacle, 0.3, 0.2)
    h_next = symbolic.barrier_expression(position + ca.DM([0.1, 0, 0]), obstacle, 0.3, 0.2)
    cbf = symbolic.discrete_cbf_expression(h_current, h_next, gamma)
    regularizer = symbolic.regularizer_expression(state, ca.DM(PAPER_PHI), 0.1, 5e-5)
    evaluate = ca.Function("evaluate", [position, obstacle, state, gamma], [h_current, cbf, regularizer])

    p = (1.0, 0.0, 0.0)
    p_obs = (0.0, 0.0, 0.0)
    x = (0, 0, 0, pi / 4, 1, 2, 3, 0)
    h, cbf_residual, reg = (float(value) for value in evaluate(p, p_obs, x, 0.1))

    expected_h = symbolic.barrier_value(p, p_obs, 0.3, 0.2)
    expected_h_next = symbolic.barrier_value((1.1, 0, 0), p_obs, 0.3, 0.2)
    assert h == pytest.approx(expected_h)
    assert cbf_residual == pytest.approx(symbolic.discrete_cbf_value(expected_h, expected_h_next, 0.1))
    assert reg == pytest.approx(symbolic.regularizer_value(x, PAPER_PHI, 0.1, 5e-5))
