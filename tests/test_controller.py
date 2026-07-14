from __future__ import annotations

import importlib.util
import math

import pytest

from lampc_cbf.controller import (
    INPUT_NAMES,
    STATE_NAMES,
    PaperMPCConfig,
    build_mpc_controller,
    paper_dynamics_matrices,
)


def test_paper_dimensions_and_default_parameters() -> None:
    config = PaperMPCConfig()

    assert len(STATE_NAMES) == 8
    assert len(INPUT_NAMES) == 4
    assert config.dt == pytest.approx(0.04)
    assert config.horizon == 15
    assert not config.uses_jerk_state
    assert not config.uses_optimal_decay
    assert config.delta_u_weights == pytest.approx((0.5, 0.5, 0.5, 1e-5))
    assert config.input_lower == pytest.approx((-0.2, -0.2, -0.2, -math.pi))
    assert config.input_upper == pytest.approx((0.2, 0.2, 0.2, math.pi))
    assert config.state_lower[:4] == pytest.approx(
        (-3.0, -3.0, 0.0, -0.55 * math.pi)
    )
    assert config.state_upper[3] == pytest.approx(0.55 * math.pi)


def test_paper_dynamics_update_pose_and_replace_displacement() -> None:
    a, b = paper_dynamics_matrices(0.04)
    state = (1.0, 2.0, 3.0, 0.2, 0.1, -0.2, 0.3, -0.4)
    control = (0.5, 0.6, -0.7, 0.8)

    next_state = tuple(
        sum(a[row][column] * state[column] for column in range(8))
        + sum(b[row][column] * control[column] for column in range(4))
        for row in range(8)
    )

    assert next_state[:4] == pytest.approx((1.004, 1.992, 3.012, 0.184))
    assert next_state[4:] == pytest.approx(control)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"dt": 0.0}, "dt must be positive"),
        ({"horizon": 0}, "horizon must be at least one"),
        ({"target": (0.0,) * 7}, "exactly 8"),
        ({"velocity_regularization": -1.0}, "non-negative"),
        ({"optimal_decay_lower": 0.0}, "optimal decay bounds"),
        ({"optimal_decay_upper": 1.1}, "optimal decay bounds"),
    ],
)
def test_invalid_configuration_is_rejected(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PaperMPCConfig(**kwargs)  # type: ignore[arg-type]


def test_missing_optional_stack_has_actionable_error() -> None:
    dependencies_available = all(
        importlib.util.find_spec(package) is not None for package in ("do_mpc", "casadi")
    )
    if dependencies_available:
        pytest.skip("optional control dependencies are installed")

    with pytest.raises(RuntimeError, match="optional packages 'do-mpc' and 'casadi'"):
        build_mpc_controller()


@pytest.mark.skipif(
    importlib.util.find_spec("do_mpc") is None
    or importlib.util.find_spec("casadi") is None,
    reason="do-mpc/CasADi control extras are not installed",
)
def test_controller_builds_with_optional_stack() -> None:
    model, mpc = build_mpc_controller()

    assert model.x["x"].shape == (8, 1)
    assert model.u["u"].shape == (4, 1)
    assert mpc.settings.n_horizon == 15
    assert mpc.settings.t_step == pytest.approx(0.04)


@pytest.mark.skipif(
    importlib.util.find_spec("do_mpc") is None
    or importlib.util.find_spec("casadi") is None,
    reason="do-mpc/CasADi control extras are not installed",
)
def test_controller_model_hook_can_declare_tvp() -> None:
    def declare(model, x, u, ca):
        del x, u, ca
        model.set_variable("_tvp", "moving_center", shape=(3, 1))

    def configure(model, mpc, x, u, ca):
        del model, x, u, ca
        template = mpc.get_tvp_template()
        mpc.set_tvp_fun(lambda _: template)

    model, _ = build_mpc_controller(
        model_builders=(declare,), constraint_builders=(configure,)
    )

    assert model.tvp["moving_center"].shape == (3, 1)


@pytest.mark.skipif(
    importlib.util.find_spec("do_mpc") is None
    or importlib.util.find_spec("casadi") is None,
    reason="do-mpc/CasADi control extras are not installed",
)
def test_controller_builds_augmented_jerk_state() -> None:
    model, _ = build_mpc_controller(PaperMPCConfig(linear_jerk_weight=0.1))

    assert model.x["x"].shape == (12, 1)


@pytest.mark.skipif(
    importlib.util.find_spec("do_mpc") is None
    or importlib.util.find_spec("casadi") is None,
    reason="do-mpc/CasADi control extras are not installed",
)
def test_controller_builds_bounded_optimal_decay_input() -> None:
    model, mpc = build_mpc_controller(
        PaperMPCConfig(optimal_decay_weight=10.0, optimal_decay_lower=0.2)
    )

    assert model.u["u"].shape == (4, 1)
    assert model.u["cbf_decay"].shape == (1, 1)
    assert float(mpc.bounds["lower", "_u", "cbf_decay"]) == pytest.approx(0.2)
    assert float(mpc.bounds["upper", "_u", "cbf_decay"]) == pytest.approx(1.0)


@pytest.mark.skipif(
    importlib.util.find_spec("do_mpc") is None
    or importlib.util.find_spec("casadi") is None,
    reason="do-mpc/CasADi control extras are not installed",
)
def test_controller_accepts_explicit_ipopt_options() -> None:
    _, mpc = build_mpc_controller(
        PaperMPCConfig(horizon=2),
        nlpsol_options={"ipopt.max_cpu_time": 0.035, "ipopt.max_iter": 17},
    )

    assert mpc.settings.nlpsol_opts["ipopt.max_cpu_time"] == pytest.approx(0.035)
    assert mpc.settings.nlpsol_opts["ipopt.max_iter"] == 17
