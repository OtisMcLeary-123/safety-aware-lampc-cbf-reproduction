"""do-mpc implementation of the controller reported in the paper.

The module deliberately has no import-time dependency on do-mpc, CasADi, or
NumPy.  This keeps configuration and model-dimension checks usable on systems
that have not installed the optional control stack yet.  Call
:func:`build_mpc_controller` to construct the symbolic model and optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, pi
from typing import Any, Callable, Sequence


STATE_NAMES = ("x", "y", "z", "psi", "dx", "dy", "dz", "dpsi")
INPUT_NAMES = ("u_x", "u_y", "u_z", "u_psi")


@dataclass(frozen=True, slots=True)
class PaperMPCConfig:
    """Numerical controller settings stated in Section V-A.2 of the paper.

    ``target`` contains position, yaw, and the four displacement states in the
    same order as :data:`STATE_NAMES`.  The paper does not publish a single
    target because it changes between task-planner subtasks, so zero is the
    neutral default and callers should normally provide the active subtask
    target.
    """

    dt: float = 0.04
    horizon: int = 15
    target: tuple[float, ...] = (0.0,) * 8
    q_weight: float = 1.0
    linear_delta_u_weight: float = 0.5
    yaw_delta_u_weight: float = 1e-5
    velocity_regularization: float = 0.1
    yaw_regularization: float = 5e-5
    position_lower: tuple[float, float, float] = (-3.0, -3.0, 0.0)
    yaw_lower: float = -0.55 * pi
    yaw_upper: float = 0.55 * pi
    linear_input_limit: float = 0.2
    yaw_input_limit: float = pi

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be at least one")
        if len(self.target) != 8:
            raise ValueError("target must contain exactly 8 state values")
        if self.yaw_lower >= self.yaw_upper:
            raise ValueError("yaw_lower must be smaller than yaw_upper")
        nonnegative = (
            self.q_weight,
            self.linear_delta_u_weight,
            self.yaw_delta_u_weight,
            self.velocity_regularization,
            self.yaw_regularization,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("objective weights must be non-negative")
        if self.linear_input_limit <= 0.0 or self.yaw_input_limit <= 0.0:
            raise ValueError("input limits must be positive")

    @property
    def state_lower(self) -> tuple[float, ...]:
        return (*self.position_lower, self.yaw_lower, -inf, -inf, -inf, -inf)

    @property
    def state_upper(self) -> tuple[float, ...]:
        return (inf, inf, inf, self.yaw_upper, inf, inf, inf, inf)

    @property
    def input_lower(self) -> tuple[float, ...]:
        return (-self.linear_input_limit,) * 3 + (-self.yaw_input_limit,)

    @property
    def input_upper(self) -> tuple[float, ...]:
        return (self.linear_input_limit,) * 3 + (self.yaw_input_limit,)

    @property
    def delta_u_weights(self) -> tuple[float, ...]:
        return (self.linear_delta_u_weight,) * 3 + (self.yaw_delta_u_weight,)


def paper_dynamics_matrices(
    dt: float = 0.04,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Return the paper's 8-by-8 ``A`` and 8-by-4 ``B`` as plain tuples."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    a = tuple(
        tuple(
            float(row == column)
            if row < 4 and column < 4
            else (dt if row < 4 and column == row + 4 else 0.0)
            for column in range(8)
        )
        for row in range(8)
    )
    b = tuple(
        tuple(float(row >= 4 and column == row - 4) for column in range(4))
        for row in range(8)
    )
    return a, b


ConstraintBuilder = Callable[[Any, Any, Any, Any, Any], None]
"""Hook signature: ``builder(model, mpc, x, u, casadi)``.

The hook is called after the do-mpc model and MPC object exist but before
``mpc.setup()``.  It is intended for the CasADi specialist's CBF constraints.
"""


def _load_control_stack() -> tuple[Any, Any]:
    try:
        import casadi as ca
        import do_mpc
    except ImportError as exc:
        raise RuntimeError(
            "Building the controller requires the optional packages 'do-mpc' "
            "and 'casadi' (with an IPOPT-enabled CasADi build). Install the "
            "project's control extras before calling build_mpc_controller()."
        ) from exc
    return do_mpc, ca


def build_mpc_controller(
    config: PaperMPCConfig | None = None,
    *,
    constraint_builders: Sequence[ConstraintBuilder] = (),
    suppress_solver_output: bool = True,
) -> tuple[Any, Any]:
    """Build and set up the paper's discrete nonlinear MPC controller.

    Returns ``(model, mpc)``.  The caller completes the standard do-mpc
    receding-horizon loop by calling ``mpc.x0 = ...``, ``mpc.set_initial_guess``
    and then ``mpc.make_step(measured_state)`` once per control cycle.

    The objective matches equations (19) and the preceding terminal/stage cost
    definitions.  do-mpc's ``set_rterm`` supplies the stage penalty on
    ``u[i] - u[i-1]`` and automatically carries the previously applied input
    into the first stage of each receding-horizon solve.
    """

    cfg = config or PaperMPCConfig()
    do_mpc, ca = _load_control_stack()

    model = do_mpc.model.Model("discrete")
    x = model.set_variable(var_type="_x", var_name="x", shape=(8, 1))
    u = model.set_variable(var_type="_u", var_name="u", shape=(4, 1))
    a_values, b_values = paper_dynamics_matrices(cfg.dt)
    a = ca.DM(a_values)
    b = ca.DM(b_values)
    model.set_rhs("x", ca.mtimes(a, x) + ca.mtimes(b, u))
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    nlpsol_opts: dict[str, Any] = {}
    if suppress_solver_output:
        nlpsol_opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
        }
    mpc.set_param(
        n_horizon=cfg.horizon,
        t_step=cfg.dt,
        n_robust=0,
        store_full_solution=True,
        nlpsol_opts=nlpsol_opts,
    )

    target = ca.DM(cfg.target)
    error = x - target
    velocity_regularizer = cfg.velocity_regularization * ca.dot(x[4:7], x[4:7])
    yaw_regularizer = cfg.yaw_regularization * ca.sin(2.0 * x[3]) ** 2
    state_cost = cfg.q_weight * ca.dot(error, error)
    terminal_cost = state_cost + velocity_regularizer + yaw_regularizer
    mpc.set_objective(mterm=terminal_cost, lterm=terminal_cost)
    delta_u = u - mpc.u_prev["u"]
    delta_u_cost = (
        cfg.linear_delta_u_weight * ca.dot(delta_u[:3], delta_u[:3])
        + cfg.yaw_delta_u_weight * delta_u[3] ** 2
    )
    # A symbolic expression permits distinct weights within one vector input.
    mpc.set_rterm(delta_u_cost)

    mpc.bounds["lower", "_x", "x"] = ca.DM(cfg.state_lower)
    mpc.bounds["upper", "_x", "x"] = ca.DM(cfg.state_upper)
    mpc.bounds["lower", "_u", "u"] = ca.DM(cfg.input_lower)
    mpc.bounds["upper", "_u", "u"] = ca.DM(cfg.input_upper)

    for builder in constraint_builders:
        builder(model, mpc, x, u, ca)

    mpc.setup()
    return model, mpc
