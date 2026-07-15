"""do-mpc implementation of the controller reported in the paper.

The module deliberately has no import-time dependency on do-mpc, CasADi, or
NumPy.  This keeps configuration and model-dimension checks usable on systems
that have not installed the optional control stack yet.  Call
:func:`build_mpc_controller` to construct the symbolic model and optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, pi
from typing import Any, Callable, Mapping, Sequence


STATE_NAMES = ("x", "y", "z", "psi", "dx", "dy", "dz", "dpsi")
INPUT_NAMES = ("u_x", "u_y", "u_z", "u_psi")
DYNAMICS_MODES = ("paper_state", "double_integrator")


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
    linear_jerk_weight: float = 0.0
    yaw_jerk_weight: float = 0.0
    optimal_decay_weight: float = 0.0
    optimal_decay_nominal: float = 1.0
    optimal_decay_lower: float = 0.1
    optimal_decay_upper: float = 1.0
    target_tvp_name: str | None = None
    velocity_regularization: float = 0.1
    yaw_regularization: float = 5e-5
    position_lower: tuple[float, float, float] = (-3.0, -3.0, 0.0)
    position_upper: tuple[float, float, float] = (inf, inf, inf)
    yaw_lower: float = -0.55 * pi
    yaw_upper: float = 0.55 * pi
    linear_input_limit: float = 0.2
    yaw_input_limit: float = pi
    # ``paper_state`` retains the repository replacement-state convention.
    # ``double_integrator`` is an explicit source-validation mode.
    dynamics_mode: str = "paper_state"

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be at least one")
        if len(self.target) != 8:
            raise ValueError("target must contain exactly 8 state values")
        if len(self.position_lower) != 3 or len(self.position_upper) != 3:
            raise ValueError("position bounds must contain exactly 3 values")
        if any(lower >= upper for lower, upper in zip(self.position_lower, self.position_upper)):
            raise ValueError("position lower bounds must be smaller than upper bounds")
        if self.yaw_lower >= self.yaw_upper:
            raise ValueError("yaw_lower must be smaller than yaw_upper")
        nonnegative = (
            self.q_weight,
            self.linear_delta_u_weight,
            self.yaw_delta_u_weight,
            self.linear_jerk_weight,
            self.yaw_jerk_weight,
            self.optimal_decay_weight,
            self.velocity_regularization,
            self.yaw_regularization,
        )
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("objective weights must be non-negative")
        if self.linear_input_limit <= 0.0 or self.yaw_input_limit <= 0.0:
            raise ValueError("input limits must be positive")
        if self.dynamics_mode not in DYNAMICS_MODES:
            raise ValueError(
                "dynamics_mode must be paper_state or double_integrator"
            )
        if self.target_tvp_name == "":
            raise ValueError("target_tvp_name must be non-empty when provided")
        if not 0.0 < self.optimal_decay_lower <= self.optimal_decay_nominal <= self.optimal_decay_upper <= 1.0:
            raise ValueError(
                "optimal decay bounds must satisfy 0 < lower <= nominal <= upper <= 1"
            )

    @property
    def uses_jerk_state(self) -> bool:
        return self.linear_jerk_weight > 0.0 or self.yaw_jerk_weight > 0.0

    @property
    def uses_optimal_decay(self) -> bool:
        return self.optimal_decay_weight > 0.0

    @property
    def state_lower(self) -> tuple[float, ...]:
        return (*self.position_lower, self.yaw_lower, -inf, -inf, -inf, -inf)

    @property
    def state_upper(self) -> tuple[float, ...]:
        return (*self.position_upper, self.yaw_upper, inf, inf, inf, inf)

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

    return dynamics_matrices(dt, mode="paper_state")


def double_integrator_dynamics_matrices(
    dt: float = 0.04,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Return exact-discretized double-integrator matrices.

    The first four states are pose and the last four are velocities. Linear
    inputs are accelerations, so position receives ``0.5*dt**2*u`` and velocity
    receives ``dt*u``. This mode is opt-in because it is not the frozen
    paper-state convention used by the paper-facing profile.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    a = tuple(
        tuple(
            float(row == column)
            if row < 4 and column < 4
            else (
                dt
                if row < 4 and column == row + 4
                else float(row == column) if row >= 4 else 0.0
            )
            for column in range(8)
        )
        for row in range(8)
    )
    b = tuple(
        tuple(
            0.5 * dt**2
            if row < 4 and column == row
            else (dt if row >= 4 and column == row - 4 else 0.0)
            for column in range(4)
        )
        for row in range(8)
    )
    return a, b


def dynamics_matrices(
    dt: float = 0.04,
    *,
    mode: str = "paper_state",
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Return the canonical A/B pair for a configured transition mode."""

    if mode == "paper_state":
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
    if mode == "double_integrator":
        return double_integrator_dynamics_matrices(dt)
    raise ValueError("mode must be paper_state or double_integrator")


def discrete_state_transition(
    state: Sequence[float],
    control: Sequence[float],
    *,
    dt: float = 0.04,
    mode: str = "paper_state",
) -> tuple[float, ...]:
    """Advance one state using the same A/B pair as the do-mpc model."""

    if len(state) != 8 or len(control) != 4:
        raise ValueError("state and control must contain 8 and 4 values")
    a, b = dynamics_matrices(dt, mode=mode)
    return tuple(
        sum(a[row][column] * float(state[column]) for column in range(8))
        + sum(b[row][column] * float(control[column]) for column in range(4))
        for row in range(8)
    )


ConstraintBuilder = Callable[[Any, Any, Any, Any, Any], None]
"""Hook signature: ``builder(model, mpc, x, u, casadi)``.

The hook is called after the do-mpc model and MPC object exist but before
``mpc.setup()``.  It is intended for the CasADi specialist's CBF constraints.
"""

ModelBuilder = Callable[[Any, Any, Any, Any], None]
"""Hook signature: ``builder(model, x, u, casadi)``.

The hook runs before ``model.setup()`` so callers can declare time-varying
parameters (TVPs) required by online CBF constraints.
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
    model_builders: Sequence[ModelBuilder] = (),
    constraint_builders: Sequence[ConstraintBuilder] = (),
    suppress_solver_output: bool = True,
    nlpsol_options: Mapping[str, Any] | None = None,
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
    state_dimension = 12 if cfg.uses_jerk_state else 8
    x = model.set_variable(
        var_type="_x", var_name="x", shape=(state_dimension, 1)
    )
    u = model.set_variable(var_type="_u", var_name="u", shape=(4, 1))
    decay = (
        model.set_variable(var_type="_u", var_name="cbf_decay", shape=(1, 1))
        if cfg.uses_optimal_decay
        else None
    )
    a_values, b_values = dynamics_matrices(cfg.dt, mode=cfg.dynamics_mode)
    a = ca.DM(a_values)
    b = ca.DM(b_values)
    paper_state_rhs = ca.mtimes(a, x[:8]) + ca.mtimes(b, u)
    if cfg.uses_jerk_state:
        # The four augmented states retain the previous input increment.
        # This makes Δ²u available symbolically at every prediction stage.
        model.set_rhs("x", ca.vertcat(paper_state_rhs, u - x[4:8]))
    else:
        model.set_rhs("x", paper_state_rhs)
    for builder in model_builders:
        builder(model, x, u, ca)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    nlpsol_opts: dict[str, Any] = {}
    if suppress_solver_output:
        nlpsol_opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
        }
    if nlpsol_options is not None:
        nlpsol_opts.update(dict(nlpsol_options))
    mpc.set_param(
        n_horizon=cfg.horizon,
        t_step=cfg.dt,
        n_robust=0,
        store_full_solution=True,
        nlpsol_opts=nlpsol_opts,
    )

    target = (
        model.tvp[cfg.target_tvp_name]
        if cfg.target_tvp_name is not None
        else ca.DM(cfg.target)
    )
    error = x[:8] - target
    velocity_regularizer = cfg.velocity_regularization * ca.dot(x[4:7], x[4:7])
    yaw_regularizer = cfg.yaw_regularization * ca.sin(2.0 * x[3]) ** 2
    state_cost = cfg.q_weight * ca.dot(error, error)
    terminal_cost = state_cost + velocity_regularizer + yaw_regularizer
    stage_cost = terminal_cost
    if cfg.uses_jerk_state:
        input_increment = u - x[4:8]
        jerk = input_increment - x[8:12]
        stage_cost += (
            cfg.linear_jerk_weight * ca.dot(jerk[:3], jerk[:3])
            + cfg.yaw_jerk_weight * jerk[3] ** 2
        )
    if decay is not None:
        stage_cost += cfg.optimal_decay_weight * (
            decay - cfg.optimal_decay_nominal
        ) ** 2
    mpc.set_objective(mterm=terminal_cost, lterm=stage_cost)
    delta_u = u - mpc.u_prev["u"]
    delta_u_cost = (
        cfg.linear_delta_u_weight * ca.dot(delta_u[:3], delta_u[:3])
        + cfg.yaw_delta_u_weight * delta_u[3] ** 2
    )
    # A symbolic expression permits distinct weights within one vector input.
    mpc.set_rterm(delta_u_cost)

    state_lower = cfg.state_lower + ((-inf,) * 4 if cfg.uses_jerk_state else ())
    state_upper = cfg.state_upper + ((inf,) * 4 if cfg.uses_jerk_state else ())
    mpc.bounds["lower", "_x", "x"] = ca.DM(state_lower)
    mpc.bounds["upper", "_x", "x"] = ca.DM(state_upper)
    mpc.bounds["lower", "_u", "u"] = ca.DM(cfg.input_lower)
    mpc.bounds["upper", "_u", "u"] = ca.DM(cfg.input_upper)
    if decay is not None:
        mpc.bounds["lower", "_u", "cbf_decay"] = cfg.optimal_decay_lower
        mpc.bounds["upper", "_u", "cbf_decay"] = cfg.optimal_decay_upper

    for builder in constraint_builders:
        builder(model, mpc, x, u, ca)

    mpc.setup()
    return model, mpc
