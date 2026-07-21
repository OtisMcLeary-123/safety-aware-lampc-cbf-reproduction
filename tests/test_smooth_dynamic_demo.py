from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from lampc_cbf.smooth_dynamic_demo import (
    ReferenceObstacleStage,
    ReferenceObstacleTVP,
    SmoothDynamicConfig,
    braking_fallback_command,
    build_reference_route,
    classify_episode_outcome,
)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"collision": True}, "collision"),
        ({"reached_goal": True}, "goal"),
        ({"truncated": True}, "environment_truncated"),
        ({"solver_rejections": 100}, "solver_failure"),
        ({"final_goal_distance": 0.295}, "controller_stall"),
        ({"final_goal_distance": 0.20}, "safety_timeout"),
    ],
)
def test_episode_outcome_classifies_timeout_cause(kwargs, expected) -> None:
    inputs = {
        "reached_goal": False,
        "collision": False,
        "truncated": False,
        "initial_goal_distance": 0.30,
        "final_goal_distance": 0.20,
        "solver_rejections": 0,
        "steps": 100,
        "stall_progress_threshold": 0.01,
    }
    inputs.update(kwargs)
    assert classify_episode_outcome(**inputs) == expected


def test_smooth_dynamic_configuration_validates_weights() -> None:
    assert SmoothDynamicConfig(delta_u_weight=2.0, jerk_weight=0.1).jerk_weight == 0.1
    with pytest.raises(ValueError, match="non-negative"):
        SmoothDynamicConfig(jerk_weight=-0.1)


def test_dynamic_configuration_accepts_velocity_only_and_direct_target():
    config = SmoothDynamicConfig(
        prediction_mode="velocity", reference_mode="direct_target"
    )
    assert config.prediction_mode == "velocity"
    assert config.reference_mode == "direct_target"


def test_3d_reference_route_has_lateral_and_vertical_waypoints() -> None:
    route, path = build_reference_route(
        (0.0, 0.0, 0.2),
        (0.0, 0.3, 0.2),
        obstacle_velocity=(0.05, 0.0, -0.01),
        combined_radius=0.035,
        route_margin=0.08,
        profile="3d_waypoints",
        waypoint_offsets=((0.14, 0.08, 0.10), (0.14, 0.23, 0.10)),
    )
    assert route.shape[1] == 3
    assert path.shape[1] == 3
    assert np.max(np.abs(route[:, 0])) > 0.0
    assert np.max(np.abs(route[:, 2] - 0.2)) > 0.0


def test_3d_route_profile_requires_two_waypoints() -> None:
    with pytest.raises(ValueError, match="at least two"):
        SmoothDynamicConfig(
            reference_route_profile="3d_waypoints",
            avoidance_waypoint_offsets=((0.1, 0.1, 0.1),),
        )


def test_dynamic_configuration_accepts_experimental_dpcbf_reflex() -> None:
    config = SmoothDynamicConfig(
        reflex_barrier_mode="dynamic_parabolic",
        reflex_side_latch_enabled=True,
        reflex_policy_library_enabled=True,
    )
    assert config.reflex_barrier_mode == "dynamic_parabolic"


def test_dynamic_configuration_validates_robot_velocity_estimator() -> None:
    assert SmoothDynamicConfig(robot_velocity_filter=1.0).robot_velocity_filter == 1.0
    with pytest.raises(ValueError, match="robot_velocity_filter"):
        SmoothDynamicConfig(robot_velocity_filter=1.1)
    with pytest.raises(ValueError, match="robot_velocity_maximum"):
        SmoothDynamicConfig(robot_velocity_maximum=0.0)
    with pytest.raises(ValueError, match="stall_progress_threshold"):
        SmoothDynamicConfig(stall_progress_threshold=-0.01)


def test_simulation_defaults_to_command_velocity_cbf_transition() -> None:
    assert SmoothDynamicConfig().cbf_transition_mode == "command_velocity"
    with pytest.raises(ValueError, match="cbf_transition_mode"):
        SmoothDynamicConfig(cbf_transition_mode="unknown")


def test_double_integrator_mode_requires_velocity_compatible_safety_stack() -> None:
    config = SmoothDynamicConfig(
        cbf_transition_mode="double_integrator",
        safety_reflex_enabled=False,
    )
    assert config.cbf_transition_mode == "double_integrator"
    with pytest.raises(ValueError, match="reflex and formal filters disabled"):
        SmoothDynamicConfig(cbf_transition_mode="double_integrator")


def test_formal_profile_requires_bounded_measurement_noise() -> None:
    with pytest.raises(ValueError, match="deterministically bounded"):
        SmoothDynamicConfig(formal_safety_filter_enabled=True)
    config = SmoothDynamicConfig(
        formal_safety_filter_enabled=True,
        measurement_noise_mode="bounded_ball",
    )
    assert config.measurement_error_bound == pytest.approx(0.005)


def test_delayed_schedule_tracks_request_time_for_ttl() -> None:
    config = SmoothDynamicConfig(
        gamma_schedule=((1.2, 0.02),),
        gamma_schedule_request_times=(0.2,),
    )
    assert config.gamma_schedule_request_times == (0.2,)
    with pytest.raises(ValueError, match="match schedule"):
        SmoothDynamicConfig(
            gamma_schedule=((1.2, 0.02),),
            gamma_schedule_request_times=(0.1, 0.2),
        )


def test_reference_progress_is_monotone() -> None:
    path = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.zeros(20), np.ones(20)]
    )
    provider = ReferenceObstacleTVP(
        path, (0.0, 2.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15,
    )

    provider.update((0.6, 0.0, 1.0), (0.0, 2.0, 1.0))
    forward_index = provider.progress_index
    provider.update((0.2, 0.0, 1.0), (0.0, 2.0, 1.0))

    assert forward_index > 0
    assert provider.progress_index == forward_index


def test_prediction_mode_schedule_validation() -> None:
    from lampc_cbf.smooth_dynamic_demo import SmoothDynamicConfig

    config = SmoothDynamicConfig(
        prediction_mode_schedule=((1.0, "velocity_tube"), (2.0, "static"))
    )
    assert config.prediction_mode_schedule[0] == (1.0, "velocity_tube")
    with pytest.raises(ValueError, match="sorted"):
        SmoothDynamicConfig(
            prediction_mode_schedule=((2.0, "static"), (1.0, "velocity_tube"))
        )
    with pytest.raises(ValueError, match="valid mode"):
        SmoothDynamicConfig(prediction_mode_schedule=((1.0, "bogus"),))
    with pytest.raises(ValueError, match="request times"):
        SmoothDynamicConfig(
            prediction_mode_schedule=((1.0, "velocity_tube"),),
            prediction_mode_schedule_request_times=(1.0, 2.0),
        )


def test_reference_provider_hot_swaps_prediction_mode() -> None:
    import numpy as np

    from lampc_cbf.smooth_dynamic_demo import ReferenceObstacleTVP

    path = np.linspace([0.0, 0.0, 0.0], [0.0, 0.3, 0.0], 50)
    provider = ReferenceObstacleTVP(
        path, (0.0, 2.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15, prediction_mode="static",
    )
    assert provider.prediction_mode == "static"
    provider.update_prediction_mode("velocity_tube")
    assert provider.prediction_mode == "velocity_tube"
    with pytest.raises(ValueError, match="static, velocity, or velocity_tube"):
        provider.update_prediction_mode("bogus")


def test_reference_provider_hot_swaps_valid_gamma() -> None:
    path = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.zeros(20), np.ones(20)]
    )
    provider = ReferenceObstacleTVP(
        path, (0.0, 2.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15,
    )

    provider.update_gamma(0.03)

    assert provider.gamma == pytest.approx(0.03)
    with pytest.raises(ValueError, match="configured gamma range"):
        provider.update_gamma(0.2)


def test_reference_provider_temporarily_tracks_liveness_subgoal() -> None:
    path = np.column_stack(
        [np.zeros(20), np.linspace(0.0, 1.0, 20), np.ones(20)]
    )
    provider = ReferenceObstacleTVP(
        path, (2.0, 2.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15,
    )
    provider.update((0.0, 0.0, 1.0), (2.0, 2.0, 1.0))
    provider.set_temporary_subgoal((0.1, 0.0, 1.0))

    diverted = provider.prediction_at_stage(1)
    provider.set_temporary_subgoal(None)
    nominal = provider.prediction_at_stage(1)

    assert diverted.reference_state[0] > 0.0
    assert diverted.reference_state[4] > 0.0
    assert nominal.reference_state[0] == pytest.approx(0.0)
    assert nominal.reference_state[5] > 0.0


def test_reference_provider_preserves_paper_transition_default() -> None:
    path = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.zeros(20), np.ones(20)]
    )
    provider = ReferenceObstacleTVP(
        path, (0.0, 2.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15,
    )

    assert provider.cbf_transition_mode == "paper_state"


def test_dynamic_configuration_accepts_paper_increment_mode() -> None:
    config = SmoothDynamicConfig(
        cbf_transition_mode="paper_increment",
        safety_reflex_enabled=False,
        formal_safety_filter_enabled=False,
    )
    assert config.cbf_transition_mode == "paper_increment"
    with pytest.raises(ValueError, match="reflex"):
        SmoothDynamicConfig(cbf_transition_mode="paper_increment")


def test_dynamic_configuration_validates_cbf_constraint_scope() -> None:
    assert SmoothDynamicConfig().cbf_constraint_scope == "horizon"
    assert (
        SmoothDynamicConfig(cbf_constraint_scope="first_step").cbf_constraint_scope
        == "first_step"
    )
    with pytest.raises(ValueError, match="cbf_constraint_scope"):
        SmoothDynamicConfig(cbf_constraint_scope="last_step")


def test_reference_provider_defaults_to_full_horizon_cbf_scope() -> None:
    path = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.zeros(20), np.ones(20)]
    )
    provider = ReferenceObstacleTVP(
        path, (0.0, 2.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15,
    )

    assert provider.cbf_constraint_scope == "horizon"
    assert provider._cbf_active_tvp is None
    assert all(
        provider.cbf_active_at_stage(stage) == 1.0 for stage in range(16)
    )


def test_one_step_scope_gates_cbf_to_stage_zero() -> None:
    path = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.zeros(20), np.ones(20)]
    )
    provider = ReferenceObstacleTVP(
        path, (0.0, 2.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15,
        cbf_constraint_scope="first_step",
    )

    assert provider.cbf_active_at_stage(0) == 1.0
    assert all(
        provider.cbf_active_at_stage(stage) == 0.0 for stage in range(1, 16)
    )
    with pytest.raises(ValueError, match="non-negative"):
        provider.cbf_active_at_stage(-1)
    with pytest.raises(ValueError, match="cbf_constraint_scope"):
        ReferenceObstacleTVP(
            path, (0.0, 2.0, 1.0), reference_speed=0.1,
            obstacle_radius=0.1, collision_radius=0.035,
            gamma=0.1, dt=0.04, horizon=15,
            cbf_constraint_scope="everywhere",
        )


def test_terminal_safe_set_gates_only_the_final_stage() -> None:
    path = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.zeros(20), np.ones(20)]
    )
    provider = ReferenceObstacleTVP(
        path, (0.0, 2.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15,
        terminal_safe_set=True,
    )

    assert provider.terminal_safe_set
    assert provider.terminal_safe_active_at_stage(14) == 1.0
    assert all(
        provider.terminal_safe_active_at_stage(stage) == 0.0
        for stage in range(16)
        if stage != 14
    )
    default_provider = ReferenceObstacleTVP(
        path, (0.0, 2.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15,
    )
    assert not default_provider.terminal_safe_set
    assert default_provider._terminal_safe_active_tvp is None
    assert SmoothDynamicConfig().terminal_safe_set_enabled is False


def test_dynamic_configuration_validates_solver_fallback_mode() -> None:
    assert SmoothDynamicConfig().solver_fallback_mode == "zero"
    assert (
        SmoothDynamicConfig(solver_fallback_mode="brake").solver_fallback_mode
        == "brake"
    )
    with pytest.raises(ValueError, match="solver_fallback_mode"):
        SmoothDynamicConfig(solver_fallback_mode="coast")
    with pytest.raises(ValueError, match="fallback_braking_decay"):
        SmoothDynamicConfig(fallback_braking_decay=1.0)


def _far_obstacle_stage() -> ReferenceObstacleStage:
    return ReferenceObstacleStage(
        reference_state=(0.0,) * 8,
        obstacle_position=(5.0, 5.0, 5.0),
        obstacle_next_position=(5.0, 5.0, 5.0),
        robust_radius=0.135,
        robust_radius_next=0.135,
    )


def test_braking_fallback_decays_velocity_when_safe() -> None:
    command, engaged = braking_fallback_command(
        (0.0, 0.0, 0.2),
        (0.10, -0.04, 0.0),
        _far_obstacle_stage(),
        gamma=0.15,
        dt=0.04,
        braking_decay=0.5,
        cbf_transition_mode="command_velocity",
        input_limit=0.2,
    )

    assert engaged
    assert command == pytest.approx((0.05, -0.02, 0.0, 0.0))


def test_braking_fallback_fails_closed_when_cbf_screen_rejects() -> None:
    # The held obstacle sits directly ahead inside the braked one-step path,
    # so the residual is negative and the command must collapse to zero.
    blocking_stage = ReferenceObstacleStage(
        reference_state=(0.0,) * 8,
        obstacle_position=(0.14, 0.0, 0.2),
        obstacle_next_position=(0.10, 0.0, 0.2),
        robust_radius=0.135,
        robust_radius_next=0.135,
    )

    command, engaged = braking_fallback_command(
        (0.0, 0.0, 0.2),
        (0.2, 0.0, 0.0),
        blocking_stage,
        gamma=0.15,
        dt=0.04,
        braking_decay=0.5,
        cbf_transition_mode="command_velocity",
        input_limit=0.2,
    )

    assert not engaged
    assert command == (0.0, 0.0, 0.0, 0.0)


def test_braking_fallback_double_integrator_respects_input_limit() -> None:
    command, engaged = braking_fallback_command(
        (0.0, 0.0, 0.2),
        (0.4, 0.0, 0.0),
        _far_obstacle_stage(),
        gamma=0.15,
        dt=0.04,
        braking_decay=0.0,
        cbf_transition_mode="double_integrator",
        input_limit=0.2,
    )

    assert engaged
    # Full stop would need -10 m/s^2; the command saturates at the limit.
    assert command[0] == pytest.approx(-0.2)
    with pytest.raises(ValueError, match="braking_decay"):
        braking_fallback_command(
            (0.0, 0.0, 0.2),
            (0.1, 0.0, 0.0),
            _far_obstacle_stage(),
            gamma=0.15,
            dt=0.04,
            braking_decay=1.0,
            cbf_transition_mode="command_velocity",
            input_limit=0.2,
        )


def test_obstacle_tvp_current_next_indexing_is_contiguous() -> None:
    path = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.zeros(20), np.ones(20)]
    )
    provider = ReferenceObstacleTVP(
        path, (0.0, 1.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15, velocity_filter=1.0,
    )
    provider.update(
        (0.0, 0.0, 1.0), (0.0, 0.96, 1.0),
        control_time=0.30, measurement_time=0.20,
    )

    stage_zero = provider.prediction_at_stage(0)
    stage_one = provider.prediction_at_stage(1)

    assert stage_zero.obstacle_position == pytest.approx((0.0, 0.94, 1.0))
    assert stage_zero.obstacle_next_position == pytest.approx((0.0, 0.932, 1.0))
    assert stage_zero.obstacle_next_position == pytest.approx(
        stage_one.obstacle_position
    )
    assert stage_zero.robust_radius_next == pytest.approx(stage_one.robust_radius)
    assert stage_one.robust_radius_next > stage_one.robust_radius
    with pytest.raises(ValueError, match=r"\[0, horizon\]"):
        provider.prediction_at_stage(16)


def test_tube_uses_bootstrap_velocity_bound_until_first_distinct_measurement():
    path = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.zeros(20), np.ones(20)]
    )
    provider = ReferenceObstacleTVP(
        path,
        (0.0, 1.0, 1.0),
        reference_speed=0.1,
        obstacle_radius=0.1,
        collision_radius=0.035,
        gamma=0.1,
        dt=0.04,
        horizon=15,
        velocity_filter=1.0,
    )
    bootstrap = provider.uncertainty_at_age(0.6)
    provider.update(
        (0.0, 0.0, 1.0),
        (0.0, 0.88, 1.0),
        control_time=0.6,
        measurement_time=0.6,
    )
    identified = provider.uncertainty_at_age(0.6)

    assert bootstrap > identified
    assert bootstrap == pytest.approx(0.170)
    assert identified == pytest.approx(0.068)


@pytest.mark.skipif(
    importlib.util.find_spec("casadi") is None,
    reason="CasADi is optional",
)
def test_stage_zero_cbf_jacobian_exposes_relative_degree_mismatch() -> None:
    import casadi as ca

    class FakeModel:
        def __init__(self, command):
            self.u = {"u": command}

        def set_variable(self, _kind, name, shape):
            return ca.MX.sym(name, *shape)

    class FakeMPC:
        def __init__(self):
            self.expression = None
            self.template = {}

        def set_nl_cons(self, _name, expression, *, ub):
            assert ub == 0.0
            self.expression = expression

        def get_tvp_template(self):
            return self.template

        def set_tvp_fun(self, function):
            self.tvp_fun = function

    path = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.zeros(20), np.ones(20)]
    )
    state = ca.MX.sym("state", 8, 1)
    command = ca.MX.sym("command", 4, 1)
    dependencies = {}
    for mode in ("paper_state", "command_velocity", "double_integrator"):
        provider = ReferenceObstacleTVP(
            path, (0.0, 2.0, 1.0), reference_speed=0.1,
            obstacle_radius=0.1, collision_radius=0.035,
            gamma=0.1, dt=0.04, horizon=15,
            cbf_transition_mode=mode,
        )
        model = FakeModel(command)
        mpc = FakeMPC()
        provider.declare(model, state, command, ca)
        provider.configure(model, mpc, state, command, ca)
        dependencies[mode] = bool(ca.depends_on(mpc.expression, command))

    assert dependencies == {
        "paper_state": False,
        "command_velocity": True,
        "double_integrator": True,
    }


@pytest.mark.skipif(
    importlib.util.find_spec("casadi") is None,
    reason="CasADi is optional",
)
def test_double_integrator_cbf_successor_matches_discrete_rhs() -> None:
    import casadi as ca

    from lampc_cbf.controller import double_integrator_dynamics_matrices

    class FakeModel:
        def __init__(self, command, rhs):
            self.u = {"u": command}
            self.rhs_list = [{"var_name": "x", "expr": rhs}]

        def set_variable(self, _kind, name, shape):
            return ca.MX.sym(name, *shape)

    class FakeMPC:
        def __init__(self):
            self.expression = None
            self.template = {}

        def set_nl_cons(self, _name, expression, *, ub):
            assert ub == 0.0
            self.expression = expression

        def get_tvp_template(self):
            return self.template

        def set_tvp_fun(self, function):
            self.tvp_fun = function

    path = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.zeros(20), np.ones(20)]
    )
    state = ca.MX.sym("state_di", 8, 1)
    command = ca.MX.sym("command_di", 4, 1)
    dt = 0.04
    rhs = ca.vertcat(
        state[:3] + dt * state[4:7] + 0.5 * dt**2 * command[:3],
        state[3] + dt * command[3],
        state[4:7] + dt * command[:3],
        command[3],
    )
    model = FakeModel(command, rhs)
    mpc = FakeMPC()
    provider = ReferenceObstacleTVP(
        path, (0.0, 2.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=dt, horizon=15,
        cbf_transition_mode="double_integrator",
    )
    provider.declare(model, state, command, ca)
    provider.configure(model, mpc, state, command, ca)

    evaluate = ca.Function(
        "evaluate_double_integrator_cbf",
        [
            state,
            command,
            provider._obstacle_tvp,
            provider._obstacle_next_tvp,
            provider._robust_radius_tvp,
            provider._robust_radius_next_tvp,
            provider._gamma_tvp,
        ],
        [mpc.expression],
    )
    state_value = np.array([[1.0], [2.0], [3.0], [0.1], [0.2], [-0.1], [0.3], [0.0]])
    command_value = np.array([[0.5], [-0.25], [0.4], [0.2]])
    obstacle_value = np.array([[1.4], [2.2], [3.1]])
    next_obstacle_value = np.array([[1.35], [2.15], [3.05]])
    radius = 0.135
    actual_constraint = float(
        evaluate(
            state_value,
            command_value,
            obstacle_value,
            next_obstacle_value,
            radius,
            radius,
            1.0,
        )
    )
    position = state_value[:3, 0]
    velocity = state_value[4:7, 0]
    acceleration = command_value[:3, 0]
    a_values, b_values = double_integrator_dynamics_matrices(dt)
    successor_state = np.asarray(
        [
            sum(a_values[row][column] * state_value[column, 0] for column in range(8))
            + sum(b_values[row][column] * command_value[column, 0] for column in range(4))
            for row in range(8)
        ],
        dtype=float,
    )
    successor = successor_state[:3]
    expected_h_next = float(np.sum((successor - next_obstacle_value[:, 0]) ** 2) - radius**2)
    assert actual_constraint == pytest.approx(-expected_h_next)
