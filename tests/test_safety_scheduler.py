import pytest

from lampc_cbf.safety_scheduler import (
    ContextAwareSafetyConfig,
    ContextAwareSafetyScheduler,
    SafetyProfile,
    SafetyProfileLifecycle,
    SafetyProfileState,
    constant_velocity_ttc,
    feedback_has_causal_opportunity,
    feedback_update_deadline,
)


def test_feedback_deadline_preserves_request_time_and_hazard_margin() -> None:
    assert feedback_update_deadline(2.0, 0.8, 1.0, reaction_margin=0.2) == pytest.approx(2.6)
    assert feedback_update_deadline(2.0, 4.0, 1.0, reaction_margin=0.2) == pytest.approx(3.0)


def test_feedback_opportunity_requires_latency_and_reaction_margin():
    assert feedback_has_causal_opportunity(3.0, 2.1, reaction_margin=0.2)
    assert not feedback_has_causal_opportunity(2.2, 2.1, reaction_margin=0.2)
    assert not feedback_has_causal_opportunity(None, 0.1)


def test_constant_velocity_ttc_detects_approach_and_ignores_departure():
    assert constant_velocity_ttc(
        (1.0, 0.0, 0.0), (-0.5, 0.0, 0.0), 0.1
    ) == pytest.approx(1.8)
    assert constant_velocity_ttc(
        (1.0, 0.0, 0.0), (0.5, 0.0, 0.0), 0.1
    ) is None


def test_scheduler_uses_ttc_before_language_preference():
    scheduler = ContextAwareSafetyScheduler()
    emergency = scheduler.select(predicted_ttc=0.8, requested_safety_level=5)
    assert emergency.emergency
    assert emergency.gamma == pytest.approx(0.02)
    assert emergency.speed_scale < 1.0


def test_scheduler_preserves_cautious_language_outside_emergency():
    profile = ContextAwareSafetyScheduler().select(
        predicted_ttc=10.0, requested_safety_level=1
    )
    assert not profile.emergency
    assert profile.gamma == pytest.approx(0.05)
    assert profile.clearance_margin > 0.0


def test_infeasible_solver_routes_to_local_reflex_not_stricter_mpc():
    profile = ContextAwareSafetyScheduler().select(
        predicted_ttc=10.0, solver_feasible=False
    )
    assert profile.emergency
    assert profile.reason == "solver_infeasible_use_local_reflex"
    assert profile.gamma == pytest.approx(0.15)


def test_validated_feedback_recovers_runtime_profile_but_retains_gamma():
    config = ContextAwareSafetyConfig(
        clear_hold_time=0.08,
        recovery_duration=0.08,
        exit_ttc_hysteresis=0.5,
    )
    lifecycle = SafetyProfileLifecycle(0.10, config)
    lifecycle.activate_provisional()

    emergency = lifecycle.step(predicted_ttc=0.8, dt=0.04)
    assert lifecycle.state is SafetyProfileState.PROVISIONAL
    assert emergency.speed_scale == pytest.approx(config.emergency_speed_scale)

    lifecycle.accept_validated_update(0.03)
    assert lifecycle.state is SafetyProfileState.VALIDATED_CAUTIOUS
    held = lifecycle.step(predicted_ttc=None, dt=0.04)
    assert held.gamma == pytest.approx(0.03)
    assert held.speed_scale < 1.0

    lifecycle.step(predicted_ttc=None, dt=0.04)
    assert lifecycle.state is SafetyProfileState.RECOVERY
    recovering = lifecycle.step(predicted_ttc=None, dt=0.04)
    assert config.emergency_speed_scale < recovering.speed_scale < 1.0
    nominal = lifecycle.step(predicted_ttc=None, dt=0.04)

    assert lifecycle.state is SafetyProfileState.NOMINAL
    assert nominal.gamma == pytest.approx(0.03)
    assert nominal.clearance_margin == pytest.approx(0.0)
    assert nominal.speed_scale == pytest.approx(1.0)


def test_ttc_hysteresis_and_solver_failure_prevent_early_release():
    config = ContextAwareSafetyConfig(
        clear_hold_time=0.08,
        recovery_duration=0.08,
        exit_ttc_hysteresis=0.5,
    )
    lifecycle = SafetyProfileLifecycle(0.10, config)
    lifecycle.activate_provisional()

    lifecycle.step(predicted_ttc=2.9, dt=0.04)
    lifecycle.step(predicted_ttc=3.1, dt=0.04)
    lifecycle.step(predicted_ttc=2.9, dt=0.04)
    lifecycle.step(predicted_ttc=3.1, dt=0.04)
    assert lifecycle.state is SafetyProfileState.PROVISIONAL

    lifecycle.step(predicted_ttc=3.1, dt=0.04, solver_feasible=False)
    lifecycle.step(predicted_ttc=3.1, dt=0.04)
    assert lifecycle.state is SafetyProfileState.PROVISIONAL
    lifecycle.step(predicted_ttc=3.1, dt=0.04)
    assert lifecycle.state is SafetyProfileState.RECOVERY


def test_hazard_reappearing_during_recovery_returns_to_cautious_state():
    config = ContextAwareSafetyConfig(clear_hold_time=0.04, recovery_duration=0.20)
    lifecycle = SafetyProfileLifecycle(0.10, config)
    lifecycle.activate_provisional()
    lifecycle.accept_validated_update(0.04)
    lifecycle.step(predicted_ttc=None, dt=0.04)
    lifecycle.step(predicted_ttc=None, dt=0.04)
    assert lifecycle.state is SafetyProfileState.RECOVERY

    profile = lifecycle.step(predicted_ttc=0.7, dt=0.04)

    assert lifecycle.state is SafetyProfileState.VALIDATED_CAUTIOUS
    assert profile.emergency
    assert profile.gamma == pytest.approx(0.04)
    assert profile.speed_scale == pytest.approx(config.emergency_speed_scale)


def test_expired_feedback_window_can_release_after_observed_hazard_clears():
    config = ContextAwareSafetyConfig(clear_hold_time=0.04, recovery_duration=0.04)
    lifecycle = SafetyProfileLifecycle(0.10, config)
    lifecycle.activate_provisional()

    lifecycle.step(predicted_ttc=None, dt=0.04)
    assert lifecycle.state is SafetyProfileState.RECOVERY
    profile = lifecycle.step(predicted_ttc=None, dt=0.04)

    assert lifecycle.state is SafetyProfileState.NOMINAL
    assert profile.speed_scale == pytest.approx(1.0)
    assert profile.clearance_margin == pytest.approx(0.0)


def test_ablation_can_disable_recovery_without_disabling_safety_profile():
    config = ContextAwareSafetyConfig(
        clear_hold_time=0.04,
        recovery_duration=0.04,
        recovery_enabled=False,
    )
    lifecycle = SafetyProfileLifecycle(0.10, config)
    lifecycle.activate_provisional()
    lifecycle.accept_validated_update(0.03)

    for _ in range(20):
        profile = lifecycle.step(predicted_ttc=None, dt=0.04)

    assert lifecycle.state is SafetyProfileState.VALIDATED_CAUTIOUS
    assert profile.gamma == pytest.approx(0.03)
    assert profile.speed_scale == pytest.approx(config.cautious_speed_scale)
    assert profile.clearance_margin == pytest.approx(
        config.cautious_clearance_margin
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 0.0},
        {"gamma": 0.1, "clearance_margin": -0.1},
        {"gamma": 0.1, "speed_scale": 1.1},
    ],
)
def test_safety_profile_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        SafetyProfile(**kwargs)
