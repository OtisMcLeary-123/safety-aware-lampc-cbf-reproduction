import pytest

from lampc_cbf.safety_scheduler import (
    ContextAwareSafetyScheduler,
    SafetyProfile,
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
