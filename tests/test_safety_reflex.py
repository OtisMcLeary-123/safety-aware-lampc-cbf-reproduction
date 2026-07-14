import pytest

from lampc_cbf.safety_reflex import (
    OperationalSpaceSafetyReflex,
    ReflexObstacle,
    SafetyReflexConfig,
)


def _reflex(**overrides):
    values = {
        "dt": 0.04,
        "lookahead_steps": 8,
        "cbf_alpha": 4.0,
        "speed_limit": 0.2,
        "uncertainty_growth_per_second": 0.0,
    }
    values.update(overrides)
    return OperationalSpaceSafetyReflex(SafetyReflexConfig(**values))


def test_gatekeeper_passes_safe_nominal_velocity():
    reflex = _reflex()
    obstacle = ReflexObstacle((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.05)
    result = reflex.gate((0.0, 0.0, 0.0), (0.0, 0.1, 0.0), (obstacle,))
    assert not result.intervened
    assert result.reason == "nominal_safe"


def test_oscbf_projects_velocity_away_from_spherical_obstacle():
    reflex = _reflex()
    obstacle = ReflexObstacle((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), 0.05)
    result = reflex.gate((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (obstacle,))
    assert result.intervened
    assert not result.backup_used
    assert result.reason == "oscbf_projection"
    assert result.velocity[0] == pytest.approx(0.15)
    assert result.filtered_minimum_clearance >= 0.0
    assert reflex.cbf_residual((0, 0, 0), result.velocity, obstacle) >= -1e-9


def test_gatekeeper_uses_backup_when_short_rollout_remains_unsafe():
    reflex = _reflex(lookahead_steps=20)
    obstacle = ReflexObstacle((0.08, 0.0, 0.0), (-0.2, 0.0, 0.0), 0.05)
    result = reflex.gate((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (obstacle,))
    assert result.intervened
    assert result.backup_used
    stationary_clearance = reflex.rollout_minimum_clearance(
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (obstacle,)
    )
    assert result.reason == "escape_policy"
    assert result.velocity != pytest.approx((0.0, 0.0, 0.0))
    assert result.filtered_minimum_clearance > stationary_clearance


def test_best_effort_escape_never_replaces_motion_with_worse_stationary_backup():
    reflex = _reflex(lookahead_steps=30, speed_limit=0.05)
    obstacle = ReflexObstacle((0.04, 0.0, 0.0), (-0.2, 0.0, 0.0), 0.05)
    result = reflex.gate((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (obstacle,))
    stationary_clearance = reflex.rollout_minimum_clearance(
        (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (obstacle,)
    )

    assert result.reason == "best_effort_escape"
    assert result.filtered_minimum_clearance >= stationary_clearance


def test_uncertainty_triggers_earlier_intervention():
    reflex = _reflex()
    nominal = (0.1, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)
    certain = ReflexObstacle((0.2, 0, 0), (0, 0, 0), 0.05)
    uncertain = ReflexObstacle((0.2, 0, 0), (0, 0, 0), 0.05, uncertainty=0.125)
    assert not reflex.gate(position, nominal, (certain,)).intervened
    assert reflex.gate(position, nominal, (uncertain,)).intervened


def test_projection_handles_multiple_obstacles_and_speed_bound():
    reflex = _reflex()
    obstacles = (
        ReflexObstacle((0.12, 0.0, 0.0), (0, 0, 0), 0.05),
        ReflexObstacle((0.0, 0.12, 0.0), (0, 0, 0), 0.05),
    )
    velocity, violation = reflex.project((0, 0, 0), (0.2, 0.2, 0), obstacles)
    assert sum(value * value for value in velocity) ** 0.5 <= 0.2 + 1e-12
    assert violation <= 1e-9


@pytest.mark.parametrize("radius", [-0.1, float("nan")])
def test_obstacle_rejects_invalid_radius(radius):
    with pytest.raises(ValueError):
        ReflexObstacle((0, 0, 0), (0, 0, 0), radius)
