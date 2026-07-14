import pytest

from lampc_cbf.formal_safety import (
    FormalDiscreteSafetyFilter,
    FormalObstacle,
    FormalSafetyConfig,
)


def _filter(**overrides):
    return FormalDiscreteSafetyFilter(FormalSafetyConfig(**overrides))


def test_nominal_is_unchanged_when_robust_cbf_and_backup_set_are_safe():
    safety_filter = _filter(robot_transition_error_bound=0.001)
    obstacle = FormalObstacle((0.0, 0.0, 0.0), (0.0, -0.1, 0.0), 0.135)
    result = safety_filter.filter((0.4, 0.0, 0.0), (0.0, 0.1, 0.0), obstacle)
    assert result.one_step_certified
    assert result.terminal_backup_certified
    assert result.robust_cbf_residual >= -1e-9
    assert not result.intervened


def test_filter_projects_unsafe_nominal_to_certified_command():
    safety_filter = _filter(robot_transition_error_bound=0.001)
    obstacle = FormalObstacle((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), 0.135)
    result = safety_filter.filter((0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), obstacle)
    assert result.intervened
    assert result.one_step_certified
    assert result.terminal_backup_certified
    assert result.velocity != pytest.approx((-0.2, 0.0, 0.0))
    assert result.robust_cbf_residual >= -1e-9


def test_certificate_fails_closed_without_backup_control_authority():
    safety_filter = _filter(
        speed_limit=0.2,
        obstacle_speed_bound=0.2,
        robot_transition_error_bound=0.001,
    )
    obstacle = FormalObstacle((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), 0.135)
    result = safety_filter.filter((0.2, 0.0, 0.0), (0.0, 0.0, 0.0), obstacle)
    assert result.backup_authority_margin < 0.0
    assert not result.one_step_certified
    assert not result.terminal_backup_certified


def test_bounds_are_validated():
    with pytest.raises(ValueError):
        FormalSafetyConfig(robot_transition_error_bound=-0.1)
