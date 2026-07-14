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
        "committed_backup_enabled": False,
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
    assert result.reason == "task_consistent_escape"
    assert result.velocity != pytest.approx((0.0, 0.0, 0.0))
    assert result.filtered_minimum_clearance > stationary_clearance


def test_task_consistent_backup_prefers_safe_command_closest_to_nominal():
    reflex = _reflex(lookahead_steps=20)
    obstacle = ReflexObstacle((0.08, 0.0, 0.0), (-0.2, 0.0, 0.0), 0.05)
    nominal = (0.2, 0.0, 0.0)

    task_consistent = reflex.gate((0.0, 0.0, 0.0), nominal, (obstacle,))
    max_clearance = _reflex(
        lookahead_steps=20, backup_selection="max_clearance"
    ).gate((0.0, 0.0, 0.0), nominal, (obstacle,))

    task_deviation = sum(
        (value - target) ** 2
        for value, target in zip(task_consistent.velocity, nominal)
    )
    clearance_deviation = sum(
        (value - target) ** 2
        for value, target in zip(max_clearance.velocity, nominal)
    )
    assert task_consistent.reason == "task_consistent_escape"
    assert max_clearance.reason == "max_clearance_escape"
    assert task_consistent.filtered_minimum_clearance >= 0.0
    assert task_consistent.maximum_cbf_violation <= 1e-9
    assert task_deviation <= clearance_deviation


def test_committed_backup_is_reused_until_nominal_becomes_safe():
    reflex = _reflex(
        lookahead_steps=20,
        committed_backup_enabled=True,
        committed_backup_steps=3,
    )
    obstacle = ReflexObstacle((0.08, 0.0, 0.0), (-0.2, 0.0, 0.0), 0.05)

    first = reflex.gate((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (obstacle,))
    second = reflex.gate((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (obstacle,))

    assert first.reason == "task_consistent_escape_committed"
    assert second.reason == "committed_backup"
    assert second.velocity == pytest.approx(first.velocity)
    assert reflex.committed_backup_steps_remaining == 1

    safe_obstacle = ReflexObstacle((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.05)
    released = reflex.gate((0.0, 0.0, 0.0), (0.0, 0.1, 0.0), (safe_obstacle,))
    assert released.reason == "nominal_safe"
    assert reflex.committed_backup_steps_remaining == 0


def test_invalidated_committed_backup_is_not_reused():
    reflex = _reflex(
        lookahead_steps=20,
        committed_backup_enabled=True,
        committed_backup_steps=3,
    )
    obstacle = ReflexObstacle((0.08, 0.0, 0.0), (-0.2, 0.0, 0.0), 0.05)
    first = reflex.gate((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (obstacle,))
    changed = ReflexObstacle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.05)

    result = reflex.gate((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (changed,))

    assert first.backup_used
    assert result.reason != "committed_backup"


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


def test_collision_cone_distinguishes_approaching_and_receding_motion():
    reflex = _reflex(barrier_mode="collision_cone")
    obstacle = ReflexObstacle((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.1)

    approaching = reflex.collision_cone_residual(
        (0.0, 0.0, 0.0), (0.2, 0.0, 0.0), obstacle
    )
    receding = reflex.collision_cone_residual(
        (0.0, 0.0, 0.0), (-0.2, 0.0, 0.0), obstacle
    )

    assert approaching < 0.0
    assert receding > 0.0


def test_dynamic_parabolic_barrier_adapts_to_distance_and_lateral_motion():
    reflex = _reflex(barrier_mode="dynamic_parabolic")
    near = ReflexObstacle((0.06, 0.0, 0.0), (0.0, 0.0, 0.0), 0.05)
    far = ReflexObstacle((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.05)

    near_approach = reflex.dynamic_parabolic_residual(
        (0.0, 0.0, 0.0), (0.2, 0.0, 0.0), near
    )
    far_approach = reflex.dynamic_parabolic_residual(
        (0.0, 0.0, 0.0), (0.2, 0.0, 0.0), far
    )
    lateral = reflex.dynamic_parabolic_residual(
        (0.0, 0.0, 0.0), (0.0, 0.2, 0.0), near
    )
    receding = reflex.dynamic_parabolic_residual(
        (0.0, 0.0, 0.0), (-0.2, 0.0, 0.0), near
    )

    assert near_approach < 0.0
    assert far_approach > near_approach
    assert lateral > near_approach
    assert receding > near_approach


def test_dynamic_parabolic_policy_library_keeps_physical_rollout_safe():
    reflex = _reflex(
        barrier_mode="dynamic_parabolic",
        side_latch_enabled=True,
        policy_library_enabled=True,
    )
    obstacle = ReflexObstacle((0.08, 0.0, 0.0), (0.0, 0.0, 0.0), 0.05)

    result = reflex.gate(
        (0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (obstacle,),
        goal_position=(0.3, 0.0, 0.0),
    )

    assert result.intervened
    assert result.filtered_minimum_clearance >= 0.0
    assert result.maximum_cbf_violation <= 1e-9


def test_collision_cone_projection_is_safe_and_minimally_changes_nominal():
    reflex = _reflex(barrier_mode="collision_cone")
    obstacle = ReflexObstacle((0.2, 0.0, 0.0), (0.0, 0.0, 0.0), 0.05)
    nominal = (0.2, 0.0, 0.0)

    projected, violation = reflex.project((0.0, 0.0, 0.0), nominal, (obstacle,))

    assert violation <= 1e-9
    assert reflex.collision_cone_residual(
        (0.0, 0.0, 0.0), projected, obstacle
    ) >= -1e-9
    assert sum(value * value for value in projected) ** 0.5 <= 0.2 + 1e-12


def test_policy_library_latches_collision_cone_side_across_nominal_perturbations():
    reflex = _reflex(
        barrier_mode="collision_cone",
        side_latch_enabled=True,
        side_latch_steps=3,
        policy_library_enabled=True,
    )
    obstacle = ReflexObstacle((0.0, 0.2, 0.0), (0.0, 0.0, 0.0), 0.05)

    first = reflex.gate(
        (0.0, 0.0, 0.0), (0.001, 0.2, 0.0), (obstacle,),
        goal_position=(0.0, 0.3, 0.0),
    )
    perturbed = reflex.gate(
        (0.0, 0.0, 0.0), (-0.001, 0.2, 0.0), (obstacle,),
        goal_position=(0.0, 0.3, 0.0),
    )

    assert first.intervened and perturbed.intervened
    assert first.avoidance_side in (-1, 1)
    assert perturbed.avoidance_side == first.avoidance_side
    assert reflex.side_switches == 0
    assert perturbed.filtered_minimum_clearance >= 0.0
    assert perturbed.maximum_cbf_violation <= 1e-9


def test_policy_library_emits_same_side_tangential_subgoal():
    reflex = _reflex(
        barrier_mode="collision_cone",
        side_latch_enabled=True,
        policy_library_enabled=True,
        tangential_subgoal_enabled=True,
    )
    obstacle = ReflexObstacle((0.0, 0.2, 0.0), (0.0, 0.0, 0.0), 0.05)

    result = reflex.gate(
        (0.0, 0.0, 0.0), (0.0, 0.2, 0.0), (obstacle,),
        goal_position=(0.0, 0.3, 0.0),
    )

    assert result.avoidance_side in (-1, 1)
    assert result.temporary_subgoal is not None
    assert result.filtered_minimum_clearance >= 0.0
    tangent = reflex._tangent_direction((0.0, 0.0, 0.0), obstacle)
    subgoal_direction = tuple(
        value for value in result.temporary_subgoal
    )
    assert result.avoidance_side * sum(
        a * b for a, b in zip(subgoal_direction, tangent)
    ) > 0.0


def test_policy_library_configuration_requires_collision_cone():
    with pytest.raises(ValueError, match="collision_cone"):
        SafetyReflexConfig(policy_library_enabled=True)
    with pytest.raises(ValueError, match="policy library"):
        SafetyReflexConfig(tangential_subgoal_enabled=True)


def test_policy_library_labels_robust_recovery_and_keeps_physical_safety():
    reflex = _reflex(
        barrier_mode="collision_cone",
        side_latch_enabled=True,
        policy_library_enabled=True,
    )
    obstacle = ReflexObstacle(
        (0.15, 0.0, 0.0), (0.0, 0.0, 0.0), 0.05, uncertainty=0.11
    )

    result = reflex.gate(
        (0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (obstacle,),
        goal_position=(0.3, 0.0, 0.0),
    )
    physical = ReflexObstacle(obstacle.position, obstacle.velocity, obstacle.radius)

    assert result.reason == "policy_library_recovery"
    assert result.robust_recovery
    assert result.avoidance_side in (-1, 1)
    assert reflex.rollout_minimum_clearance(
        (0.0, 0.0, 0.0), result.velocity, (physical,),
        include_uncertainty_growth=False,
    ) >= 0.0
    assert reflex.collision_cone_residual(
        (0.0, 0.0, 0.0), result.velocity, physical
    ) >= -1e-9


@pytest.mark.parametrize("radius", [-0.1, float("nan")])
def test_obstacle_rejects_invalid_radius(radius):
    with pytest.raises(ValueError):
        ReflexObstacle((0, 0, 0), (0, 0, 0), radius)
