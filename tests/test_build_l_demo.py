import pytest

from lampc_cbf.build_l_demo import BuildLDemoConfig, _evaluate_task_success


def test_build_l_config_defaults() -> None:
    config = BuildLDemoConfig()
    assert config.gamma == pytest.approx(0.15)
    assert config.max_move_steps == 100
    assert config.render_stride == 5


@pytest.mark.parametrize("gamma", [0.0, 1.1])
def test_build_l_config_rejects_invalid_gamma(gamma: float) -> None:
    with pytest.raises(ValueError, match="gamma"):
        BuildLDemoConfig(gamma=gamma)


def test_language_guided_blue_on_red_configuration() -> None:
    config = BuildLDemoConfig(
        gamma=0.05,
        cube_indices=(0,),
        place_blue_on_red=True,
        dynamic_obstacle=True,
        user_instruction="Safely put the blue cube on the red cube.",
        llm_safety_level=2,
    )

    assert config.cube_indices == (0,)
    assert config.dynamic_obstacle is True


def test_blue_on_red_rejects_multiple_cube_indices() -> None:
    with pytest.raises(ValueError, match="blue-on-red"):
        BuildLDemoConfig(place_blue_on_red=True)


def test_advisory_waypoint_does_not_fail_successful_safe_task() -> None:
    success, collision_free = _evaluate_task_success(
        [
            {"stage": "approach", "reached": True, "required_for_success": True},
            {
                "stage": "transport_avoid",
                "reached": False,
                "required_for_success": False,
            },
            {"stage": "place", "reached": True, "required_for_success": True},
        ],
        cubes_placed=1,
        expected_cubes=1,
        clearances=(0.02,),
        dynamic_clearances=(0.03,),
    )
    assert success
    assert collision_free


def test_collision_or_failed_required_stage_still_fails_task() -> None:
    failed_stage, _ = _evaluate_task_success(
        [{"stage": "place", "reached": False, "required_for_success": True}],
        cubes_placed=1,
        expected_cubes=1,
        clearances=(0.02,),
        dynamic_clearances=(0.03,),
    )
    collision, collision_free = _evaluate_task_success(
        [{"stage": "place", "reached": True, "required_for_success": True}],
        cubes_placed=1,
        expected_cubes=1,
        clearances=(0.02,),
        dynamic_clearances=(-0.001,),
    )
    assert not failed_stage
    assert not collision
    assert not collision_free
