import pytest

from lampc_cbf.build_l_demo import BuildLDemoConfig


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
