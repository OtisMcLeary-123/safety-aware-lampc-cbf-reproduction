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
