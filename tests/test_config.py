import pytest

from lampc_cbf.config import PaperConfig


def test_paper_defaults() -> None:
    config = PaperConfig()

    assert config.dt == 0.04
    assert config.horizon == 15
    assert config.prediction_duration == pytest.approx(0.6)
    assert config.state_weights == (1.0,) * 8
    assert config.linear_input_weight == 0.5
    assert config.angular_input_weight == 1e-5
    assert config.velocity_regularization_weight == 0.1
    assert config.rotation_regularization_weight == 5e-5


def test_gamma_ranges_distinguish_experiment_from_theory() -> None:
    config = PaperConfig()

    assert config.validate_gamma(0.15) == 0.15
    with pytest.raises(ValueError, match="experimental"):
        config.validate_gamma(0.2)
    assert config.validate_gamma(1.0, experimental=False) == 1.0


@pytest.mark.parametrize("gamma", [0.0, -0.1, float("inf"), float("nan")])
def test_invalid_gamma_is_rejected(gamma: float) -> None:
    with pytest.raises(ValueError):
        PaperConfig(gamma=gamma)

