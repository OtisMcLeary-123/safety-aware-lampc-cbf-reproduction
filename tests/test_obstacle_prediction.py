import pytest

from lampc_cbf.obstacle_prediction import (
    ConstantVelocityObserver,
    UncertaintyTubeConfig,
    predict_constant_velocity,
)


def test_constant_velocity_prediction():
    assert predict_constant_velocity((1, 2, 3), (0.5, -1, 0), 0.4) == pytest.approx(
        (1.2, 1.6, 3.0)
    )


def test_uncertainty_tube_grows_with_age_and_includes_latency():
    tube = UncertaintyTubeConfig(
        measurement_sigma=0.005,
        confidence_multiplier=3.0,
        velocity_error_bound=0.03,
        model_error_growth=0.005,
        max_relative_speed=0.4,
        total_latency=0.04,
    )
    assert tube.measurement_bound == pytest.approx(0.015)
    assert tube.latency_bound == pytest.approx(0.016)
    assert tube.inflation(0.0) == pytest.approx(0.031)
    assert tube.inflation(0.6) == pytest.approx(0.052)


def test_observer_estimates_velocity_and_predicts_from_timestamp():
    observer = ConstantVelocityObserver((0.0, 0.0, 0.0), velocity_filter=1.0)
    assert observer.observe((0.1, -0.2, 0.0), 0.5)
    assert observer.velocity == pytest.approx((0.2, -0.4, 0.0))
    assert observer.predict(1.0) == pytest.approx((0.2, -0.4, 0.0))


def test_observer_rejects_time_reversal():
    observer = ConstantVelocityObserver((0.0, 0.0, 0.0), initial_time=1.0)
    with pytest.raises(ValueError, match="monotone"):
        observer.observe((0.0, 0.0, 0.0), 0.9)


@pytest.mark.parametrize("mode", ["unknown", "cv"])
def test_smooth_config_rejects_unknown_prediction_mode(mode):
    from lampc_cbf.smooth_dynamic_demo import SmoothDynamicConfig

    with pytest.raises(ValueError, match="prediction_mode"):
        SmoothDynamicConfig(prediction_mode=mode)
