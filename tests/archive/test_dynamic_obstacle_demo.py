from __future__ import annotations

import numpy as np
import pytest

from lampc_cbf.dynamic_obstacle_demo import (
    DynamicObstacleConfig,
    OnlineObstacleCBF,
)


def test_paper_dynamic_obstacle_defaults() -> None:
    config = DynamicObstacleConfig()

    assert config.sensor_period == pytest.approx(0.67)
    assert config.measurement_noise_sigma == pytest.approx(0.005)
    assert config.gamma <= 0.15
    assert config.obstacle_velocity == pytest.approx((0.06, 0.0, 0.0))


def test_online_cbf_measurement_is_updated_safely() -> None:
    cbf = OnlineObstacleCBF(
        (0.0, 0.1, 0.2), obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15,
    )

    cbf.update_measurement((0.2, 0.3, 0.4))

    assert cbf.measurement == pytest.approx((0.2, 0.3, 0.4))
    with pytest.raises(ValueError, match="finite 3-vector"):
        cbf.update_measurement((np.nan, 0.0, 0.0))
