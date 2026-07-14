from __future__ import annotations

import numpy as np
import pytest

from lampc_cbf.smooth_dynamic_demo import ReferenceObstacleTVP, SmoothDynamicConfig


def test_smooth_dynamic_configuration_validates_weights() -> None:
    assert SmoothDynamicConfig(delta_u_weight=2.0, jerk_weight=0.1).jerk_weight == 0.1
    with pytest.raises(ValueError, match="non-negative"):
        SmoothDynamicConfig(jerk_weight=-0.1)


def test_dynamic_configuration_accepts_velocity_only_and_direct_target():
    config = SmoothDynamicConfig(
        prediction_mode="velocity", reference_mode="direct_target"
    )
    assert config.prediction_mode == "velocity"
    assert config.reference_mode == "direct_target"


def test_reference_progress_is_monotone() -> None:
    path = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.zeros(20), np.ones(20)]
    )
    provider = ReferenceObstacleTVP(
        path, (0.0, 2.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15,
    )

    provider.update((0.6, 0.0, 1.0), (0.0, 2.0, 1.0))
    forward_index = provider.progress_index
    provider.update((0.2, 0.0, 1.0), (0.0, 2.0, 1.0))

    assert forward_index > 0
    assert provider.progress_index == forward_index


def test_reference_provider_hot_swaps_valid_gamma() -> None:
    path = np.column_stack(
        [np.linspace(0.0, 1.0, 20), np.zeros(20), np.ones(20)]
    )
    provider = ReferenceObstacleTVP(
        path, (0.0, 2.0, 1.0), reference_speed=0.1,
        obstacle_radius=0.1, collision_radius=0.035,
        gamma=0.1, dt=0.04, horizon=15,
    )

    provider.update_gamma(0.03)

    assert provider.gamma == pytest.approx(0.03)
    with pytest.raises(ValueError, match="experimental interval"):
        provider.update_gamma(0.2)
