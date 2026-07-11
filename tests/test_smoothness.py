from __future__ import annotations

import numpy as np
import pytest

from lampc_cbf.smoothness import (
    calculate_smoothness_metrics,
    make_reference_bspline,
    make_visual_smoothing_spline,
)


def test_straight_constant_speed_path_has_zero_curvature_and_jerk() -> None:
    t = np.arange(20) * 0.04
    positions = np.column_stack([0.2 * t, np.zeros_like(t), np.ones_like(t)])

    metrics = calculate_smoothness_metrics(positions, 0.04)

    assert metrics.path_length == pytest.approx(0.2 * t[-1])
    assert metrics.mean_curvature == pytest.approx(0.0, abs=1e-10)
    assert metrics.max_curvature == pytest.approx(0.0, abs=1e-10)
    assert metrics.acceleration_rms == pytest.approx(0.0, abs=1e-10)
    assert metrics.jerk_rms == pytest.approx(0.0, abs=1e-9)


def test_reference_bspline_preserves_endpoints() -> None:
    points = np.array(
        [[0.0, 0.0, 0.2], [-0.2, 0.1, 0.2], [-0.2, 0.2, 0.2], [0.0, 0.3, 0.2]]
    )

    reference = make_reference_bspline(points, sample_count=100)
    visual = make_visual_smoothing_spline(reference, sample_count=120)

    assert reference.shape == (100, 3)
    assert visual.shape == (120, 3)
    assert reference[0] == pytest.approx(points[0])
    assert reference[-1] == pytest.approx(points[-1])
