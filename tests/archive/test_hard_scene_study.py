from __future__ import annotations

import numpy as np
import pytest

from lampc_cbf.hard_scene_study import (
    HardSceneStudyConfig,
    _bootstrap_success_interval,
)


def test_hard_scene_configuration_matches_paper_speed_range() -> None:
    config = HardSceneStudyConfig()

    assert config.episodes == 50
    assert config.speed_lower == pytest.approx(0.025)
    assert config.speed_upper == pytest.approx(0.20)
    assert config.bootstrap_resamples == 10_000


def test_bootstrap_interval_for_constant_outcomes_is_exact() -> None:
    rng = np.random.default_rng(7)

    assert _bootstrap_success_interval([True] * 5, 100, rng) == (1.0, 1.0)
    assert _bootstrap_success_interval([False] * 5, 100, rng) == (0.0, 0.0)
