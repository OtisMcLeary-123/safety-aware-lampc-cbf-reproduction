from __future__ import annotations

import pytest

from lampc_cbf.demo import (
    DemoConfig,
    make_static_cbf_builder,
    paper_control_to_safe_panda_action,
)


def test_demo_config_validates_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        DemoConfig(gamma=0.0)


def test_action_mapping_normalizes_paper_input_bounds() -> None:
    np = pytest.importorskip("numpy")
    action = paper_control_to_safe_panda_action((0.2, -0.1, 0.0, 1.0), 3)
    assert action == pytest.approx(np.array([1.0, -0.5, 0.0]))


def test_action_mapping_appends_neutral_gripper() -> None:
    action = paper_control_to_safe_panda_action((0.2, 0.0, 0.0, 1.0), 4)
    assert action == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_cbf_builder_rejects_invalid_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        make_static_cbf_builder(
            ((0.0, 0.0, 0.0),),
            obstacle_radius=0.1,
            collision_radius=0.035,
            gamma=1.1,
        )
