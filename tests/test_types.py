import pytest

from lampc_cbf.types import ControlInput, Obstacle, RobotState


def test_state_matches_paper_order() -> None:
    state = RobotState.from_vector((1, 2, 3, 4, 5, 6, 7, 8))

    assert state.position == (1, 2, 3)
    assert state.as_vector() == (1, 2, 3, 4, 5, 6, 7, 8)


def test_domain_types_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="eight"):
        RobotState.from_vector((1, 2, 3))
    with pytest.raises(ValueError, match="finite"):
        ControlInput(0, 0, 0, float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        Obstacle((0, 0, 0), radius=-1)

