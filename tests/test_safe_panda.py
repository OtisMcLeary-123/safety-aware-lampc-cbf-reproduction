from __future__ import annotations

import importlib.util

import pytest

from lampc_cbf.safe_panda import (
    SafePandaAdapter,
    SafePandaConfig,
    SafePandaDependencyError,
    make_safe_panda,
    simulator_calibration_sample,
)


def test_simulator_calibration_separates_model_and_action_error() -> None:
    sample = simulator_calibration_sample(
        (0.0, 0.0, 0.0),
        (0.01, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        0.04,
    )
    assert sample.model_transition_error == pytest.approx(0.01)
    assert sample.action_tracking_error == pytest.approx(0.0)
from lampc_cbf.types import ControlInput


class FakeEnv:
    def __init__(self) -> None:
        self.ticks = 0
        self.actions = []
        self.reset_seed = None

    def _observation(self):
        return {
            "state": [self.ticks, 2, 3, 0.5, 0.1, 0.2, 0.3, 0.4],
            "obstacles": [
                {"position": [float(self.ticks), 1.0, 2.0], "radius": 0.15, "name": "ball"}
            ],
        }

    def reset(self, *, seed=None):
        self.ticks = 0
        self.reset_seed = seed
        return self._observation(), {"seed": seed}

    def step(self, action):
        self.actions.append(action)
        self.ticks += 1
        return self._observation(), 0.0, self.ticks >= 2, False, {}


def test_state_action_and_gymnasium_done_mapping():
    env = FakeEnv()
    adapter = SafePandaAdapter(
        env,
        SafePandaConfig(env_id="fake", obstacle_noise_sigma=0.0, seed=17),
    )

    state, obstacles = adapter.reset()
    assert env.reset_seed == 17
    assert state.as_vector() == (0.0, 2.0, 3.0, 0.5, 0.1, 0.2, 0.3, 0.4)
    assert obstacles[0].position == (0.0, 1.0, 2.0)

    assert adapter.step(ControlInput(1.0, 2.0, 3.0, 4.0)) is False
    assert env.actions == [[1.0, 2.0, 3.0, 4.0]]
    assert adapter.step(ControlInput(0.0, 0.0, 0.0, 0.0)) is True


def test_zero_order_hold_uses_simulation_time():
    env = FakeEnv()
    adapter = SafePandaAdapter(
        env,
        SafePandaConfig(
            env_id="fake",
            control_dt=0.04,
            sensing_period=0.67,
            obstacle_noise_sigma=0.0,
        ),
    )
    _, initial = adapter.reset()

    for _ in range(16):
        adapter.step(ControlInput(0, 0, 0, 0))
        _, held = adapter.observe()
        assert held == initial

    adapter.step(ControlInput(0, 0, 0, 0))  # t=0.68, first update after 0.67 s
    _, updated = adapter.observe()
    assert updated[0].position == (17.0, 1.0, 2.0)


def test_noise_and_reset_are_deterministic():
    adapter = SafePandaAdapter(FakeEnv(), SafePandaConfig(env_id="fake", seed=123))
    _, first = adapter.reset()
    _, repeated = adapter.reset()
    assert first == repeated
    assert first[0].position != (0.0, 1.0, 2.0)


def test_custom_extractors_and_action_mapper_support_other_apis():
    class OddEnv(FakeEnv):
        def _observation(self):
            return {"robot": list(range(10)), "moving_ball": (4, 5, 6)}

    env = OddEnv()
    config = SafePandaConfig(
        env_id="user-selected-v9",
        state_extractor=lambda obs: obs["robot"][:8],
        obstacle_extractor=lambda obs: [{"center": obs["moving_ball"], "radius": 0.2}],
        action_mapper=lambda action: {"cartesian": action},
        obstacle_noise_sigma=0.0,
    )
    adapter = SafePandaAdapter(env, config)
    state, obstacles = adapter.reset()
    adapter.step(ControlInput(1, 2, 3, 4))

    assert state.as_vector() == tuple(float(value) for value in range(8))
    assert obstacles[0].position == (4.0, 5.0, 6.0)
    assert env.actions[-1] == {"cartesian": (1, 2, 3, 4)}


def test_factory_wraps_missing_environment_with_install_help():
    class FakeBackend:
        @staticmethod
        def make(env_id, **kwargs):
            raise KeyError(env_id)

    with pytest.raises(SafePandaDependencyError, match="user-chosen-id") as error:
        make_safe_panda(SafePandaConfig(env_id="user-chosen-id"), gym_backend=FakeBackend())
    assert "panda-gym" in str(error.value)


@pytest.mark.skipif(
    importlib.util.find_spec("gymnasium") is None
    or (
        importlib.util.find_spec("panda_gym") is None
        and importlib.util.find_spec("safe_panda_gym") is None
    ),
    reason="optional Safe Panda Gym dependencies are not installed",
)
def test_optional_real_dependency_import_smoke():
    # Environment ids differ across Safe Panda Gym releases.  Import success is
    # the only version-neutral smoke assertion; creation is covered by the fake.
    import gymnasium  # noqa: F401

    assert importlib.util.find_spec("panda_gym") or importlib.util.find_spec("safe_panda_gym")
