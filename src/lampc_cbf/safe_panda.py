"""Optional Safe Panda Gym integration.

The paper reports Cartesian state and input vectors, but does not publish a
specific Gym environment id or the observation schema used by its Safe Panda
Gym fork.  This module therefore keeps the simulator dependency optional and
provides explicit extraction/mapping hooks for different package versions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import importlib
import random
from math import isfinite, sqrt
from typing import Any

from .types import ControlInput, Obstacle, RobotState

Observation = Any
StateExtractor = Callable[[Observation], Sequence[float]]
ObstacleExtractor = Callable[[Observation], Sequence[Obstacle | Mapping[str, Any] | Sequence[float]]]
ActionMapper = Callable[[tuple[float, float, float, float]], Any]


@dataclass(frozen=True, slots=True)
class SimulatorCalibrationSample:
    """One-step mismatch between the paper model and Safe Panda motion."""

    model_transition_error: float
    action_tracking_error: float


def simulator_calibration_sample(
    previous_position: Sequence[float],
    next_position: Sequence[float],
    model_velocity: Sequence[float],
    commanded_velocity: Sequence[float],
    dt: float,
) -> SimulatorCalibrationSample:
    """Compare observed Cartesian motion with model-state and action predictions."""

    vectors = (
        tuple(float(value) for value in previous_position),
        tuple(float(value) for value in next_position),
        tuple(float(value) for value in model_velocity),
        tuple(float(value) for value in commanded_velocity),
    )
    if any(len(vector) != 3 for vector in vectors):
        raise ValueError("calibration inputs must be 3-vectors")
    if not isfinite(dt) or dt <= 0.0 or not all(
        isfinite(value) for vector in vectors for value in vector
    ):
        raise ValueError("calibration inputs and dt must be finite; dt must be positive")
    previous, observed, model, commanded = vectors

    def error(velocity: tuple[float, ...]) -> float:
        return sqrt(
            sum(
                (actual - (prior + dt * speed)) ** 2
                for prior, actual, speed in zip(previous, observed, velocity)
            )
        )

    return SimulatorCalibrationSample(error(model), error(commanded))


class SafePandaDependencyError(RuntimeError):
    """Raised when no compatible optional Gym installation can be used."""


@dataclass(slots=True)
class SafePandaConfig:
    """Configuration for one Safe Panda Gym environment.

    Custom extractors are the compatibility boundary for simulator forks.  In
    their absence, the adapter accepts an eight-value observation (or a mapping
    containing ``state``/``robot_state``/``observation``) and common obstacle
    mapping forms.
    """

    env_id: str
    make_kwargs: Mapping[str, Any] = field(default_factory=dict)
    control_dt: float = 0.04
    sensing_period: float = 0.67
    obstacle_noise_sigma: float = 0.005
    seed: int | None = 0
    default_obstacle_radius: float = 0.0
    state_extractor: StateExtractor | None = None
    obstacle_extractor: ObstacleExtractor | None = None
    action_mapper: ActionMapper | None = None

    def __post_init__(self) -> None:
        if not self.env_id:
            raise ValueError("env_id must be non-empty")
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if self.sensing_period <= 0.0:
            raise ValueError("sensing_period must be positive")
        if self.obstacle_noise_sigma < 0.0:
            raise ValueError("obstacle_noise_sigma must be non-negative")
        if self.default_obstacle_radius < 0.0:
            raise ValueError("default_obstacle_radius must be non-negative")


def _flat_numeric(value: Any) -> tuple[float, ...]:
    """Convert array-like input to a flat tuple without requiring NumPy."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        flattened: list[float] = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                flattened.extend(_flat_numeric(item))
            else:
                flattened.append(float(item))
        return tuple(flattened)
    raise ValueError("expected an array-like numeric value")


def _default_state(observation: Observation) -> Sequence[float]:
    candidate = observation
    if isinstance(observation, Mapping):
        for key in ("state", "robot_state", "observation"):
            if key in observation:
                candidate = observation[key]
                break
        else:
            names = ("x", "y", "z", "psi", "dx", "dy", "dz", "dpsi")
            if all(name in observation for name in names):
                candidate = [observation[name] for name in names]
            else:
                raise ValueError(
                    "cannot find robot state; provide state_extractor for this observation schema"
                )
    values = _flat_numeric(candidate)
    if len(values) != 8:
        raise ValueError(
            f"robot state must contain exactly 8 values, got {len(values)}; "
            "provide state_extractor for this environment"
        )
    return values


def _default_obstacles(observation: Observation) -> Sequence[Any]:
    if not isinstance(observation, Mapping):
        return ()
    for key in ("obstacles", "obstacle_states"):
        if key in observation:
            value = observation[key]
            if isinstance(value, Mapping):
                return (value,)
            return value
    for key in ("obstacle", "obstacle_position"):
        if key in observation:
            return ({"position": observation[key]},)
    return ()


def _coerce_obstacle(item: Any, index: int, default_radius: float) -> Obstacle:
    if isinstance(item, Obstacle):
        return item
    if isinstance(item, Mapping):
        position = item.get("position", item.get("pos", item.get("center")))
        if position is None:
            raise ValueError("obstacle mapping must contain position, pos, or center")
        values = _flat_numeric(position)
        radius = float(item.get("radius", default_radius))
        name = str(item.get("name", f"obstacle_{index}"))
    else:
        values = _flat_numeric(item)
        if len(values) == 4:
            values, radius = values[:3], values[3]
        else:
            radius = default_radius
        name = f"obstacle_{index}"
    if len(values) != 3:
        raise ValueError(f"obstacle position must contain 3 values, got {len(values)}")
    return Obstacle(position=(values[0], values[1], values[2]), radius=radius, name=name)


class SafePandaAdapter:
    """Translate Gym observations/actions to the paper's integration types."""

    def __init__(self, env: Any, config: SafePandaConfig) -> None:
        self.env = env
        self.config = config
        self._rng = random.Random(config.seed)
        self._observation: Observation | None = None
        self._sim_time = 0.0
        self._next_sensing_time = 0.0
        self._held_obstacles: tuple[Obstacle, ...] = ()

    def reset(self, *, seed: int | None = None) -> tuple[RobotState, tuple[Obstacle, ...]]:
        """Reset the environment and all deterministic sensor state."""

        actual_seed = self.config.seed if seed is None else seed
        self._rng.seed(actual_seed)
        try:
            result = self.env.reset(seed=actual_seed)
        except TypeError:  # legacy Gym API
            if hasattr(self.env, "seed"):
                self.env.seed(actual_seed)
            result = self.env.reset()
        self._observation = result[0] if isinstance(result, tuple) and len(result) == 2 else result
        self._sim_time = 0.0
        self._next_sensing_time = 0.0
        self._held_obstacles = ()
        return self.observe()

    def observe(self) -> tuple[RobotState, tuple[Obstacle, ...]]:
        if self._observation is None:
            return self.reset()
        extractor = self.config.state_extractor or _default_state
        state = RobotState.from_vector(tuple(float(value) for value in extractor(self._observation)))
        if self._sim_time + 1e-12 >= self._next_sensing_time:
            self._held_obstacles = self._sample_obstacles(self._observation)
            while self._next_sensing_time <= self._sim_time + 1e-12:
                self._next_sensing_time += self.config.sensing_period
        return state, self._held_obstacles

    def _sample_obstacles(self, observation: Observation) -> tuple[Obstacle, ...]:
        extractor = self.config.obstacle_extractor or _default_obstacles
        raw = extractor(observation)
        sampled: list[Obstacle] = []
        sigma = self.config.obstacle_noise_sigma
        for index, item in enumerate(raw):
            obstacle = _coerce_obstacle(item, index, self.config.default_obstacle_radius)
            noisy_position = tuple(
                coordinate + self._rng.gauss(0.0, sigma) for coordinate in obstacle.position
            )
            sampled.append(
                Obstacle(position=noisy_position, radius=obstacle.radius, name=obstacle.name)
            )
        return tuple(sampled)

    def step(self, action: ControlInput) -> bool:
        vector = action.as_vector()
        mapped_action = self.config.action_mapper(vector) if self.config.action_mapper else list(vector)
        result = self.env.step(mapped_action)
        if not isinstance(result, tuple) or len(result) not in (4, 5):
            raise RuntimeError("environment step() must return the Gym 4- or Gymnasium 5-tuple")
        self._observation = result[0]
        self._sim_time += self.config.control_dt
        if len(result) == 5:
            terminated, truncated = bool(result[2]), bool(result[3])
            return terminated or truncated
        return bool(result[2])

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()


def _load_gym_backend() -> Any:
    errors: list[str] = []
    for module_name in ("gymnasium", "gym"):
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
    detail = "; ".join(errors)
    raise SafePandaDependencyError(
        "Safe Panda integration is optional and no Gym backend is installed. "
        "Install the Safe-panda-gym/panda-gym version used by your environment "
        f"together with gymnasium (or legacy gym). Import errors: {detail}"
    )


def make_safe_panda(
    config: SafePandaConfig, *, gym_backend: Any | None = None
) -> SafePandaAdapter:
    """Create an adapter without assuming a particular registered env id."""

    backend = gym_backend or _load_gym_backend()
    registration_errors: list[str] = []
    for package in ("safe_panda_gym", "panda_gym"):
        try:
            importlib.import_module(package)
        except ImportError as exc:
            registration_errors.append(f"{package}: {exc}")
    try:
        env = backend.make(config.env_id, **dict(config.make_kwargs))
    except Exception as exc:
        registrations = "; ".join(registration_errors) or "registration packages imported"
        raise SafePandaDependencyError(
            f"Could not create Safe Panda environment {config.env_id!r}. "
            "Check the env_id and install a compatible Safe-panda-gym or panda-gym "
            f"release. Registration status: {registrations}. Original error: {exc}"
        ) from exc
    return SafePandaAdapter(env, config)
