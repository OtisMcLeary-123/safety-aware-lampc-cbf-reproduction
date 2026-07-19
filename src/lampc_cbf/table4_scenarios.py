"""Validation and provenance helpers for the deterministic Table-4 scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence


SCENARIO_KEYS = frozenset(
    {
        "episode_id",
        "category",
        "obstacle_start_pos",
        "obstacle_velocity",
        "intervention_time",
        "noise_sigma",
    }
)
CATEGORY_COUNTS = {
    "Head-on Collision": 15,
    "Orthogonal Crossing": 15,
    "Boundary & Kinematic Limits": 10,
    "High Unbounded Noise": 10,
}
CONTROL_PERIOD_SECONDS = 0.04
SENSOR_PERIOD_SECONDS = 0.67
MAX_EE_SPEED_MPS = 0.20
EE_COLLISION_RADIUS_M = 0.05
MIN_OBSTACLE_SPEED_MPS = 0.0125
MAX_OBSTACLE_SPEED_MPS = 0.10


def _vector(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a length-3 array")
    values = tuple(float(item) for item in value)
    if not all(isfinite(item) for item in values):
        raise ValueError(f"{name} must contain finite numbers")
    return values


def _close(actual: float, expected: float, tolerance: float = 2e-5) -> bool:
    return abs(actual - expected) <= tolerance


@dataclass(frozen=True, slots=True)
class Table4Scenario:
    episode_id: int
    category: str
    obstacle_start_pos: tuple[float, float, float]
    obstacle_velocity: tuple[float, float, float]
    intervention_time: float
    noise_sigma: float

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> "Table4Scenario":
        if set(item) != SCENARIO_KEYS:
            missing = sorted(SCENARIO_KEYS - set(item))
            extra = sorted(set(item) - SCENARIO_KEYS)
            raise ValueError(f"scenario keys mismatch; missing={missing}, extra={extra}")
        return cls(
            episode_id=int(item["episode_id"]),
            category=str(item["category"]),
            obstacle_start_pos=_vector(item["obstacle_start_pos"], "obstacle_start_pos"),
            obstacle_velocity=_vector(item["obstacle_velocity"], "obstacle_velocity"),
            intervention_time=float(item["intervention_time"]),
            noise_sigma=float(item["noise_sigma"]),
        )

    @property
    def speed_mps(self) -> float:
        return sqrt(sum(value * value for value in self.obstacle_velocity))

    @property
    def initial_distance_m(self) -> float:
        return sqrt(sum(value * value for value in self.obstacle_start_pos))

    @property
    def conservative_reaction_distance_m(self) -> float:
        return EE_COLLISION_RADIUS_M + (
            MAX_EE_SPEED_MPS + self.speed_mps
        ) * SENSOR_PERIOD_SECONDS

    @property
    def reaction_margin_m(self) -> float:
        return self.initial_distance_m - self.conservative_reaction_distance_m

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "category": self.category,
            "obstacle_start_pos": list(self.obstacle_start_pos),
            "obstacle_velocity": list(self.obstacle_velocity),
            "intervention_time": self.intervention_time,
            "noise_sigma": self.noise_sigma,
        }


def _validate_category_rules(scenarios: Sequence[Table4Scenario]) -> None:
    by_category: dict[str, list[Table4Scenario]] = {name: [] for name in CATEGORY_COUNTS}
    for scenario in scenarios:
        if scenario.category not in by_category:
            raise ValueError(f"unknown scenario category: {scenario.category}")
        by_category[scenario.category].append(scenario)
    for category, expected_count in CATEGORY_COUNTS.items():
        actual = by_category[category]
        if len(actual) != expected_count:
            raise ValueError(
                f"category {category!r} requires {expected_count} scenarios, got {len(actual)}"
            )

    head_on = by_category["Head-on Collision"]
    for index, scenario in enumerate(head_on):
        expected_speed = MIN_OBSTACLE_SPEED_MPS + index * (
            MAX_OBSTACLE_SPEED_MPS - MIN_OBSTACLE_SPEED_MPS
        ) / 14.0
        if scenario.obstacle_start_pos != (0.0, 0.44, 0.0):
            raise ValueError("head-on scenarios must start at [0, 0.44, 0]")
        if not _close(scenario.speed_mps, expected_speed) or scenario.obstacle_velocity[1] >= 0:
            raise ValueError("head-on scenarios must use the linear speed sweep toward -y")
        if scenario.intervention_time != 0.2 or scenario.noise_sigma != 0.005:
            raise ValueError("head-on intervention/noise values do not match the contract")

    crossing = by_category["Orthogonal Crossing"]
    for index, scenario in enumerate(crossing):
        expected_speed = MIN_OBSTACLE_SPEED_MPS + index * (
            MAX_OBSTACLE_SPEED_MPS - MIN_OBSTACLE_SPEED_MPS
        ) / 14.0
        expected_time = 0.1 + index * 0.2 / 14.0
        x, y, z = scenario.obstacle_start_pos
        vx, vy, vz = scenario.obstacle_velocity
        if not 0.1 <= y <= 0.2 or z != 0.0:
            raise ValueError("crossing scenarios must use y in [0.1, 0.2] and z=0")
        if not _close(scenario.speed_mps, expected_speed) or vy != 0.0 or vz != 0.0:
            raise ValueError("crossing scenarios must use the linear x speed sweep")
        if x * vx >= 0.0 or not _close(scenario.intervention_time, expected_time):
            raise ValueError("crossing trajectories must move toward the y-axis")
        if scenario.noise_sigma != 0.005:
            raise ValueError("crossing noise must be 0.005")

    boundary = by_category["Boundary & Kinematic Limits"]
    for index, scenario in enumerate(boundary):
        expected_speed = 0.09 + index * 0.01 / 9.0
        expected_time = 0.35 + index * 0.05 / 9.0
        x, y, z = scenario.obstacle_start_pos
        if not 0.03 <= abs(x) <= EE_COLLISION_RADIUS_M or y != 0.36 or z != 0.0:
            raise ValueError("boundary scenarios must pass just inside the 0.035 m radius")
        if not _close(scenario.speed_mps, expected_speed) or scenario.obstacle_velocity[1] >= 0:
            raise ValueError("boundary scenarios must use the near-maximum -y sweep")
        if not _close(scenario.intervention_time, expected_time) or scenario.noise_sigma != 0.005:
            raise ValueError("boundary intervention/noise values do not match the contract")

    high_noise = by_category["High Unbounded Noise"]
    for index, scenario in enumerate(high_noise):
        expected_noise = 0.01 + index * 0.04 / 9.0
        if not _close(scenario.noise_sigma, expected_noise):
            raise ValueError("high-noise values must increase from 0.01 to 0.05")
        if not any(
            scenario.obstacle_start_pos == source.obstacle_start_pos
            and scenario.obstacle_velocity == source.obstacle_velocity
            and _close(scenario.intervention_time, source.intervention_time)
            for source in scenarios[:30]
        ):
            raise ValueError("high-noise scenarios must reuse a Group 1 or Group 2 trajectory")


def validate_scenarios(scenarios: Sequence[Table4Scenario]) -> tuple[Table4Scenario, ...]:
    values = tuple(scenarios)
    if len(values) != 50:
        raise ValueError(f"exactly 50 scenarios are required, got {len(values)}")
    if tuple(scenario.episode_id for scenario in values) != tuple(range(1, 51)):
        raise ValueError("episode_id values must be exactly 1 through 50")
    for scenario in values:
        if any(value < -3.0 for value in scenario.obstacle_start_pos):
            raise ValueError(f"episode {scenario.episode_id} violates workspace lower bounds")
        if not 0.0 <= scenario.intervention_time <= 0.4:
            raise ValueError(f"episode {scenario.episode_id} has invalid intervention time")
        if scenario.noise_sigma < 0.0 or not isfinite(scenario.noise_sigma):
            raise ValueError(f"episode {scenario.episode_id} has invalid noise sigma")
        if not MIN_OBSTACLE_SPEED_MPS <= scenario.speed_mps <= MAX_OBSTACLE_SPEED_MPS:
            raise ValueError(f"episode {scenario.episode_id} has invalid obstacle speed")
        if scenario.reaction_margin_m < 0.0:
            raise ValueError(
                f"episode {scenario.episode_id} lacks the conservative sensor reaction margin"
            )
    _validate_category_rules(values)
    return values


def load_scenarios(path: str | Path) -> tuple[Table4Scenario, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("scenario file must contain a JSON array")
    return validate_scenarios(
        [Table4Scenario.from_mapping(item) for item in payload]
    )


def scenario_summary(scenarios: Sequence[Table4Scenario]) -> dict[str, Any]:
    values = validate_scenarios(scenarios)
    return {
        "episodes": len(values),
        "category_counts": {
            category: sum(item.category == category for item in values)
            for category in CATEGORY_COUNTS
        },
        "speed_min_mps": min(item.speed_mps for item in values),
        "speed_max_mps": max(item.speed_mps for item in values),
        "noise_min": min(item.noise_sigma for item in values),
        "noise_max": max(item.noise_sigma for item in values),
        "reaction_margin_min_m": min(item.reaction_margin_m for item in values),
        "coordinate_frame": "EE-relative scenario frame: x=lateral, y=goal-forward, z=up",
        "ee_start_position_m": [0.0, 0.0, 0.0],
        "goal_position_m": [0.0, 0.30, 0.0],
        "ee_collision_radius_m": EE_COLLISION_RADIUS_M,
        "sensor_period_s": SENSOR_PERIOD_SECONDS,
        "control_period_s": CONTROL_PERIOD_SECONDS,
        "table4_exact_reproduction": False,
        "note": "Groups 2-4 form a deterministic stress extension; paper Table 4 does not publish this full scenario grid.",
    }


def scenario_runner_kwargs(
    scenario: Table4Scenario, *, seed_base: int = 20260715
) -> dict[str, Any]:
    """Resolve one scenario into the existing SmoothDynamicConfig interface."""

    return {
        "seed": seed_base + scenario.episode_id - 1,
        "max_steps": 220,
        "sensor_period": SENSOR_PERIOD_SECONDS,
        "measurement_noise_sigma": scenario.noise_sigma,
        "measurement_noise_mode": "gaussian",
        "reference_speed": 0.08,
        "goal_offset": (0.0, 0.30, 0.0),
        "obstacle_start_offset": scenario.obstacle_start_pos,
        "obstacle_velocity": scenario.obstacle_velocity,
        "obstacle_radius": 0.0,
        "collision_radius": EE_COLLISION_RADIUS_M,
        "prediction_mode": "static",
        "known_obstacle_velocity": False,
        "safety_reflex_enabled": False,
        "formal_safety_filter_enabled": False,
        "optimal_decay_weight": 0.0,
        "reference_mode": "direct_target",
        "safety_mode": "cbf",
        "cbf_transition_mode": "command_velocity",
        "gamma_update_ttl": 8.8,
        "solver_max_constraint_violation": 1e-6,
        "solver_max_cpu_time": 0.035,
        "control_deadline": CONTROL_PERIOD_SECONDS,
        "reject_deadline_miss": False,
        "save_plots": False,
        "save_metrics": False,
    }


def contextual_feedback_context(
    scenario: Table4Scenario,
    *,
    current_gamma: float = 0.15,
    reference_speed_mps: float = 0.08,
) -> Any:
    """Estimate the scenario hazard state at its intervention timestamp."""

    from .contextual_gamma import FeedbackHazardContext

    time = scenario.intervention_time
    robot_position = (0.0, reference_speed_mps * time, 0.0)
    obstacle_position = tuple(
        position + velocity * time
        for position, velocity in zip(
            scenario.obstacle_start_pos, scenario.obstacle_velocity
        )
    )
    relative_position = tuple(
        obstacle - robot
        for obstacle, robot in zip(obstacle_position, robot_position)
    )
    distance = sqrt(sum(value * value for value in relative_position))
    clearance = distance - EE_COLLISION_RADIUS_M
    relative_velocity = (
        scenario.obstacle_velocity[0],
        scenario.obstacle_velocity[1] - reference_speed_mps,
        scenario.obstacle_velocity[2],
    )
    closing_speed = (
        -sum(
            position * velocity
            for position, velocity in zip(relative_position, relative_velocity)
        )
        / distance
        if distance > 0.0
        else 0.0
    )
    ttc = max(0.0, clearance) / closing_speed if closing_speed > 0.0 else None
    return FeedbackHazardContext(
        current_gamma=current_gamma,
        obstacle_distance_m=distance,
        combined_radius_m=EE_COLLISION_RADIUS_M,
        clearance_m=clearance,
        predicted_ttc_s=ttc,
        obstacle_speed_mps=scenario.speed_mps,
        minimum_cbf_residual=None,
        intervention_time_s=time,
    )
