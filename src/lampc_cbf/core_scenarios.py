"""Three-core-family Safe Panda scenario sampling, preflight, and execution.

Stage 0 of ``docs/SAFE_PANDA_CORE_SCENARIO_PLAN.md``: a deterministic
Latin-hypercube generator over the machine-readable plan
(``configs/safe_panda_core_scenarios_150_plan.json``), the non-negotiable
preflight geometry gates, one-shot instance freezing with hashing, and a
resumable per-row-checkpointed runner on the smooth-dynamic interface.

Reconstruction choices not pinned by the plan (recorded in the preflight
summary): the goal-inflation gate is evaluated at episode start with the
0.05 m goal tolerance; the encounter gate measures the obstacle path's
closest approach to the straight start-goal segment on the 0.04 s episode
time grid; runner kwargs mirror the frozen 8-D double-integrator profile
(``table4_scenarios.scenario_runner_kwargs``) where the plan is silent.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Mapping, Sequence

import numpy as np

PLAN_PATH = Path("configs/safe_panda_core_scenarios_150_plan.json")
INSTANCES_PATH = Path("configs/safe_panda_core_scenarios_150_instances.json")
GOAL_TOLERANCE_M = 0.05
PILOT_EPISODE_INDICES = (4, 14, 24, 34, 44)


class PreflightError(RuntimeError):
    """A preflight gate failed in a way that must stop the pipeline."""


@dataclass(frozen=True, slots=True)
class PreflightRecord:
    initial_true_clearance_m: float
    goal_to_obstacle_start_m: float
    path_closest_approach_m: float
    encounter_margin_m: float
    attempts: int


@dataclass(frozen=True, slots=True)
class ScenarioInstance:
    scenario_id: str
    family_index: int
    episode_index: int
    episode_id: str
    geometry_seed: int
    measurement_seed: int
    crossing_side: int | None
    parameters: dict[str, float]
    goal_offset: tuple[float, float, float]
    obstacle_start_offset: tuple[float, float, float]
    obstacle_velocity: tuple[float, float, float]
    obstacle_radius_m: float
    measurement_noise_sigma_m: float
    feedback_intervention_time_s: float
    unit_point: tuple[float, ...]
    preflight: PreflightRecord


def load_plan(path: str | Path = PLAN_PATH) -> dict[str, Any]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("profile") != "safe_panda_core_scenarios_150_plan":
        raise ValueError("unexpected plan profile")
    return plan


def _uniform_specs(family: Mapping[str, Any]) -> list[tuple[str, float, float]]:
    """Uniform perturbation names with bounds, in stable plan order."""

    specs: list[tuple[str, float, float]] = []
    for name, spec in family["perturbations"].items():
        if spec["distribution"] == "uniform":
            specs.append((name, float(spec["low"]), float(spec["high"])))
    return specs


def _side_field(family: Mapping[str, Any]) -> str | None:
    for name, spec in family["perturbations"].items():
        if spec["distribution"] == "balanced_categorical":
            if tuple(spec["values"]) != (-1, 1):
                raise ValueError("balanced categorical must be (-1, 1)")
            return name
    return None


def latin_hypercube(count: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic LHC in [0, 1): one stratified permutation per dimension."""

    points = np.empty((count, dims), dtype=float)
    for dim in range(dims):
        strata = rng.permutation(count)
        points[:, dim] = (strata + rng.random(count)) / count
    return points


def _resolve_parameters(
    specs: Sequence[tuple[str, float, float]], unit_point: Sequence[float]
) -> dict[str, float]:
    return {
        name: low + (high - low) * float(u)
        for (name, low, high), u in zip(specs, unit_point)
    }


def _geometry(
    family_id: str,
    params: Mapping[str, float],
    side: int | None,
    ee_radius: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Map sampled parameters to goal, obstacle start, and obstacle velocity."""

    if family_id == "CS1_HEAD_ON_CLOSURE":
        goal = (params["goal_x_m"], params["goal_y_m"], params["goal_z_m"])
        start = (
            params["obstacle_start_x_m"],
            params["obstacle_start_y_m"],
            params["obstacle_start_z_m"],
        )
        velocity = (
            params["obstacle_velocity_x_mps"],
            params["obstacle_velocity_y_mps"],
            params["obstacle_velocity_z_mps"],
        )
    elif family_id == "CS2_ORTHOGONAL_3D_CROSSING":
        assert side is not None
        goal = (params["goal_x_m"], params["goal_y_m"], params["goal_z_m"])
        start = (
            side * params["obstacle_start_abs_x_m"],
            params["obstacle_start_y_m"],
            params["obstacle_start_z_m"],
        )
        velocity = (
            -side * params["obstacle_velocity_abs_x_mps"],
            params["obstacle_velocity_y_mps"],
            params["obstacle_velocity_z_mps"],
        )
    elif family_id == "CS3_GRAZING_NEAR_LIMIT":
        assert side is not None
        goal = (
            side * params["goal_abs_x_m"],
            params["goal_y_m"],
            params["goal_z_m"],
        )
        start = (
            side
            * (params["obstacle_radius_m"] + ee_radius + params["grazing_margin_m"]),
            params["obstacle_start_y_m"],
            params["obstacle_start_z_m"],
        )
        velocity = (
            params["obstacle_velocity_x_mps"],
            params["obstacle_velocity_y_mps"],
            params["obstacle_velocity_z_mps"],
        )
    else:
        raise ValueError(f"unknown scenario family {family_id}")
    return goal, start, velocity


def _segment_closest_approach(
    goal: Sequence[float],
    start: Sequence[float],
    velocity: Sequence[float],
    *,
    dt: float,
    steps: int,
) -> float:
    """Min distance from the obstacle path to the start-goal segment."""

    origin = np.zeros(3)
    segment = np.asarray(goal, dtype=float) - origin
    seg_norm_sq = float(segment @ segment)
    times = np.arange(steps + 1) * dt
    centers = np.asarray(start, dtype=float) + np.outer(times, velocity)
    if seg_norm_sq == 0.0:
        return float(np.min(np.linalg.norm(centers, axis=1)))
    fractions = np.clip((centers @ segment) / seg_norm_sq, 0.0, 1.0)
    nearest = np.outer(fractions, segment)
    return float(np.min(np.linalg.norm(centers - nearest, axis=1)))


def check_preflight(
    plan: Mapping[str, Any],
    goal: Sequence[float],
    start: Sequence[float],
    velocity: Sequence[float],
    obstacle_radius: float,
    *,
    attempts: int,
) -> PreflightRecord | None:
    """Return a record when every gate passes, else None."""

    gates = plan["preflight_constraints"]
    runtime = plan["runtime"]
    ee_radius = float(runtime["ee_collision_radius_m"])
    values = (*goal, *start, *velocity, obstacle_radius)
    if not all(np.isfinite(values)):
        return None
    bounds = gates["workspace_bounds_relative_m"]
    for point in (goal, start):
        for axis, coordinate in zip(("x", "y", "z"), point):
            low, high = bounds[axis]
            if not low <= coordinate <= high:
                return None
    combined_radius = obstacle_radius + ee_radius
    initial_clearance = float(np.linalg.norm(start)) - combined_radius
    if initial_clearance < float(gates["initial_true_clearance_min_m"]):
        return None
    goal_to_start = float(np.linalg.norm(np.subtract(goal, start)))
    if goal_to_start <= combined_radius + GOAL_TOLERANCE_M:
        return None
    closest = _segment_closest_approach(
        goal,
        start,
        velocity,
        dt=float(runtime["control_period_s"]),
        steps=int(runtime["maximum_steps"]),
    )
    encounter_margin = closest - combined_radius
    if encounter_margin > float(gates["predicted_closest_approach_upper_m"]):
        return None
    return PreflightRecord(
        initial_true_clearance_m=initial_clearance,
        goal_to_obstacle_start_m=goal_to_start,
        path_closest_approach_m=closest,
        encounter_margin_m=encounter_margin,
        attempts=attempts,
    )


def generate_instances(plan: Mapping[str, Any]) -> list[ScenarioInstance]:
    """Deterministically resolve all 150 episodes, or fail preflight."""

    sampling = plan["sampling"]
    runtime = plan["runtime"]
    seed_base = int(sampling["seed_base"])
    stride = int(sampling["scenario_seed_stride"])
    measurement_offset = int(sampling["measurement_seed_offset"])
    episodes = int(sampling["episodes_per_scenario"])
    max_attempts = int(
        plan["preflight_constraints"]["maximum_sampling_attempts_per_episode"]
    )
    ee_radius = float(runtime["ee_collision_radius_m"])

    instances: list[ScenarioInstance] = []
    for family_index, family in enumerate(plan["scenario_families"]):
        family_id = family["id"]
        specs = _uniform_specs(family)
        side_field = _side_field(family)
        family_seed = seed_base + family_index * stride
        hypercube = latin_hypercube(
            episodes, len(specs), np.random.default_rng(family_seed)
        )
        for episode_index in range(episodes):
            geometry_seed = family_seed + episode_index
            side = None
            if side_field is not None:
                side = -1 if episode_index % 2 == 0 else 1
            resample_rng = np.random.default_rng(geometry_seed)
            record = None
            unit = hypercube[episode_index]
            for attempt in range(max_attempts):
                if attempt > 0:
                    unit = resample_rng.random(len(specs))
                params = _resolve_parameters(specs, unit)
                goal, start, velocity = _geometry(family_id, params, side, ee_radius)
                record = check_preflight(
                    plan,
                    goal,
                    start,
                    velocity,
                    params["obstacle_radius_m"],
                    attempts=attempt + 1,
                )
                if record is not None:
                    break
            if record is None:
                raise PreflightError(
                    f"{family_id} episode {episode_index}: preflight exhausted "
                    f"after {max_attempts} attempts"
                )
            instances.append(
                ScenarioInstance(
                    scenario_id=family_id,
                    family_index=family_index,
                    episode_index=episode_index,
                    episode_id=f"CS{family_index + 1}-E{episode_index:02d}",
                    geometry_seed=geometry_seed,
                    measurement_seed=geometry_seed + measurement_offset,
                    crossing_side=side,
                    parameters={key: float(value) for key, value in params.items()},
                    goal_offset=goal,
                    obstacle_start_offset=start,
                    obstacle_velocity=velocity,
                    obstacle_radius_m=float(params["obstacle_radius_m"]),
                    measurement_noise_sigma_m=float(
                        params["measurement_noise_sigma_m"]
                    ),
                    feedback_intervention_time_s=float(
                        params["feedback_intervention_time_s"]
                    ),
                    unit_point=tuple(float(value) for value in unit),
                    preflight=record,
                )
            )
    _check_integrity(plan, instances)
    return instances


def _check_integrity(
    plan: Mapping[str, Any], instances: Sequence[ScenarioInstance]
) -> None:
    expected = int(plan["sampling"]["total_episodes_per_method"])
    if len(instances) != expected:
        raise PreflightError(f"expected {expected} instances, got {len(instances)}")
    ids = [instance.episode_id for instance in instances]
    seeds = [instance.geometry_seed for instance in instances]
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise PreflightError("episode ids and seeds must be unique")
    for family_index, family in enumerate(plan["scenario_families"]):
        members = [i for i in instances if i.family_index == family_index]
        if len(members) != int(plan["sampling"]["episodes_per_scenario"]):
            raise PreflightError(f"{family['id']}: wrong episode count")
        if _side_field(family) is not None:
            sides = [i.crossing_side for i in members]
            if sides.count(-1) != sides.count(1):
                raise PreflightError(f"{family['id']}: unbalanced sides")


def instances_hash(instances: Sequence[ScenarioInstance]) -> str:
    canonical = json.dumps(
        [asdict(instance) for instance in instances], sort_keys=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def instances_payload(
    instances: Sequence[ScenarioInstance], *, plan_path: str | Path = PLAN_PATH
) -> dict[str, Any]:
    plan_bytes = Path(plan_path).read_bytes()
    return {
        "schema_version": 1,
        "plan_profile": "safe_panda_core_scenarios_150_plan",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "instances_sha256": instances_hash(instances),
        "instances": [asdict(instance) for instance in instances],
    }


def write_frozen_instances(
    instances: Sequence[ScenarioInstance],
    *,
    path: str | Path = INSTANCES_PATH,
    plan_path: str | Path = PLAN_PATH,
) -> dict[str, Any]:
    """Write the resolved instances exactly once; refuse silent divergence."""

    payload = instances_payload(instances, plan_path=plan_path)
    target = Path(path)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("instances_sha256") != payload["instances_sha256"]:
            raise PreflightError(
                "frozen instance file exists with a different hash; a range "
                "change requires plan v2 and a new instance path"
            )
        return existing
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_frozen_instances(
    path: str | Path = INSTANCES_PATH,
) -> tuple[dict[str, Any], list[ScenarioInstance]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    instances = [
        ScenarioInstance(
            **{
                **raw,
                "parameters": dict(raw["parameters"]),
                "goal_offset": tuple(raw["goal_offset"]),
                "obstacle_start_offset": tuple(raw["obstacle_start_offset"]),
                "obstacle_velocity": tuple(raw["obstacle_velocity"]),
                "unit_point": tuple(raw["unit_point"]),
                "preflight": PreflightRecord(**raw["preflight"]),
            }
        )
        for raw in payload["instances"]
    ]
    if instances_hash(instances) != payload["instances_sha256"]:
        raise PreflightError("frozen instance file failed its own hash check")
    return payload, instances


def smoke_episode_ids(instances: Sequence[ScenarioInstance]) -> list[str]:
    """One median condition per family: unit point closest to (0.5, ...)."""

    chosen: list[str] = []
    for family_index in sorted({i.family_index for i in instances}):
        members = [i for i in instances if i.family_index == family_index]
        best = min(
            members,
            key=lambda i: float(
                np.linalg.norm(np.asarray(i.unit_point) - 0.5)
            ),
        )
        chosen.append(best.episode_id)
    return chosen


def pilot_episode_ids(instances: Sequence[ScenarioInstance]) -> list[str]:
    return [
        instance.episode_id
        for instance in instances
        if instance.episode_index in PILOT_EPISODE_INDICES
    ]


# Frozen-contract keys no controller profile may override (plan runtime block).
FROZEN_RUNNER_KEYS = frozenset(
    {
        "seed",
        "gamma",
        "max_steps",
        "sensor_period",
        "measurement_noise_sigma",
        "measurement_noise_mode",
        "goal_offset",
        "obstacle_start_offset",
        "obstacle_velocity",
        "obstacle_radius",
        "collision_radius",
        "reference_mode",
        "control_deadline",
    }
)


def load_controller_profile_manifest(
    path: str | Path, *, instances_sha256: str
) -> dict[str, Any]:
    """Load a versioned remedy-profile manifest with override validation."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("provider_requests", 1) != 0:
        raise ValueError("controller profiles must declare zero provider requests")
    if manifest["instances_sha256"] != instances_sha256:
        raise PreflightError(
            "controller profile manifest is pinned to a different frozen instance set"
        )
    overrides = manifest["runner_kwargs_overrides"]
    frozen_hits = sorted(set(overrides) & FROZEN_RUNNER_KEYS)
    if frozen_hits:
        raise ValueError(
            "controller profile may not override frozen-contract keys: "
            + ", ".join(frozen_hits)
        )
    return manifest


def load_llm_feedback_manifest(
    path: str | Path, *, instances_sha256: str
) -> dict[str, Any]:
    """Load the predeclared LLM-feedback arm manifest with pin validation."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest["instances_sha256"] != instances_sha256:
        raise PreflightError(
            "LLM feedback manifest is pinned to a different frozen instance set"
        )
    overrides = manifest["controller_base"]["runner_kwargs_overrides"]
    frozen_hits = sorted(set(overrides) & FROZEN_RUNNER_KEYS)
    if frozen_hits:
        raise ValueError(
            "LLM manifest may not override frozen-contract keys: "
            + ", ".join(frozen_hits)
        )
    if int(manifest["provider"]["request_budget_max"]) < 1:
        raise ValueError("request budget must be positive")
    return manifest


def load_scripted_feedback_manifest(
    path: str | Path, *, instances_sha256: str
) -> dict[str, Any]:
    """Load and validate a deterministic scripted-feedback profile."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("provider_requests", 1) != 0:
        raise ValueError("scripted feedback manifests must declare zero provider requests")
    if manifest["instances_sha256"] != instances_sha256:
        raise PreflightError(
            "scripted feedback manifest is pinned to a different frozen instance set"
        )
    script = manifest["script"]
    if not script:
        raise ValueError("script must contain at least one event")
    offsets = [float(event["offset_s"]) for event in script]
    if offsets != sorted(offsets) or offsets[0] < 0.0:
        raise ValueError("script offsets must be sorted and non-negative")
    for event in script:
        if not 0.0 < float(event["gamma"]) <= 0.15:
            raise ValueError("scripted gamma must be in (0, 0.15]")
    return manifest


def scripted_gamma_schedule(
    instance: ScenarioInstance, manifest: Mapping[str, Any]
) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
    """Hard-coded (time, gamma) events for one episode, plus request times."""

    latency = float(manifest.get("response_latency_s", 0.0))
    base = instance.feedback_intervention_time_s
    schedule = tuple(
        (base + float(event["offset_s"]) + latency, float(event["gamma"]))
        for event in manifest["script"]
    )
    request_times = tuple(base + float(event["offset_s"]) for event in manifest["script"])
    return schedule, request_times


def load_prediction_feedback_manifest(
    path: str | Path, *, instances_sha256: str
) -> dict[str, Any]:
    """Load and validate a deterministic channel-2 (prediction-switch)
    scripted-feedback profile: language triggers an obstacle PREDICTION
    mode switch instead of a gamma change."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("provider_requests", 1) != 0:
        raise ValueError(
            "prediction feedback manifests must declare zero provider requests"
        )
    if manifest["instances_sha256"] != instances_sha256:
        raise PreflightError(
            "prediction feedback manifest is pinned to a different frozen instance set"
        )
    script = manifest["script"]
    if not script:
        raise ValueError("script must contain at least one event")
    offsets = [float(event["offset_s"]) for event in script]
    if offsets != sorted(offsets) or offsets[0] < 0.0:
        raise ValueError("script offsets must be sorted and non-negative")
    for event in script:
        if event["mode"] not in {"static", "velocity", "velocity_tube"}:
            raise ValueError("scripted prediction mode must be a valid mode")
    return manifest


def scripted_prediction_schedule(
    instance: ScenarioInstance, manifest: Mapping[str, Any]
) -> tuple[tuple[tuple[float, str], ...], tuple[float, ...]]:
    """Hard-coded (time, mode) events for one episode, plus request times."""

    latency = float(manifest.get("response_latency_s", 0.0))
    base = instance.feedback_intervention_time_s
    schedule = tuple(
        (base + float(event["offset_s"]) + latency, str(event["mode"]))
        for event in manifest["script"]
    )
    request_times = tuple(
        base + float(event["offset_s"]) for event in manifest["script"]
    )
    return schedule, request_times


def runner_kwargs(
    instance: ScenarioInstance, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Frozen-contract SmoothDynamicConfig kwargs for one episode."""

    runtime = plan["runtime"]
    return {
        "seed": instance.measurement_seed,
        "gamma": float(runtime["fixed_gamma"]),
        "max_steps": int(runtime["maximum_steps"]),
        "sensor_period": float(runtime["sensing_period_s"]),
        "measurement_noise_sigma": instance.measurement_noise_sigma_m,
        "measurement_noise_mode": "gaussian",
        "reference_speed": 0.08,
        "goal_offset": instance.goal_offset,
        "obstacle_start_offset": instance.obstacle_start_offset,
        "obstacle_velocity": instance.obstacle_velocity,
        "obstacle_radius": instance.obstacle_radius_m,
        "collision_radius": float(runtime["ee_collision_radius_m"]),
        "prediction_mode": "static",
        "known_obstacle_velocity": False,
        "safety_reflex_enabled": False,
        "formal_safety_filter_enabled": False,
        "optimal_decay_weight": 0.0,
        "reference_mode": str(runtime["reference_mode"]),
        "safety_mode": "cbf",
        "cbf_transition_mode": str(runtime["cbf_transition_mode"]),
        "solver_max_constraint_violation": 1e-6,
        "solver_max_cpu_time": 0.035,
        "control_deadline": float(runtime["control_period_s"]),
        "reject_deadline_miss": False,
        "save_plots": False,
        "save_metrics": False,
        "save_animation": False,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    spread = (
        z
        * ((proportion * (1 - proportion) + z**2 / (4 * total)) / total) ** 0.5
        / denominator
    )
    return (max(0.0, center - spread), min(1.0, center + spread))


def bootstrap_interval(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = mean,
    resamples: int = 10000,
    seed: int = 20260716,
) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    stats = np.asarray(
        [
            statistic(data[rng.integers(0, len(data), len(data))].tolist())
            for _ in range(resamples)
        ]
    )
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


EPISODE_ROW_FIELDS = (
    "episode_id",
    "scenario_id",
    "episode_index",
    "crossing_side",
    "geometry_seed",
    "measurement_seed",
    "method",
    "goal_contract",
    "outcome",
    "joint_success",
    "reached_goal",
    "collision",
    "steps",
    "minimum_true_clearance_m",
    "minimum_observed_clearance_m",
    "minimum_true_cbf_residual",
    "true_cbf_violation_steps",
    "steps_to_goal",
    "final_goal_distance_m",
    "solver_failures",
    "solver_rejections",
    "deadline_misses",
    "emergency_fallbacks",
    "path_length_m",
    "control_effort_acceleration_rms",
    "mean_model_transition_error_m",
    "max_model_transition_error_m",
    "error",
)


def _episode_row(
    instance: ScenarioInstance, method: str, result: Any, error: str | None
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "episode_id": instance.episode_id,
        "scenario_id": instance.scenario_id,
        "episode_index": instance.episode_index,
        "crossing_side": instance.crossing_side,
        "geometry_seed": instance.geometry_seed,
        "measurement_seed": instance.measurement_seed,
        "method": method,
        "goal_contract": "ee_reach_only",
        "error": error or "",
    }
    if result is None:
        row.update({field: "" for field in EPISODE_ROW_FIELDS if field not in row})
        row["outcome"] = "crashed"
        row["joint_success"] = False
        return row
    joint = bool(
        result.outcome == "goal" and result.reached_goal and not result.collision
    )
    row.update(
        {
            "outcome": result.outcome,
            "joint_success": joint,
            "reached_goal": result.reached_goal,
            "collision": result.collision,
            "steps": result.steps,
            "minimum_true_clearance_m": result.minimum_true_clearance,
            "minimum_observed_clearance_m": result.minimum_measured_clearance,
            "minimum_true_cbf_residual": result.minimum_true_cbf_residual,
            "true_cbf_violation_steps": result.true_cbf_violation_steps,
            "steps_to_goal": result.steps if joint else "",
            "final_goal_distance_m": result.final_goal_distance,
            "solver_failures": result.solver_failures,
            "solver_rejections": result.solver_rejections,
            "deadline_misses": result.deadline_misses,
            "emergency_fallbacks": result.emergency_fallbacks,
            "path_length_m": result.smoothness.path_length,
            "control_effort_acceleration_rms": result.smoothness.acceleration_rms,
            "mean_model_transition_error_m": result.mean_model_transition_error,
            "max_model_transition_error_m": result.max_model_transition_error,
        }
    )
    return row


def _family_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["outcome"] != "crashed"]
    successes = sum(bool(row["joint_success"]) for row in completed)
    collisions = sum(bool(row["collision"]) for row in completed)
    clearances = [
        float(row["minimum_true_clearance_m"])
        for row in completed
        if row["minimum_true_clearance_m"] != ""
    ]
    return {
        "rows": len(rows),
        "crashed": len(rows) - len(completed),
        "successes": successes,
        "success_rate": successes / len(completed) if completed else None,
        "success_wilson_95": wilson_interval(successes, len(completed)),
        "collisions": collisions,
        "collision_wilson_95": wilson_interval(collisions, len(completed)),
        "min_true_clearance_mean_m": mean(clearances) if clearances else None,
        "min_true_clearance_median_m": median(clearances) if clearances else None,
        "min_true_clearance_mean_bootstrap_95": bootstrap_interval(clearances),
        "min_true_clearance_median_bootstrap_95": bootstrap_interval(
            clearances, statistic=median
        ),
        "solver_failure_episodes": sum(
            int(row["solver_failures"] or 0) > 0 for row in completed
        ),
        "solver_failures_total": sum(int(row["solver_failures"] or 0) for row in completed),
        "solver_rejections_total": sum(
            int(row["solver_rejections"] or 0) for row in completed
        ),
        "deadline_misses_total": sum(
            int(row["deadline_misses"] or 0) for row in completed
        ),
        "emergency_fallbacks_total": sum(
            int(row["emergency_fallbacks"] or 0) for row in completed
        ),
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]], *, instances_sha256: str, method: str
) -> dict[str, Any]:
    families = sorted({row["scenario_id"] for row in rows})
    per_family = {
        family: _family_summary([row for row in rows if row["scenario_id"] == family])
        for family in families
    }
    rates = [
        summary["success_rate"]
        for summary in per_family.values()
        if summary["success_rate"] is not None
    ]
    return {
        "profile": "safe_panda_core_scenarios_150",
        "method": method,
        "goal_contract": "ee_reach_only",
        "whole_arm_collision_certificate": False,
        "instances_sha256": instances_sha256,
        "rows": len(rows),
        "families": per_family,
        "macro_success_rate": mean(rates) if rates else None,
        "worst_family_success_rate": min(rates) if rates else None,
    }


def run_core_benchmark(
    *,
    instances_path: str | Path = INSTANCES_PATH,
    plan_path: str | Path = PLAN_PATH,
    output_dir: str | Path = "artifacts/safe_panda_core_scenarios_150",
    episode_ids: Sequence[str] | None = None,
    method: str = "safe_panda_core_fixed_g015",
    episode_runner: Callable[..., Any] | None = None,
    scripted_feedback_manifest: str | Path | None = None,
    controller_profile_manifest: str | Path | None = None,
    llm_feedback_manifest: str | Path | None = None,
    llm_mapper: Any | None = None,
    prediction_feedback_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Resumable frozen-instance execution with a checkpoint after every row."""

    if episode_runner is None:
        from .smooth_dynamic_demo import SmoothDynamicConfig, run_smooth_dynamic_demo

        def episode_runner(**kwargs: Any) -> Any:
            return run_smooth_dynamic_demo(SmoothDynamicConfig(**kwargs))

    plan = load_plan(plan_path)
    payload, instances = load_frozen_instances(instances_path)
    feedback_manifest = None
    if scripted_feedback_manifest is not None:
        feedback_manifest = load_scripted_feedback_manifest(
            scripted_feedback_manifest,
            instances_sha256=payload["instances_sha256"],
        )
        method = str(feedback_manifest["method"])
    prediction_manifest = None
    if prediction_feedback_manifest is not None:
        prediction_manifest = load_prediction_feedback_manifest(
            prediction_feedback_manifest,
            instances_sha256=payload["instances_sha256"],
        )
        method = str(prediction_manifest["method"])
    profile_manifest = None
    if controller_profile_manifest is not None:
        profile_manifest = load_controller_profile_manifest(
            controller_profile_manifest,
            instances_sha256=payload["instances_sha256"],
        )
        method = str(profile_manifest["method"])
    llm_manifest = None
    llm_requests_used = 0
    if llm_feedback_manifest is not None:
        llm_manifest = load_llm_feedback_manifest(
            llm_feedback_manifest,
            instances_sha256=payload["instances_sha256"],
        )
        method = str(llm_manifest["method"])
        if llm_mapper is None:
            from .paper_continuous_gamma import (
                NIMContinuousGammaConfig,
                NIMContinuousGammaMapper,
            )

            checkpoint_dir = Path(llm_manifest["provider"]["checkpoint_dir"])
            llm_mapper = NIMContinuousGammaMapper(
                NIMContinuousGammaConfig(
                    model=str(llm_manifest["provider"]["model"]),
                    checkpoint_path=str(checkpoint_dir / "decisions.jsonl"),
                )
            )
    selected = [
        instance
        for instance in instances
        if episode_ids is None or instance.episode_id in set(episode_ids)
    ]
    if episode_ids is not None and len(selected) != len(set(episode_ids)):
        raise ValueError("episode_ids contains unknown episode ids")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "run_checkpoint.json"
    episodes_path = root / "episodes.csv"
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("instances_sha256") != payload["instances_sha256"]:
            raise PreflightError(
                "checkpoint belongs to a different frozen instance set"
            )
        with episodes_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                completed[row["episode_id"]] = row
    else:
        episodes_path.write_text("", encoding="utf-8")
        with episodes_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(EPISODE_ROW_FIELDS)).writeheader()

    for instance in selected:
        if instance.episode_id in completed:
            continue
        kwargs = runner_kwargs(instance, plan)
        kwargs["output_dir"] = str(root / f"episode_{instance.episode_id}")
        if feedback_manifest is not None:
            schedule, request_times = scripted_gamma_schedule(
                instance, feedback_manifest
            )
            kwargs["gamma_schedule"] = schedule
            kwargs["gamma_schedule_request_times"] = request_times
            kwargs["gamma_update_ttl"] = float(
                feedback_manifest["gamma_update_ttl_s"]
            )
        if prediction_manifest is not None:
            schedule, request_times = scripted_prediction_schedule(
                instance, prediction_manifest
            )
            kwargs["prediction_mode_schedule"] = schedule
            kwargs["prediction_mode_schedule_request_times"] = request_times
        if profile_manifest is not None:
            kwargs.update(profile_manifest["runner_kwargs_overrides"])
        if llm_manifest is not None:
            protocol = llm_manifest["feedback_protocol"]
            decision = llm_mapper.infer(
                str(protocol["user_utterance"]),
                current_gamma=kwargs["gamma"],
                task="Move gripper to the reach target.",
            )
            if not decision.cache_hit:
                llm_requests_used += 1
                if llm_requests_used > int(
                    llm_manifest["provider"]["request_budget_max"]
                ):
                    raise RuntimeError("LLM request budget exhausted")
            kwargs.update(llm_manifest["controller_base"]["runner_kwargs_overrides"])
            kwargs["gamma_range_mode"] = str(protocol["gamma_range_mode"])
            if not decision.fallback_used:
                apply_time = (
                    instance.feedback_intervention_time_s
                    + decision.latency_seconds
                )
                kwargs["gamma_schedule"] = ((apply_time, decision.gamma),)
                kwargs["gamma_schedule_request_times"] = (
                    instance.feedback_intervention_time_s,
                )
                kwargs["gamma_update_ttl"] = float(protocol["gamma_update_ttl_s"])
            with (root / "llm_decisions.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    json.dumps(
                        {
                            "episode_id": instance.episode_id,
                            "gamma": decision.gamma,
                            "table2_label": decision.table2_label,
                            "latency_seconds": decision.latency_seconds,
                            "cache_hit": decision.cache_hit,
                            "fallback_used": decision.fallback_used,
                            "error": decision.error,
                        }
                    )
                    + "\n"
                )
        result, error = None, None
        try:
            result = episode_runner(**kwargs)
        except Exception as exc:  # crashed episodes are recorded, never retried
            error = f"{type(exc).__name__}: {exc}"
        row = _episode_row(instance, method, result, error)
        with episodes_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(EPISODE_ROW_FIELDS)).writerow(row)
        completed[instance.episode_id] = {
            key: row.get(key, "") for key in EPISODE_ROW_FIELDS
        }
        checkpoint_path.write_text(
            json.dumps(
                {
                    "instances_sha256": payload["instances_sha256"],
                    "method": method,
                    "completed_rows": len(completed),
                    "last_episode_id": instance.episode_id,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    rows = [completed[i.episode_id] for i in selected if i.episode_id in completed]
    normalized = [_normalize_csv_row(row) for row in rows]
    summary = summarize_rows(
        normalized, instances_sha256=payload["instances_sha256"], method=method
    )
    summary["selected_episodes"] = len(selected)
    (root / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def mcnemar_exact_pvalue(
    baseline: Sequence[bool], method: Sequence[bool]
) -> float:
    """Exact two-sided McNemar test on paired binary outcomes."""

    baseline_only = sum(a and not b for a, b in zip(baseline, method))
    method_only = sum(b and not a for a, b in zip(baseline, method))
    discordant = baseline_only + method_only
    if discordant == 0:
        return 1.0
    tail = sum(
        comb(discordant, i) for i in range(min(baseline_only, method_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, keyed like the input."""

    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, pvalue) in enumerate(ordered):
        running = max(running, (total - rank) * pvalue)
        adjusted[key] = min(1.0, running)
    return adjusted


def paired_comparison(
    baseline_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Family-labeled paired analysis on identical resolved instances."""

    baseline_by_id = {row["episode_id"]: row for row in baseline_rows}
    method_by_id = {row["episode_id"]: row for row in method_rows}
    shared = sorted(set(baseline_by_id) & set(method_by_id))
    if len(shared) != len(baseline_by_id) or len(shared) != len(method_by_id):
        raise ValueError("paired comparison requires identical episode sets")
    families = sorted({baseline_by_id[i]["scenario_id"] for i in shared})
    per_family: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for family in families:
        ids = [i for i in shared if baseline_by_id[i]["scenario_id"] == family]
        base = [bool(baseline_by_id[i]["joint_success"]) for i in ids]
        meth = [bool(method_by_id[i]["joint_success"]) for i in ids]
        pvalue = mcnemar_exact_pvalue(base, meth)
        raw_pvalues[family] = pvalue
        per_family[family] = {
            "episodes": len(ids),
            "baseline_successes": sum(base),
            "method_successes": sum(meth),
            "success_difference": (sum(meth) - sum(base)) / len(ids),
            "baseline_only_wins": sum(a and not b for a, b in zip(base, meth)),
            "method_only_wins": sum(b and not a for a, b in zip(base, meth)),
            "mcnemar_exact_p": pvalue,
        }
    adjusted = holm_adjust(raw_pvalues)
    for family in families:
        per_family[family]["holm_adjusted_p"] = adjusted[family]
    diffs = [per_family[family]["success_difference"] for family in families]
    return {
        "families": per_family,
        "macro_success_difference": mean(diffs) if diffs else None,
        "worst_family_success_difference": min(diffs) if diffs else None,
    }


def _normalize_csv_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """CSV round-trips as strings; restore the types summaries rely on."""

    normalized = dict(row)
    for key in ("joint_success", "reached_goal", "collision"):
        normalized[key] = str(row.get(key, "")).lower() == "true"
    for key in (
        "solver_failures",
        "solver_rejections",
        "deadline_misses",
        "emergency_fallbacks",
    ):
        value = row.get(key, "")
        normalized[key] = int(float(value)) if value not in ("", None) else 0
    value = row.get("minimum_true_clearance_m", "")
    normalized["minimum_true_clearance_m"] = (
        float(value) if value not in ("", None) else ""
    )
    return normalized
