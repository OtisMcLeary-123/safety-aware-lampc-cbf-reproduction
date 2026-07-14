"""Physics-budgeted development run for the selected C3BF liveness stack."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import csv
import json
from math import ceil, isfinite, pi, sqrt
import multiprocessing
from pathlib import Path
from typing import Any, Mapping, Sequence

from .collision_cone_liveness_ablation import run_liveness_condition
from .remediation_benchmark import deterministic_scenario_grid


@dataclass(frozen=True, slots=True)
class PhysicsTimeoutInputs:
    control_period: float = 0.04
    goal_distance: float = 0.30
    reference_speed: float = 0.08
    obstacle_radius: float = 0.10
    collision_radius: float = 0.035
    route_margin: float = 0.08
    sensor_period: float = 0.67
    feedback_latency: float = 0.4971896839560941
    safety_recovery_duration: float = 0.80

    def __post_init__(self) -> None:
        values = asdict(self).values()
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("timeout inputs must be finite and non-negative")
        if self.control_period == 0.0 or self.reference_speed == 0.0:
            raise ValueError("control period and reference speed must be positive")


@dataclass(frozen=True, slots=True)
class PhysicsTimeoutBudget:
    direct_path_time: float
    detour_radius: float
    detour_extra_distance: float
    detour_time: float
    sensing_and_recovery_reserve: float
    total_seconds: float
    max_steps: int


def derive_physics_timeout(inputs: PhysicsTimeoutInputs) -> PhysicsTimeoutBudget:
    """Bound one semicircular detour plus sensing and recovery delays."""

    direct_path_time = inputs.goal_distance / inputs.reference_speed
    detour_radius = (
        inputs.obstacle_radius + inputs.collision_radius + inputs.route_margin
    )
    # Replace the straight diameter through the obstacle by a semicircle.
    detour_extra_distance = (pi - 2.0) * detour_radius
    detour_time = detour_extra_distance / inputs.reference_speed
    sensing_and_recovery_reserve = (
        inputs.sensor_period
        + inputs.feedback_latency
        + inputs.safety_recovery_duration
    )
    total_seconds = (
        direct_path_time + detour_time + sensing_and_recovery_reserve
    )
    return PhysicsTimeoutBudget(
        direct_path_time=direct_path_time,
        detour_radius=detour_radius,
        detour_extra_distance=detour_extra_distance,
        detour_time=detour_time,
        sensing_and_recovery_reserve=sensing_and_recovery_reserve,
        total_seconds=total_seconds,
        max_steps=ceil(total_seconds / inputs.control_period),
    )


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson score interval for a binomial rate."""

    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Wilson inputs must satisfy 0 <= successes <= trials")
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (rate + z**2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * sqrt(rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == trials else min(1.0, center + half_width)
    return lower, upper


@dataclass(frozen=True, slots=True)
class LivenessDevelopmentConfig:
    episodes: int = 100
    seed: int = 20260714
    workers: int = 4
    output_dir: str = "artifacts/collision_cone_liveness_development_100"
    prerequisite_summary: str = (
        "artifacts/collision_cone_liveness_ablation_20/summary.json"
    )
    timeout_inputs: PhysicsTimeoutInputs = field(
        default_factory=PhysicsTimeoutInputs
    )
    max_collision_rate: float = 0.05
    max_controller_stall_rate: float = 0.10
    max_solver_rejection_rate: float = 0.01
    max_deadline_miss_rate: float = 0.01
    p99_solve_time_limit: float = 0.04

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.workers < 1:
            raise ValueError("episodes and workers must be positive")
        if self.p99_solve_time_limit <= 0.0:
            raise ValueError("p99_solve_time_limit must be positive")
        for value in (
            self.max_collision_rate,
            self.max_controller_stall_rate,
            self.max_solver_rejection_rate,
            self.max_deadline_miss_rate,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("gate rates must be in [0, 1]")


def validate_liveness_prerequisite(path: str | Path) -> dict[str, Any]:
    prerequisite = Path(path)
    if not prerequisite.is_file():
        raise FileNotFoundError(f"prerequisite summary is missing: {prerequisite}")
    payload = json.loads(prerequisite.read_text(encoding="utf-8"))
    if payload.get("protocol") != "collision-cone-liveness-ablation-v1":
        raise ValueError("prerequisite uses the wrong protocol")
    if payload.get("gate", {}).get("passed") is not True:
        raise ValueError("prerequisite liveness gate did not pass")
    target = payload.get("methods", {}).get(
        "policy_library_tangential_subgoal"
    )
    if not isinstance(target, dict):
        raise ValueError("prerequisite target method is missing")
    return payload


def _run_condition(payload: Mapping[str, Any]) -> dict[str, Any]:
    return run_liveness_condition(
        payload["condition"],
        int(payload["max_steps"]),
        variant_name="selected_c3bf_policy_library_tangential_subgoal",
        barrier_mode="collision_cone",
        side_latch=True,
        policy_library=True,
        tangential_subgoal=True,
    )


def summarize_liveness_development_rows(
    rows: Sequence[Mapping[str, Any]],
    config: LivenessDevelopmentConfig,
    timeout: PhysicsTimeoutBudget,
) -> dict[str, Any]:
    if len(rows) != config.episodes:
        raise ValueError(f"development does not have {config.episodes} episodes")
    total_steps = sum(int(row["steps"]) for row in rows)
    outcomes = {
        outcome: sum(row["outcome"] == outcome for row in rows)
        for outcome in (
            "goal",
            "collision",
            "safety_timeout",
            "controller_stall",
            "solver_failure",
            "environment_truncated",
        )
    }
    collisions = sum(bool(row["collision"]) for row in rows)
    successes = sum(bool(row["success"]) for row in rows)
    stalls = outcomes["controller_stall"]
    collision_rate = collisions / len(rows)
    stall_rate = stalls / len(rows)
    solver_rejection_rate = sum(
        int(row["solver_rejections"]) for row in rows
    ) / total_steps
    deadline_miss_rate = sum(
        int(row["deadline_misses"]) for row in rows
    ) / total_steps
    maximum_p99 = max(float(row["p99_solve_time"]) for row in rows)
    interventions = sum(int(row["reflex_interventions"]) for row in rows)
    backups = sum(int(row["reflex_backups"]) for row in rows)
    robust_recoveries = sum(
        int(row["reflex_robust_recoveries"]) for row in rows
    )
    solver_rejections = sum(int(row["solver_rejections"]) for row in rows)
    deadline_misses = sum(int(row["deadline_misses"]) for row in rows)
    checks = {
        "collision": collision_rate <= config.max_collision_rate,
        "controller_stall": stall_rate <= config.max_controller_stall_rate,
        "feasibility": solver_rejection_rate <= config.max_solver_rejection_rate,
        "timing": (
            deadline_miss_rate <= config.max_deadline_miss_rate
            and maximum_p99 <= config.p99_solve_time_limit
        ),
    }
    return {
        "protocol": "collision-cone-liveness-development-v1",
        "config": asdict(config),
        "physics_timeout": asdict(timeout),
        "method": "selected_c3bf_policy_library_tangential_subgoal",
        "episodes": len(rows),
        "outcomes": outcomes,
        "success_rate": successes / len(rows),
        "success_rate_wilson_95": list(wilson_interval(successes, len(rows))),
        "collision_rate": collision_rate,
        "collision_rate_wilson_95": list(wilson_interval(collisions, len(rows))),
        "controller_stall_rate": stall_rate,
        "controller_stall_rate_wilson_95": list(
            wilson_interval(stalls, len(rows))
        ),
        "mean_net_goal_progress": sum(
            float(row["net_goal_progress"]) for row in rows
        )
        / len(rows),
        "mean_minimum_true_clearance": sum(
            float(row["minimum_true_clearance"]) for row in rows
        )
        / len(rows),
        "minimum_true_clearance": min(
            float(row["minimum_true_clearance"]) for row in rows
        ),
        "total_control_steps": total_steps,
        "mean_episode_steps": total_steps / len(rows),
        "reflex_interventions": interventions,
        "reflex_intervention_rate": interventions / total_steps,
        "reflex_backups": backups,
        "robust_recoveries": robust_recoveries,
        "robust_recovery_rate": robust_recoveries / total_steps,
        "mean_side_switches": sum(
            int(row["reflex_side_switches"]) for row in rows
        )
        / len(rows),
        "solver_rejection_rate": solver_rejection_rate,
        "solver_rejections": solver_rejections,
        "deadline_miss_rate": deadline_miss_rate,
        "deadline_misses": deadline_misses,
        "maximum_episode_p99_solve_time": maximum_p99,
        "gate": {"passed": all(checks.values()), "checks": checks},
        "interpretation_rule": (
            "The timeout is derived before execution from path, detour, sensing, "
            "feedback, and recovery terms. Raw simulator trajectories are the "
            "only collision and clearance evidence. Robust-recovery steps retain "
            "physical filtering but are not counted as robust-feasible."
        ),
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["episode"]))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)


def run_liveness_development(
    config: LivenessDevelopmentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LivenessDevelopmentConfig()
    validate_liveness_prerequisite(cfg.prerequisite_summary)
    timeout = derive_physics_timeout(cfg.timeout_inputs)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = [
        {"condition": condition, "max_steps": timeout.max_steps}
        for condition in deterministic_scenario_grid(cfg.episodes, seed=cfg.seed)
    ]
    rows: list[dict[str, Any]] = []
    if cfg.workers == 1:
        for completed, payload in enumerate(payloads, start=1):
            rows.append(_run_condition(payload))
            print(f"[liveness-development] completed {completed}/{cfg.episodes}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=context) as executor:
            futures = [executor.submit(_run_condition, payload) for payload in payloads]
            for completed, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                print(
                    f"[liveness-development] completed {completed}/{cfg.episodes}",
                    flush=True,
                )
    _write_rows(output_dir / "episodes.csv", rows)
    summary = summarize_liveness_development_rows(rows, cfg, timeout)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_liveness_development(), indent=2))
