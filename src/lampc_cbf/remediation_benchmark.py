"""Gated ablation protocol for feedback timing and optimal-decay MPC-CBF."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import csv
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RemediationVariant:
    name: str
    online_feedback: bool
    optimal_decay_weight: float
    reject_late_updates: bool = True


VARIANTS: tuple[RemediationVariant, ...] = (
    RemediationVariant("fixed_no_decay", False, 0.0),
    RemediationVariant("fixed_optimal_decay", False, 10.0),
    RemediationVariant("async_optimal_decay_accept_late", True, 10.0, False),
    RemediationVariant("async_optimal_decay_reject_late", True, 10.0, True),
)


@dataclass(frozen=True, slots=True)
class RemediationBenchmarkConfig:
    episodes: int = 20
    seed: int = 20260714
    workers: int = 4
    max_steps: int = 140
    feedback_gamma: float = 0.02
    feedback_latency: float = 0.4971896839560941
    feedback_ttc_threshold: float = 1.5
    output_dir: str = "artifacts/remediation_ablation_20"
    max_collision_rate: float = 0.05
    max_solver_rejection_rate: float = 0.01
    max_deadline_miss_rate: float = 0.01
    p99_solve_time_limit: float = 0.04
    min_causal_opportunity_rate: float = 0.80

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.workers < 1 or self.max_steps < 4:
            raise ValueError("positive episodes/workers and max_steps >= 4 are required")
        if not 0.0 < self.feedback_gamma <= 0.15:
            raise ValueError("feedback_gamma must be in (0, 0.15]")
        if self.feedback_latency < 0.0 or self.feedback_ttc_threshold <= 0.0:
            raise ValueError("feedback latency/threshold are invalid")
        if self.p99_solve_time_limit <= 0.0:
            raise ValueError("p99_solve_time_limit must be positive")
        for value in (
            self.max_collision_rate,
            self.max_solver_rejection_rate,
            self.max_deadline_miss_rate,
            self.min_causal_opportunity_rate,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("gate rates must be in [0, 1]")


def deterministic_scenario_grid(
    episodes: int, *, seed: int = 20260714
) -> list[dict[str, Any]]:
    """Create a reproducible grid spanning obstacle speed and signed offset."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    import numpy as np

    speed_count = max(1, int(np.ceil(np.sqrt(episodes))))
    offset_count = max(1, int(np.ceil(episodes / speed_count)))
    speeds = np.linspace(0.025, 0.20, speed_count)
    magnitudes = np.linspace(0.045, 0.080, max(1, (offset_count + 1) // 2))
    offsets = [sign * value for value in magnitudes for sign in (-1.0, 1.0)]
    grid = [(float(speed), float(offset)) for speed in speeds for offset in offsets]
    return [
        {
            "episode": index,
            "seed": seed + index,
            "obstacle_speed": speed,
            "lateral_offset": offset,
        }
        for index, (speed, offset) in enumerate(grid[:episodes])
    ]


def _run_condition(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    from .smooth_dynamic_demo import SmoothDynamicConfig, run_smooth_dynamic_demo

    condition = payload["condition"]
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        result = run_smooth_dynamic_demo(
            SmoothDynamicConfig(
                delta_u_weight=2.0,
                gamma=0.15,
                seed=int(condition["seed"]),
                max_steps=int(payload["max_steps"]),
                reference_speed=0.08,
                reference_mode="straight",
                safety_mode="cbf",
                prediction_mode="velocity_tube",
                safety_reflex_enabled=True,
                optimal_decay_weight=variant.optimal_decay_weight,
                feedback_ttc_threshold=(
                    float(payload["feedback_ttc_threshold"])
                    if variant.online_feedback else None
                ),
                feedback_response_latency=float(payload["feedback_latency"]),
                feedback_gamma=(
                    float(payload["feedback_gamma"])
                    if variant.online_feedback else None
                ),
                reject_feedback_without_causal_opportunity=variant.reject_late_updates,
                obstacle_start_offset=(condition["lateral_offset"], 0.44, 0.0),
                obstacle_velocity=(0.0, -condition["obstacle_speed"], 0.0),
                save_animation=False,
                save_plots=False,
                save_metrics=False,
                output_dir=f"/tmp/lampc-remediation-{os.getpid()}",
            )
        )
        rows.append(
            {
                **condition,
                "variant": variant.name,
                "success": bool(result.reached_goal and not result.collision),
                "collision": result.collision,
                "steps": result.steps,
                "minimum_true_clearance": result.minimum_true_clearance,
                "mean_solve_time": result.mean_solve_time,
                "max_solve_time": result.max_solve_time,
                "p99_solve_time": result.p99_solve_time,
                "solver_rejections": result.solver_rejections,
                "solver_max_cpu_time_exits": result.solver_max_cpu_time_exits,
                "solver_infeasible_exits": result.solver_infeasible_exits,
                "solver_unknown_exits": result.solver_unknown_exits,
                "deadline_misses": result.deadline_misses,
                "feedback_causal_opportunity": result.feedback_causal_opportunity,
                "gamma_updates_applied": result.gamma_updates_applied,
                "feedback_updates_rejected_late": result.feedback_updates_rejected_late,
                "most_infeasible_stage": result.most_infeasible_stage,
                "infeasible_stage_events": result.infeasible_stage_events,
                "mean_optimal_decay": result.mean_optimal_decay,
            }
        )
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: (int(row["episode"]), str(row["variant"])))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)


def summarize_remediation_rows(
    rows: Sequence[Mapping[str, Any]], config: RemediationBenchmarkConfig
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant.name]
        if len(selected) != config.episodes:
            raise ValueError(f"{variant.name} does not have {config.episodes} episodes")
        steps = sum(int(row["steps"]) for row in selected)
        summaries[variant.name] = {
            "episodes": len(selected),
            "success_rate": sum(bool(row["success"]) for row in selected) / len(selected),
            "collision_rate": sum(bool(row["collision"]) for row in selected) / len(selected),
            "mean_minimum_true_clearance": sum(
                float(row["minimum_true_clearance"]) for row in selected
            ) / len(selected),
            "solver_rejection_rate": sum(
                int(row["solver_rejections"]) for row in selected
            ) / steps,
            "deadline_miss_rate": sum(
                int(row["deadline_misses"]) for row in selected
            ) / steps,
            "maximum_solve_time": max(float(row["max_solve_time"]) for row in selected),
            "maximum_episode_p99_solve_time": max(
                float(row["p99_solve_time"]) for row in selected
            ),
            "solver_max_cpu_time_exits": sum(
                int(row["solver_max_cpu_time_exits"]) for row in selected
            ),
            "solver_infeasible_exits": sum(
                int(row["solver_infeasible_exits"]) for row in selected
            ),
            "solver_unknown_exits": sum(
                int(row["solver_unknown_exits"]) for row in selected
            ),
            "causal_opportunity_rate": sum(
                bool(row["feedback_causal_opportunity"]) for row in selected
            ) / len(selected) if variant.online_feedback else None,
            "gamma_updates_applied": sum(int(row["gamma_updates_applied"]) for row in selected),
            "late_updates_rejected": sum(
                int(row["feedback_updates_rejected_late"]) for row in selected
            ),
            "infeasible_stage_events": sum(
                int(row["infeasible_stage_events"]) for row in selected
            ),
        }
    target = summaries["async_optimal_decay_reject_late"]
    checks = {
        "collision": target["collision_rate"] <= config.max_collision_rate,
        "feasibility": (
            target["solver_rejection_rate"] <= config.max_solver_rejection_rate
        ),
        "timing": (
            target["deadline_miss_rate"] <= config.max_deadline_miss_rate
            and target["maximum_episode_p99_solve_time"]
            <= config.p99_solve_time_limit
        ),
        "causal_opportunity": (
            target["causal_opportunity_rate"] >= config.min_causal_opportunity_rate
        ),
    }
    return {
        "protocol": "remediation-ablation-v1",
        "config": asdict(config),
        "methods": summaries,
        "gate": {"passed": all(checks.values()), "checks": checks},
        "feedback_source": (
            "deterministic replay gamma with measured NIM latency; not a claim of "
            "validated live NIM semantic output"
        ),
    }


def run_remediation_benchmark(
    config: RemediationBenchmarkConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RemediationBenchmarkConfig()
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payloads = [
        {
            "condition": condition,
            "max_steps": cfg.max_steps,
            "feedback_gamma": cfg.feedback_gamma,
            "feedback_latency": cfg.feedback_latency,
            "feedback_ttc_threshold": cfg.feedback_ttc_threshold,
        }
        for condition in deterministic_scenario_grid(cfg.episodes, seed=cfg.seed)
    ]
    rows: list[dict[str, Any]] = []
    if cfg.workers == 1:
        for index, payload in enumerate(payloads, 1):
            rows.extend(_run_condition(payload))
            print(f"[remediation] completed {index}/{cfg.episodes}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=context) as executor:
            futures = [executor.submit(_run_condition, payload) for payload in payloads]
            for index, future in enumerate(as_completed(futures), 1):
                rows.extend(future.result())
                print(f"[remediation] completed {index}/{cfg.episodes}", flush=True)
    _write_rows(root / "episodes.csv", rows)
    summary = summarize_remediation_rows(rows, cfg)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
