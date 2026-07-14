"""Paired ablation for timeout-profile recovery and measured velocity feedback."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import csv
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .remediation_benchmark import deterministic_scenario_grid


@dataclass(frozen=True, slots=True)
class TimeoutRecoveryVariant:
    name: str
    profile_recovery: bool
    observed_velocity: bool


VARIANTS: tuple[TimeoutRecoveryVariant, ...] = (
    TimeoutRecoveryVariant("sticky_command_velocity", False, False),
    TimeoutRecoveryVariant("recovery_only", True, False),
    TimeoutRecoveryVariant("observed_velocity_only", False, True),
    TimeoutRecoveryVariant("recovery_observed_velocity", True, True),
)


@dataclass(frozen=True, slots=True)
class TimeoutRecoveryBenchmarkConfig:
    episodes: int = 20
    seed: int = 20260714
    workers: int = 4
    max_steps: int = 140
    output_dir: str = "artifacts/timeout_recovery_ablation_20"
    feedback_gamma: float = 0.02
    feedback_latency: float = 0.4971896839560941
    feedback_ttc_threshold: float = 1.5
    max_collision_rate: float = 0.05
    max_solver_rejection_rate: float = 0.01
    max_deadline_miss_rate: float = 0.01
    p99_solve_time_limit: float = 0.04
    min_causal_opportunity_rate: float = 0.80
    min_safe_profile_release_rate: float = 0.90

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.workers < 1 or self.max_steps < 4:
            raise ValueError("positive episodes/workers and max_steps >= 4 are required")
        if not 0.0 < self.feedback_gamma <= 0.15:
            raise ValueError("feedback_gamma must be in (0, 0.15]")
        if self.feedback_latency < 0.0 or self.feedback_ttc_threshold <= 0.0:
            raise ValueError("feedback latency/threshold are invalid")
        for rate in (
            self.max_collision_rate,
            self.max_solver_rejection_rate,
            self.max_deadline_miss_rate,
            self.min_causal_opportunity_rate,
            self.min_safe_profile_release_rate,
        ):
            if not 0.0 <= rate <= 1.0:
                raise ValueError("gate rates must be in [0, 1]")
        if self.p99_solve_time_limit <= 0.0:
            raise ValueError("p99_solve_time_limit must be positive")


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
                optimal_decay_weight=10.0,
                feedback_ttc_threshold=float(payload["feedback_ttc_threshold"]),
                feedback_response_latency=float(payload["feedback_latency"]),
                feedback_gamma=float(payload["feedback_gamma"]),
                reject_feedback_without_causal_opportunity=True,
                safety_profile_recovery_enabled=variant.profile_recovery,
                robot_velocity_feedback_enabled=variant.observed_velocity,
                obstacle_start_offset=(condition["lateral_offset"], 0.44, 0.0),
                obstacle_velocity=(0.0, -condition["obstacle_speed"], 0.0),
                save_animation=False,
                save_plots=False,
                save_metrics=False,
                output_dir=f"/tmp/lampc-timeout-ablation-{os.getpid()}",
            )
        )
        rows.append(
            {
                **condition,
                "variant": variant.name,
                "profile_recovery": variant.profile_recovery,
                "observed_velocity": variant.observed_velocity,
                "outcome": result.outcome,
                "success": bool(result.reached_goal and not result.collision),
                "collision": result.collision,
                "steps": result.steps,
                "minimum_true_clearance": result.minimum_true_clearance,
                "final_goal_distance": result.final_goal_distance,
                "net_goal_progress": result.net_goal_progress,
                "mean_goal_progress_rate": result.mean_goal_progress_rate,
                "final_speed_scale": result.final_speed_scale,
                "final_clearance_margin": result.final_clearance_margin,
                "safety_profile_transitions": result.safety_profile_transitions,
                "mean_model_transition_error": result.mean_model_transition_error,
                "solver_rejections": result.solver_rejections,
                "deadline_misses": result.deadline_misses,
                "p99_solve_time": result.p99_solve_time,
                "feedback_causal_opportunity": result.feedback_causal_opportunity,
            }
        )
    return rows


def summarize_timeout_recovery_rows(
    rows: Sequence[Mapping[str, Any]],
    config: TimeoutRecoveryBenchmarkConfig,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant.name]
        if len(selected) != config.episodes:
            raise ValueError(f"{variant.name} does not have {config.episodes} episodes")
        total_steps = sum(int(row["steps"]) for row in selected)
        safe_rows = [row for row in selected if not bool(row["collision"])]
        summaries[variant.name] = {
            "episodes": len(selected),
            "success_rate": sum(bool(row["success"]) for row in selected)
            / len(selected),
            "collision_rate": sum(bool(row["collision"]) for row in selected)
            / len(selected),
            "outcomes": {
                outcome: sum(row["outcome"] == outcome for row in selected)
                for outcome in (
                    "goal",
                    "collision",
                    "safety_timeout",
                    "controller_stall",
                    "solver_failure",
                    "environment_truncated",
                )
            },
            "mean_final_goal_distance": sum(
                float(row["final_goal_distance"]) for row in selected
            )
            / len(selected),
            "mean_net_goal_progress": sum(
                float(row["net_goal_progress"]) for row in selected
            )
            / len(selected),
            "mean_model_transition_error": sum(
                float(row["mean_model_transition_error"]) for row in selected
            )
            / len(selected),
            "solver_rejection_rate": sum(
                int(row["solver_rejections"]) for row in selected
            )
            / total_steps,
            "deadline_miss_rate": sum(
                int(row["deadline_misses"]) for row in selected
            )
            / total_steps,
            "maximum_episode_p99_solve_time": max(
                float(row["p99_solve_time"]) for row in selected
            ),
            "nominal_profile_at_end_rate": sum(
                float(row["final_speed_scale"]) >= 0.999
                and float(row["final_clearance_margin"]) <= 1e-9
                for row in selected
            )
            / len(selected),
            "nominal_profile_at_end_safe_rate": sum(
                float(row["final_speed_scale"]) >= 0.999
                and float(row["final_clearance_margin"]) <= 1e-9
                for row in safe_rows
            )
            / len(safe_rows)
            if safe_rows
            else None,
            "causal_opportunity_rate": sum(
                bool(row["feedback_causal_opportunity"]) for row in selected
            )
            / len(selected),
        }
    target = summaries["recovery_observed_velocity"]
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
            target["causal_opportunity_rate"]
            >= config.min_causal_opportunity_rate
        ),
        "safe_profile_release": (
            target["nominal_profile_at_end_safe_rate"] is not None
            and target["nominal_profile_at_end_safe_rate"]
            >= config.min_safe_profile_release_rate
        ),
    }
    return {
        "protocol": "timeout-recovery-ablation-v1",
        "config": asdict(config),
        "methods": summaries,
        "gate": {"passed": all(checks.values()), "checks": checks},
        "action_calibration_included": False,
        "action_calibration_rationale": (
            "Current-action tracking residual is already sub-millimetre; no "
            "global command gain is applied without an axis-wise calibration grid."
        ),
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: (int(row["episode"]), str(row["variant"])))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)


def run_timeout_recovery_benchmark(
    config: TimeoutRecoveryBenchmarkConfig | None = None,
) -> dict[str, Any]:
    cfg = config or TimeoutRecoveryBenchmarkConfig()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    context = multiprocessing.get_context("spawn")
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=context) as executor:
        futures = [executor.submit(_run_condition, payload) for payload in payloads]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"[timeout-ablation] completed {completed}/{len(futures)}", flush=True)
    _write_rows(output_dir / "episodes.csv", rows)
    summary = summarize_timeout_recovery_rows(rows, cfg)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
